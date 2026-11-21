"""Offline contract tests. Never send an API request."""

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from ssvd.dialogue.pilot_api import MODELS, OpenAIPilotAPI, PilotRequestRejected
from ssvd.dialogue.regeneration_roles import next_step


def request(model=MODELS[0], *, phase="user"):
    replies = [{"text": "空はどんな色？"}] if phase == "assistant" else []
    return next_step("synthetic", "ocr", "data:image/png;base64,AA==", replies, model=model)["request"]


def counted(tokens=1000):
    return {"object": "response.input_tokens", "input_tokens": tokens}


def generated(model, text='{"text":"青いね。"}', **overrides):
    return {
        "id": "resp_mock", "model": model, "status": "completed", "service_tier": "default",
        "usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200,
                  "input_tokens_details": {"cached_tokens": 150, "cache_write_tokens": 50},
                  "output_tokens_details": {"reasoning_tokens": 80}},
        "output": [{"type": "reasoning"}, {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": text}]}],
        **overrides,
    }


class PilotAPITests(unittest.TestCase):
    def setUp(self):
        self.api = OpenAIPilotAPI("test-token")
        guard = patch.object(self.api._opener, "open", side_effect=AssertionError("network disabled in test"))
        self.open = guard.start()
        self.addCleanup(guard.stop)

    def test_count_has_the_same_image_instructions_and_format_no_paid_tools(self):
        for phase, limit in (("user", 1024), ("assistant", 1024)):
            original = request(phase=phase)
            body = self.api.prepare({**original, "tools": [{"type": "web_search"}]})
            self.assertNotIn("tools", body)
            self.assertEqual(body["max_output_tokens"], limit)
            self.assertEqual(body["reasoning"], {"effort": "low"})
            self.assertEqual(body["prompt_cache_options"], {"mode": "explicit"})
            self.assertEqual(body["service_tier"], "default")
            self.assertFalse(body["store"])
            self.assertEqual(body["input"], original["input"])
            self.assertEqual(body["input"][0]["content"][1], {
                "type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "high"
            })
            self.assertEqual(self.api.prepare(body), body)
            with patch.object(self.api, "_post", return_value=counted()) as post:
                cost = self.api.reserve_usd(body)
            url, sent = post.call_args.args
            self.assertTrue(url.endswith("/v1/responses/input_tokens"))
            self.assertEqual(sent, {k: body[k] for k in ("model", "input", "instructions", "reasoning", "text")})
            self.assertAlmostEqual(cost, (1000 * 0.125 + limit * 0.5) / 1e6, places=12)
            with patch.object(self.api, "_post", return_value=generated(MODELS[0])) as send:
                self.api.generate(body)
            self.assertEqual(send.call_args.args[1]["input"], sent["input"])

    def test_missing_image_or_invalid_cap_is_rejected_before_billing(self):
        for phase in ("user", "assistant"):
            body = request(phase=phase)
            with self.assertRaises(ValueError):
                self.api.reserve_usd({**body, "max_output_tokens": 99999})
            body["input"][0]["content"].pop()
            with self.assertRaises(ValueError):
                self.api.reserve_usd(body)
        self.open.assert_not_called()

    def test_usage_covers_cache_writes_and_reads_and_includes_reasoning_once(self):
        for model, expected_cost in ((MODELS[0], 0.00018775), (MODELS[1], 0.00187)):
            body = request(model)
            actual = generated(model + "-2026-01-01")
            with patch.object(self.api, "_post", side_effect=[counted(), actual]):
                reserved = self.api.reserve_usd(body)
                reply, receipt = self.api.generate(body)
            self.assertEqual(reply, {"text": "青いね。"})
            self.assertEqual(receipt["usage"], actual["usage"])
            self.assertEqual(receipt["response_id"], "resp_mock")
            self.assertAlmostEqual(receipt["cost_usd"], expected_cost, places=12)
            self.assertGreaterEqual(reserved, receipt["cost_usd"])

    def test_terra_flex_uses_confirmed_rates_and_reserves_standard_ceiling(self):
        api = OpenAIPilotAPI("test-token", service_tier="flex")
        model = "gpt-6.1-terra"
        body = request(model)
        response = generated(model, service_tier="flex")
        with patch.object(api, "_post", side_effect=[counted(), response]) as send:
            reserved = api.reserve_usd(body)
            _, receipt = api.generate(body)
        self.assertEqual(send.call_args.args[1]["service_tier"], "flex")
        self.assertAlmostEqual(reserved, (1000 * 1250 + 1024 * 5000) / 1e9, places=12)
        self.assertEqual(receipt["service_tier"], "flex")
        self.assertAlmostEqual(receipt["cost_usd"], 0.000935, places=12)
        for tier in ("default", None):
            response = generated(model)
            if tier is None:
                del response["service_tier"]
            with patch.object(api, "_post", side_effect=[counted(), response]) as send:
                api.reserve_usd(body)
                with self.assertRaisesRegex(RuntimeError, "verify billing"):
                    api.generate(body)
                self.assertEqual(send.call_count, 2)
        with self.assertRaises(ValueError):
            api.prepare(request("gpt-6-luna"))
        with self.assertRaises(ValueError):
            OpenAIPilotAPI("test-token", service_tier="auto")

    def test_content_failures_still_return_a_billable_receipt(self):
        body = request()
        for response in (generated(MODELS[0], "not JSON"), generated(MODELS[0], "[]"),
                         generated(MODELS[0], status="incomplete"), generated(MODELS[0], output=[]),
                         generated(MODELS[0], output=[{"role": "assistant", "type": "message",
                                                      "content": [{"type": "refusal", "refusal": "declined"}]}])):
            with patch.object(self.api, "_post", side_effect=[counted(), response]):
                self.api.reserve_usd(body)
                reply, receipt = self.api.generate(body)
            self.assertEqual(reply, {})
            self.assertEqual(receipt["usage"], response["usage"])

    def test_unknown_usage_or_nonfinal_or_unexpected_provider_output_cannot_settle(self):
        body = request()
        for changes in ({"usage": None}, {"service_tier": "priority"}, {"model": MODELS[1]},
                        {"usage": {"input_tokens": 1001, "output_tokens": 0}},
                        {"status": "in_progress"}, {"usage": {"input_tokens": 2, "output_tokens": True}}):
            response = {**generated(MODELS[0]), **changes}
            with patch.object(self.api, "_post", side_effect=[counted(), response]):
                self.api.reserve_usd(body)
                with self.assertRaises(RuntimeError):
                    self.api.generate(body)

    def test_no_generation_without_a_fresh_exact_quote(self):
        body = request()
        with patch.object(self.api, "_post", side_effect=[counted(), generated(MODELS[0])]) as post:
            with self.assertRaises(ValueError):
                self.api.generate(body)
            self.api.reserve_usd(body)
            with self.assertRaises(ValueError):
                self.api.generate({**body, "instructions": "different"})
            self.api.generate(body)
            with self.assertRaises(ValueError):
                self.api.generate(body)
            self.assertEqual(post.call_count, 2)
        for count in (counted(True), counted(0), counted(272001), {}):
            with patch.object(self.api, "_post", return_value=count):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.api.reserve_usd(body)
                with self.assertRaises(ValueError):
                    self.api.generate(body)

    def test_http_errors_are_sanitized_and_never_retried(self):
        body = request()
        for code in (403, 429, 301, 400, 408, 409, 500):
            with patch.object(self.api, "_post", return_value=counted()):
                self.api.reserve_usd(body)
            self.open.reset_mock()
            self.open.side_effect = HTTPError("https://private", code, "private body", {}, io.BytesIO(b"private"))
            with self.assertRaises(RuntimeError) as caught:
                self.api.generate(body)
            self.assertEqual(isinstance(caught.exception, PilotRequestRejected), code in (403, 429))
            self.assertNotIn("private", str(caught.exception))
            self.assertNotIn("test-token", str(caught.exception))
            self.open.assert_called_once()
            with self.assertRaises(ValueError):
                self.api.generate(body)

    def test_ambiguous_transport_failure_consumes_quote_and_redacts_input(self):
        body = request()
        with patch.object(self.api, "_post", return_value=counted()):
            self.api.reserve_usd(body)
        self.open.side_effect = TimeoutError("private request data")
        with self.assertRaises(RuntimeError) as caught:
            self.api.generate(body)
        self.assertNotIsInstance(caught.exception, PilotRequestRejected)
        self.assertNotIn("private request data", str(caught.exception))
        with self.assertRaises(ValueError):
            self.api.generate(body)
        self.open.assert_called_once()

    def test_uses_fixed_official_endpoint_with_explicit_auth_and_json(self):
        self.open.side_effect = None
        self.open.return_value = io.BytesIO(json.dumps(counted()).encode())
        self.api.reserve_usd(request())
        http_request = self.open.call_args.args[0]
        self.assertEqual(http_request.full_url, "https://api.openai.com/v1/responses/input_tokens")
        self.assertEqual(http_request.get_method(), "POST")
        self.assertEqual(http_request.get_header("Authorization"), "Bearer test-token")
        self.assertNotIn("test-token", http_request.data.decode())
        client_id = "f8c354ab-4b54-4bf9-ad41-ed637319de80"
        self.open.return_value = io.BytesIO(json.dumps(generated(MODELS[0])).encode())
        self.api.generate(request(), request_id=client_id)
        posted = self.open.call_args.args[0]
        self.assertEqual(posted.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(posted.get_header("X-client-request-id"), client_id)
        self.assertIsNone(http_request.get_header("X-client-request-id"))

    def test_credentials_model_and_remote_images_are_rejected(self):
        for invalid in ("", "key\nrest", " key"):
            with self.assertRaises(ValueError):
                OpenAIPilotAPI(invalid)
        with self.assertRaises(ValueError):
            self.api.prepare(request("other"))
        body = request()
        body["input"][0]["content"][1]["image_url"] = "https://somewhere.invalid/photo"
        with self.assertRaises(ValueError):
            self.api.reserve_usd(body)
        self.open.assert_not_called()
