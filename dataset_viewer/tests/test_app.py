from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from aiohttp.test_utils import TestClient, TestServer

from dataset_viewer.app import MAX_LIMIT, APP_REFRESH_STATE, create_app
from ssvd.review_store import ReviewConflictError


class FakeCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.audio = root / "sample.wav"
        self.image = root / "sample.png"
        self.audio.write_bytes(b"RIFF" + bytes(range(64)))
        self.image.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(32)))
        self.last_list_kwargs: dict[str, Any] = {}
        self.refresh_calls: list[bool] = []
        self.closed = False

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "last_refresh": "2026-09-28T00:00:00+09:00",
            "last_changed": "2026-09-28T00:00:00+09:00",
            "catalog_revision": 7,
            "collections": ["visual_v3"],
            "errors": [],
        }

    def dashboard(self, collection: str | None = None, category: str | None = None) -> dict[str, Any]:
        return {
            "totals": {
                "items": 2,
                "images": 1,
                "dialogues": 2,
                "audio": 2,
                "alignments": 2,
                "audio_duration_seconds": 12.5,
                "audio_duration_known": 2,
            },
            "groups": {
                "visual": {
                    "items": 1,
                    "images": 1,
                    "dialogues": 1,
                    "audio": 1,
                    "alignments": 1,
                },
                "nonvisual": {
                    "items": 1,
                    "images": 0,
                    "dialogues": 1,
                    "audio": 1,
                    "alignments": 1,
                },
            },
            "collection": collection,
            "category": category,
        }

    def tree(self) -> dict[str, Any]:
        return {
            "collections": [
                {
                    "id": "visual_v3",
                    "label": "Visual v3",
                    "count": 1,
                    "categories": [{"id": "general", "label": "General", "count": 1}],
                }
            ]
        }

    def list_items(
        self,
        *,
        page: int = 1,
        limit: int,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.last_list_kwargs = {"page": page, "limit": limit, "offset": offset, "filters": filters or {}}
        return {
            "items": [{"token": "known", "uid": "uid-1", "category": "general"}],
            "page": page,
            "limit": limit,
            "total": 1,
            "pages": 1,
            "filters": filters or {},
        }

    def get_item(
        self, token: str, filters: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        if token != "known":
            return None
        return {
            "token": token,
            "uid": "uid-1",
            "dialogue": [{"speaker": "user", "text": "何が見えますか？"}],
            "alignment": [{"start": 0.0, "end": 1.0, "text": "人物です"}],
            "media": {"image": True, "audio": True},
            "navigation": {"previous": None, "next": None, "index": 1, "total": 1},
            "filters": filters or {},
        }

    def resolve_media(self, token: str, kind: str) -> Path:
        if token != "known":
            raise KeyError(token)
        if kind == "image":
            return self.image
        if kind == "audio":
            return self.audio
        raise KeyError(kind)

    def review_item(
        self,
        token: str,
        *,
        fingerprint: str,
        decision: str,
        note: str = "",
        reviewer: str | None = None,
    ) -> dict[str, Any]:
        if token != "known":
            raise KeyError(token)
        if fingerprint != "a" * 64:
            raise ReviewConflictError("stale dialogue")
        return {
            "ok": True,
            "uid": "uid-1",
            "content_sha256": fingerprint,
            "review_status": decision,
            "dataset_usable": decision == "approved",
            "note": note,
            "reviewer": reviewer,
        }

    def refresh(self, force: bool = False) -> dict[str, Any]:
        self.refresh_calls.append(force)
        return self.status()

    def edit_dialogue(self, token, *, fingerprint, dialogue, auxiliary_task=None, note="", reviewer=None):
        if token != "known":
            raise KeyError(token)
        if fingerprint != "a" * 64:
            raise ReviewConflictError("stale dialogue")
        return {"ok": True, "item": {"dialogue": dialogue, "review_status": "approved"}}

    def close(self) -> None:
        self.closed = True


class DatasetViewerApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_interactive_requests_notify_scanner_before_dispatch(self):
        calls = []
        self.catalog.request_interactive_pause = lambda: calls.append("pause")
        for method, path, body in (
            ("get", "/api/items/known", None),
            ("put", "/api/items/known/review", {"content_sha256": "a" * 64, "decision": "approved"}),
            ("put", "/api/items/known/dialogue", {"content_sha256": "a" * 64, "dialogue": [{"text": "修正"}]}),
        ):
            before = len(calls)
            response = await getattr(self.client, method)(path, **({"json": body} if body else {}))
            self.assertEqual(response.status, 200)
            self.assertEqual(len(calls), before + 1)
        # Status polling alone should not hold indexing indefinitely.
        before = len(calls)
        await self.client.get("/api/status")
        self.assertEqual(len(calls), before)

    async def test_slow_image_resolution_does_not_block_approval_or_health(self):
        original = self.catalog.resolve_media
        started, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            started.set()
            release.wait(3)
            return original(*args, **kwargs)
        self.catalog.resolve_media = blocked
        request = asyncio.create_task(self.client.get("/api/items/known/media/image"))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            response = await asyncio.wait_for(self.client.get("/api/health"), .5)
            self.assertEqual(response.status, 200)
            self.assertIn("app;dur=", response.headers["Server-Timing"])
            approval = await asyncio.wait_for(self.client.put("/api/items/known/review", json={
                "content_sha256": "a" * 64, "decision": "approved"}), .5)
            self.assertEqual(approval.status, 200)
        finally:
            release.set()
            image = await request
            await image.read()

    async def test_slow_catalog_queries_do_not_block_health(self):
        original = self.catalog.get_item
        started, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            started.set()
            release.wait(3)
            return original(*args, **kwargs)
        self.catalog.get_item = blocked
        request = asyncio.create_task(self.client.get("/api/items/sample-token"))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            response = await asyncio.wait_for(self.client.get("/api/health"), .5)
            self.assertEqual(response.status, 200)
        finally:
            release.set()
            await request

    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.catalog = FakeCatalog(self.root)
        app = create_app(
            repo_root=self.root,
            catalog=self.catalog,
            start_background_refresh=False,
        )
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self.tempdir.cleanup()

    async def test_health_dashboard_tree_and_status(self) -> None:
        response = await self.client.get("/api/health")
        self.assertEqual(response.status, 200)
        self.assertTrue((await response.json())["ok"])
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

        response = await self.client.get("/api/dashboard")
        dashboard = await response.json()
        self.assertEqual(dashboard["totals"]["images"], 1)
        self.assertEqual(dashboard["totals"]["dialogues"], 2)
        self.assertEqual(dashboard["totals"]["audio"], 2)
        self.assertEqual(dashboard["totals"]["audio_duration_seconds"], 12.5)
        self.assertEqual(dashboard["totals"]["audio_duration_known"], 2)
        self.assertEqual(dashboard["groups"]["visual"]["dialogues"], 1)
        self.assertEqual(dashboard["groups"]["nonvisual"]["images"], 0)
        self.assertEqual(dashboard["groups"]["nonvisual"]["dialogues"], 1)
        response = await self.client.get("/api/tree")
        self.assertEqual((await response.json())["collections"][0]["id"], "visual_v3")
        response = await self.client.get("/api/status")
        payload = await response.json()
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["catalog_revision"], 7)
        self.assertEqual(payload["last_changed"], "2026-09-28T00:00:00+09:00")
        self.assertFalse(payload["refresh"]["running"])
        self.assertEqual(payload["refresh"]["interval_seconds"], 10.0)

    async def test_static_assets_are_same_origin(self) -> None:
        response = await self.client.get("/")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "text/html")
        html = await response.text()
        self.assertIn('id="metric-total-audio"', html)
        self.assertIn('id="metric-total-audio-duration"', html)
        self.assertIn('id="visual-progress-title"', html)
        self.assertIn('id="metric-visual-dialogues"', html)
        self.assertIn('id="nonvisual-progress-title"', html)
        self.assertIn('id="metric-nonvisual-dialogues"', html)
        self.assertIn('id="metric-nonvisual-audio"', html)
        self.assertIn('id="metric-nonvisual-alignments"', html)
        response = await self.client.get("/static/app.js")
        self.assertEqual(response.status, 200)
        self.assertIn("javascript", response.content_type)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        javascript = await response.text()
        self.assertIn("const STATUS_POLL_INTERVAL_MS = 10000;", javascript)
        self.assertIn("catalogRevision", javascript)
        self.assertIn("raw.catalog_revision", javascript)
        self.assertIn("dom.metricTotalAudio.textContent", javascript)
        self.assertIn("dom.metricTotalAudioDuration.textContent", javascript)
        self.assertIn('dashboardGroupCounts(raw, "visual"', javascript)
        self.assertIn('dashboardGroupCounts(raw, "nonvisual"', javascript)

    async def test_items_caps_limit_and_preserves_filters(self) -> None:
        response = await self.client.get(
            "/api/items?page=3&limit=999&category=general&has_audio=true"
            "&review_status=pending&review_required=true&auxiliary_kind=counting"
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["limit"], MAX_LIMIT)
        self.assertEqual(self.catalog.last_list_kwargs["limit"], MAX_LIMIT)
        self.assertEqual(self.catalog.last_list_kwargs["page"], 3)
        self.assertEqual(self.catalog.last_list_kwargs["filters"]["category"], "general")
        self.assertTrue(self.catalog.last_list_kwargs["filters"]["has_audio"])
        self.assertEqual(
            self.catalog.last_list_kwargs["filters"]["review_status"], "pending"
        )
        self.assertTrue(
            self.catalog.last_list_kwargs["filters"]["review_required"]
        )
        self.assertEqual(
            self.catalog.last_list_kwargs["filters"]["auxiliary_kind"], "counting"
        )

        response = await self.client.get("/api/items?limit=0")
        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "bad_request")

    async def test_item_detail_and_structured_not_found(self) -> None:
        response = await self.client.get("/api/items/known")
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["uid"], "uid-1")

        response = await self.client.get("/api/items/unknown")
        self.assertEqual(response.status, 404)
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "not_found")

    async def test_review_requires_current_content_sha256(self) -> None:
        response = await self.client.put(
            "/api/items/known/review",
            json={
                "content_sha256": "a" * 64,
                "decision": "approved",
                "note": "画像を確認",
            },
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["review_status"], "approved")
        self.assertTrue(payload["dataset_usable"])

        response = await self.client.put(
            "/api/items/known/review",
            json={"content_sha256": "b" * 64, "decision": "approved"},
        )
        self.assertEqual(response.status, 409)

        response = await self.client.put(
            "/api/items/known/review",
            json={
                "content_sha256": "a" * 64,
                "fingerprint": "b" * 64,
                "decision": "approved",
            },
        )
        self.assertEqual(response.status, 400)

    async def test_dialogue_editor_api_validates_body_and_stale_draft(self):
        response = await self.client.put("/api/items/known/dialogue", json={
            "content_sha256": "a" * 64, "dialogue": [{"text": "修正した本文"}]})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["item"]["review_status"], "approved")
        response = await self.client.put("/api/items/known/dialogue", json={
            "content_sha256": "b" * 64, "dialogue": [{"text": "古い編集"}]})
        self.assertEqual(response.status, 409)
        response = await self.client.put("/api/items/known/dialogue", json={
            "content_sha256": "a" * 64, "dialogue": "invalid"})
        self.assertEqual(response.status, 400)

    async def test_registered_media_supports_range(self) -> None:
        response = await self.client.get("/api/items/known/media/image")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "image/png")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        await response.read()

        response = await self.client.get(
            "/api/items/known/media/audio", headers={"Range": "bytes=4-11"}
        )
        self.assertEqual(response.status, 206)
        self.assertEqual(response.headers["Accept-Ranges"], "bytes")
        self.assertEqual(response.headers["Content-Range"], "bytes 4-11/68")
        self.assertEqual(await response.read(), bytes(range(8)))

        response = await self.client.get("/api/items/known/media/alignment")
        self.assertEqual(response.status, 404)

    async def test_refresh_is_background_and_force_is_forwarded(self) -> None:
        response = await self.client.post("/api/refresh", json={"force": True})
        self.assertEqual(response.status, 202)
        self.assertTrue((await response.json())["accepted"])
        for _ in range(100):
            if self.catalog.refresh_calls:
                break
            await asyncio.sleep(0.001)
        self.assertEqual(self.catalog.refresh_calls, [True])

    async def test_background_refresh_runs_at_startup_and_periodically(self) -> None:
        catalog = FakeCatalog(self.root)
        app = create_app(
            repo_root=self.root,
            catalog=catalog,
            start_background_refresh=True,
            refresh_interval=0.01,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for _ in range(200):
                if len(catalog.refresh_calls) >= 2:
                    break
                await asyncio.sleep(0.001)
            self.assertGreaterEqual(len(catalog.refresh_calls), 2)
        finally:
            await client.close()

        calls_after_cleanup = len(catalog.refresh_calls)
        await asyncio.sleep(0.03)
        self.assertEqual(len(catalog.refresh_calls), calls_after_cleanup)
        self.assertTrue(catalog.closed)

    async def test_periodic_index_defers_during_review_and_resumes_when_idle(self):
        catalog = FakeCatalog(self.root)
        app = create_app(repo_root=self.root, catalog=catalog,
                         start_background_refresh=True, refresh_interval=.01)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            await client.get("/api/items/known")
            await asyncio.sleep(.02)  # Let the startup refresh finish.
            count = len(catalog.refresh_calls)
            await asyncio.sleep(.03)
            self.assertEqual(len(catalog.refresh_calls), count)
            app[APP_REFRESH_STATE].last_interaction -= 31
            await asyncio.sleep(.03)
            self.assertGreater(len(catalog.refresh_calls), count)
            app[APP_REFRESH_STATE].last_interaction = asyncio.get_running_loop().time()
            count = len(catalog.refresh_calls)
            await client.post("/api/refresh", json={"force": False})
            await asyncio.sleep(.02)
            self.assertGreater(len(catalog.refresh_calls), count)
        finally:
            await client.close()

    async def test_zero_refresh_interval_only_runs_startup_refresh(self) -> None:
        catalog = FakeCatalog(self.root)
        app = create_app(
            repo_root=self.root,
            catalog=catalog,
            start_background_refresh=True,
            refresh_interval=0,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for _ in range(100):
                if catalog.refresh_calls:
                    break
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.03)
            self.assertEqual(catalog.refresh_calls, [False])
        finally:
            await client.close()
        self.assertTrue(catalog.closed)


if __name__ == "__main__":
    unittest.main()
