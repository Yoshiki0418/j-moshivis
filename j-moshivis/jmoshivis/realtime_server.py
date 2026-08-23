"""Browser realtime demo server for J-MoshiVis.

The original MoshiVis demo streams Opus packets. The sphn build installed in
this workspace does not expose the same read APIs as that server expects, so
this server uses a tiny raw-PCM WebSocket protocol instead:

Client -> server:
  8 + encoded image bytes
  1 + float32 little-endian mono PCM at 24 kHz
  2                 finish utterance; server feeds a short silence tail
  9                 reset streaming state

Server -> client:
  0 + utf-8 JSON     ready/session info
  1 + float32 PCM    generated mono audio at 24 kHz
  3 + utf-8 text     status message
  4 + utf-8 text     error message
  7 + gate byte + utf-8 token piece
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import sentencepiece as spm
import torch
from aiohttp import web
from huggingface_hub import hf_hub_download
from PIL import Image

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from jmoshivis.models.loaders import get_moshi_vis
from moshi.models.loaders import get_mimi


TARGET_SAMPLE_RATE = 24_000
DEFAULT_CONFIG = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
DEFAULT_CHECKPOINT = "/workspace/j-moshivis/checkpoints/step_30000.safetensors"
DEFAULT_TOKENIZER = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"
DEFAULT_STATIC = "/workspace/j-moshivis/web"
MIMI_REPO = "kyutai/moshika-vis-pytorch-bf16"
MIMI_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"


def log(level: str, message: str) -> None:
    print(f"[{level.upper():5}] {message}", flush=True)


def bytes_message(kind: int, payload: bytes = b"") -> bytes:
    return bytes([kind]) + payload


async def send_text(ws: web.WebSocketResponse, kind: int, text: str) -> None:
    await ws.send_bytes(bytes_message(kind, text.encode("utf-8")))


@dataclass
class LoadedModels:
    mimi: Any
    moshi_vis: Any
    image_embedder: Any
    tokenizer: spm.SentencePieceProcessor
    image_processor: ImageProcessor


class RealtimeState:
    def __init__(
        self,
        models: LoadedModels,
        device: str,
        dtype: torch.dtype,
        silence_after_stop: float,
    ) -> None:
        self.models = models
        self.device = device
        self.dtype = dtype
        self.silence_after_stop = silence_after_stop
        self.frame_size = int(models.mimi.sample_rate / models.mimi.frame_rate)
        self.lock = asyncio.Lock()

        # Keep streaming states alive and reset them per WebSocket session.
        self.models.mimi.streaming_forever(1)
        self.models.moshi_vis.streaming_forever(1)

    def reset_streams(self) -> None:
        self.models.mimi.reset_streaming()
        self.models.moshi_vis.reset_streaming()

    def encode_image(self, payload: bytes) -> torch.Tensor:
        image = Image.open(BytesIO(payload)).convert("RGB")
        image_tensor = self.models.image_processor.transform(image)
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.models.image_embedder(image_tensor)
        ca_src = out["cross_attention_src"]
        return ca_src.to(self.dtype)

    async def process_pcm(
        self,
        ws: web.WebSocketResponse,
        frame: np.ndarray,
        ca_src: torch.Tensor,
    ) -> None:
        if frame.size == 0:
            return
        chunk = torch.from_numpy(frame.astype(np.float32, copy=False))
        chunk = chunk.to(device=self.device)[None, None, :]

        with torch.no_grad():
            codes = self.models.mimi.encode(chunk)
            for code_idx in range(codes.shape[-1]):
                tokens, gate = self.models.moshi_vis.step(
                    codes[:, :, code_idx : code_idx + 1],
                    ca_src=ca_src,
                )
                if tokens is None:
                    continue

                audio_tokens = tokens[:, 1:, :]
                wav = self.models.mimi.decode(audio_tokens)
                pcm = wav[0, 0].detach().float().cpu().numpy().astype(np.float32)
                await ws.send_bytes(bytes_message(1, pcm.tobytes()))

                text_id = int(tokens[0, 0, 0].item())
                if text_id not in (0, 3):
                    piece = self.models.tokenizer.id_to_piece(text_id).replace("▁", " ")
                    gate_value = float(gate) if not torch.is_tensor(gate) else float(gate.detach().float().mean().item())
                    gate_byte = int(max(0.0, min(1.0, (gate_value - 0.005) / 0.016)) * 255)
                    await ws.send_bytes(bytes([7, gate_byte]) + piece.encode("utf-8"))

    async def feed_silence(
        self,
        ws: web.WebSocketResponse,
        ca_src: torch.Tensor,
        seconds: float,
    ) -> None:
        frames = int(round(seconds * TARGET_SAMPLE_RATE / self.frame_size))
        if frames <= 0:
            return
        await send_text(ws, 3, f"音声停止後の余白を生成中: {seconds:.1f}s")
        silence = np.zeros(self.frame_size, dtype=np.float32)
        for _ in range(frames):
            if ws.closed:
                return
            await self.process_pcm(ws, silence, ca_src)
            await asyncio.sleep(0)

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024)
        await ws.prepare(request)

        async with self.lock:
            self.reset_streams()
            ca_src: torch.Tensor | None = None
            remainder = np.empty(0, dtype=np.float32)

            try:
                self.models.moshi_vis.update_gen_kwargs(
                    temp_text=_maybe_float(request, "text_temperature"),
                    temp=_maybe_float(request, "audio_temperature"),
                    top_k_text=_maybe_int(request, "text_topk"),
                    top_k=_maybe_int(request, "audio_topk"),
                )

                first = await ws.receive()
                if first.type != aiohttp.WSMsgType.BINARY:
                    raise RuntimeError("最初のメッセージは画像バイナリである必要があります")
                first_data = first.data
                if not first_data or first_data[0] != 8:
                    raise RuntimeError("最初に画像を送信してください")

                ca_src = self.encode_image(first_data[1:])
                await ws.send_bytes(
                    bytes_message(
                        0,
                        json.dumps(
                            {
                                "sample_rate": TARGET_SAMPLE_RATE,
                                "frame_size": self.frame_size,
                            },
                            ensure_ascii=False,
                        ).encode("utf-8"),
                    )
                )
                await send_text(ws, 3, "画像を読み込みました。マイク入力を開始できます。")
                log("info", f"session ready; ca_src={tuple(ca_src.shape)}")

                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(str(ws.exception()))
                    if message.type != aiohttp.WSMsgType.BINARY:
                        continue

                    data = message.data
                    if not data:
                        continue

                    kind = data[0]
                    payload = data[1:]
                    started = time.time()

                    if kind == 1:
                        pcm = np.frombuffer(payload, dtype=np.float32)
                        if pcm.size == 0:
                            continue
                        if remainder.size:
                            pcm = np.concatenate([remainder, pcm])
                            remainder = np.empty(0, dtype=np.float32)
                        while pcm.size >= self.frame_size:
                            frame = pcm[: self.frame_size]
                            pcm = pcm[self.frame_size :]
                            await self.process_pcm(ws, frame, ca_src)
                        remainder = pcm.copy()
                    elif kind == 2:
                        if remainder.size:
                            padded = np.zeros(self.frame_size, dtype=np.float32)
                            padded[: remainder.size] = remainder
                            remainder = np.empty(0, dtype=np.float32)
                            await self.process_pcm(ws, padded, ca_src)
                        await self.feed_silence(ws, ca_src, self.silence_after_stop)
                        await send_text(ws, 3, "応答生成を続けました。必要ならもう一度話してください。")
                    elif kind == 8:
                        ca_src = self.encode_image(payload)
                        await send_text(ws, 3, "画像を更新しました。")
                    elif kind == 9:
                        self.reset_streams()
                        remainder = np.empty(0, dtype=np.float32)
                        await send_text(ws, 3, "会話状態をリセットしました。")

                    elapsed_ms = 1000 * (time.time() - started)
                    if elapsed_ms > 250:
                        log("info", f"handled message kind={kind} in {elapsed_ms:.1f}ms")
            except Exception as exc:  # noqa: BLE001
                log("error", str(exc))
                if not ws.closed:
                    await send_text(ws, 4, str(exc))
            finally:
                self.reset_streams()
                log("info", "session closed")

        return ws


def _maybe_float(request: web.Request, name: str) -> float | None:
    value = request.rel_url.query.get(name)
    return None if value is None or value == "" else float(value)


def _maybe_int(request: web.Request, name: str) -> int | None:
    value = request.rel_url.query.get(name)
    return None if value is None or value == "" else int(value)


def load_models(args: argparse.Namespace, device: str, dtype: torch.dtype) -> LoadedModels:
    log("info", f"loading Mimi on {device}")
    mimi_weight = hf_hub_download(repo_id=MIMI_REPO, filename=MIMI_FILE)
    mimi = get_mimi(mimi_weight, device)
    mimi.eval()

    log("info", f"loading J-MoshiVis checkpoint: {args.checkpoint}")
    config = KyuteyeConfig.from_yml(args.config)
    moshi_vis, image_embedder = get_moshi_vis(
        config,
        moshi_weight=Path(args.checkpoint),
        device=device,
        dtype=dtype,
    )
    moshi_vis.eval()
    image_embedder.eval()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer)

    image_processor = ImageProcessor(size=(args.image_size, args.image_size), device=device)
    return LoadedModels(
        mimi=mimi,
        moshi_vis=moshi_vis,
        image_embedder=image_embedder,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )


def build_app(state: RealtimeState, static_dir: Path) -> web.Application:
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "sample_rate": TARGET_SAMPLE_RATE,
                "frame_size": state.frame_size,
            }
        )

    async def index(_: web.Request) -> web.FileResponse:
        return web.FileResponse(static_dir / "index.html")

    app.router.add_get("/api/health", health)
    app.router.add_get("/", index)
    app.router.add_static("/", path=static_dir, follow_symlinks=True, name="static")
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the J-MoshiVis realtime web demo.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--static", default=DEFAULT_STATIC)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--silence-after-stop", type=float, default=8.0)
    parser.add_argument("--ssl-cert", default=None)
    parser.add_argument("--ssl-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    static_dir = Path(args.static).resolve()
    if not (static_dir / "index.html").exists():
        raise FileNotFoundError(f"static UI not found: {static_dir / 'index.html'}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    if device == "cpu":
        dtype = torch.float32

    torch.manual_seed(42)
    models = load_models(args, device, dtype)
    state = RealtimeState(
        models=models,
        device=device,
        dtype=dtype,
        silence_after_stop=args.silence_after_stop,
    )
    app = build_app(state, static_dir)

    ssl_context = None
    scheme = "http"
    if args.ssl_cert and args.ssl_key:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)
        scheme = "https"

    log("info", f"open {scheme}://{args.host}:{args.port}")
    with torch.no_grad():
        web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
