# 対話テキスト → ステレオ対話音声データセット構築 (GPU並列・再開可能)
#
# 使い方 (1シャード):
#   CUDA_VISIBLE_DEVICES=1 /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python ssvd/tts/build_speech_dataset.py \
#       --dialogues data/dialogues_merged/dialogues_final.jsonl --outdir data/speech/data_stereo \
#       --shard 0 --num_shards 6
#   複数シャードの一括起動は run_build_speech.sh を参照
#
# 出力 (UIDごと):
#   <outdir>/<uid>/stereo_dialogue.wav   24kHz ステレオ (ch0=アシスタント, ch1=ユーザー)
#   <outdir>/<uid>/dialogue.json         発話ごとの [speaker, text, start, end, voice, params]
#   <outdir>/<uid>/image.jpg             対応画像 (--allow-no-image 時は省略)
# 方針:
#   アシスタント = ほのか固定。発話内容に応じて base / intonation1.3 / happy を切替 (voice_pool.json)
#   ユーザー     = 男女16話者からUIDごとに1人固定。発話ごとに抑揚・速度・スタイルを軽く揺らす
import argparse
import hashlib
import io
import json
import random
import re
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from math import gcd
from scipy.signal import resample_poly
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dataset_keys import speech_sample_uid  # noqa: E402
from tts_engine import load_pool, ROOT  # noqa: E402

SSVD_DIR = Path(__file__).resolve().parents[1]
if str(SSVD_DIR) not in sys.path:
    sys.path.insert(0, str(SSVD_DIR))

from review_store import content_fingerprint, default_review_store, image_file_sha256  # noqa: E402

TARGET_SR = 24000
EMOTIVE_RE = re.compile(r"^(わあ|わー|おお|お、|あ、|あっ|へえ|えっ|うわ|すごい|素敵|いいですね|かわいい|きれい|楽しそう|おいしそう|美味しそう|いいなあ|いいなぁ)|[!！]|すごく|とても(きれい|素敵|かわいい)|楽し(い|そう)|嬉し|素敵|かわいい|きれいです")
HEDGE_RE = re.compile(r"かもしれ|でしょうか|かなと思|ように見え|っぽい|気がします|思います")


class ReviewAssetError(RuntimeError):
    """A review-gated source cannot be snapshotted safely."""


class StaleSpeechMetadataError(ReviewAssetError):
    """Existing speech belongs to different reviewable dialogue content."""


def stable_rng(uid: str) -> random.Random:
    return random.Random(int(hashlib.sha256(uid.encode()).hexdigest()[:16], 16))


def assistant_params(text: str, policy: dict, rng: random.Random) -> tuple[str, dict]:
    if EMOTIVE_RE.search(text):
        return "happy", dict(policy["happy"])
    if HEDGE_RE.search(text):
        return "base", dict(policy["base"])
    key = "base" if rng.random() < 0.5 else "intonation"
    return key, dict(policy[key])


def user_params(voice, var: dict, rng: random.Random) -> dict:
    kw = {
        "intonation_scale": round(rng.uniform(*var["intonation_scale"]), 2),
        "length": round(rng.uniform(*var["length"]), 2),
        "sdp_ratio": round(rng.uniform(*var["sdp_ratio"]), 2),
    }
    alt = [s for s in voice.styles if s != voice.default_style]
    if alt and rng.random() < var["alt_style_prob"]:
        kw["style"] = rng.choice(alt)
        kw["style_weight"] = 1.0
    return kw


def trim_silence(audio: np.ndarray, sr: int, thresh: float = 0.01, keep_ms: int = 60) -> np.ndarray:
    """先頭・末尾の無音を除去 (ギャップは別途制御するため)"""
    idx = np.where(np.abs(audio) > thresh)[0]
    if len(idx) == 0:
        return audio
    keep = int(sr * keep_ms / 1000)
    s, e = max(0, idx[0] - keep), min(len(audio), idx[-1] + keep)
    return audio[s:e]


def sample_gap_ms(rng: random.Random) -> int:
    """日本語会話の応答間隔: 93% 沈黙(平均200ms, sd500) / 7% オーバーラップ(平均-100ms, sd80)"""
    if rng.random() < 0.93:
        g = -1
        while g < 0:
            g = rng.gauss(200, 500)
    else:
        g = rng.gauss(-100, 80)
    return int(max(-500, min(2000, g)))


def is_valid_text(t: str) -> bool:
    if not t or not t.strip() or len(t) > 300:
        return False
    if re.search(r"(.)\1{10,}", t):
        return False
    return len(re.findall(r"[a-zA-Z0-9]", t)) / max(1, len(t)) < 0.9


def _requires_review(record: Mapping[str, object]) -> bool:
    """Fail closed for every VLM-visual auxiliary record, even if malformed."""

    if record.get("review_required") is True:
        return True
    if record.get("verification_status") == "unverified_vlm_visual":
        return True
    auxiliary = record.get("auxiliary_task")
    return bool(
        isinstance(auxiliary, Mapping)
        and auxiliary.get("grounding_source") == "vlm_visual"
    )


def _review_record_from_source(
    record: Mapping[str, object],
    dialogue: list[dict],
    *,
    source_uid: str,
    image: str,
    image_sha256: str,
) -> dict:
    """Build the exact payload recoverable from generated speech metadata."""

    return {
        "uid": source_uid,
        "image": image,
        "image_sha256": image_sha256,
        "task_mode": record.get("task_mode"),
        "dialogue": dialogue,
        "auxiliary_task": record.get("auxiliary_task"),
        "verification_status": record.get("verification_status"),
        "generator": record.get("generator"),
    }


def _review_record_from_speech_meta(meta: Mapping[str, object]) -> dict:
    speaker_names = {"USER": "ユーザー", "ASSISTANT": "アシスタント"}
    dialogue = []
    segments = meta.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            speaker = speaker_names.get(segment.get("speaker"))
            text = segment.get("text")
            if speaker is not None and isinstance(text, str):
                dialogue.append({"speaker": speaker, "text": text})
    return {
        "uid": meta.get("source_uid", meta.get("uid")),
        "image": meta.get("image"),
        "image_sha256": meta.get("image_sha256"),
        "task_mode": meta.get("source_task_mode", meta.get("task_mode", "normal")),
        "dialogue": dialogue,
        "auxiliary_task": meta.get("auxiliary_task"),
        "verification_status": meta.get("verification_status"),
        "generator": meta.get("generator"),
    }


def _review_core_fingerprint(record: Mapping[str, object]) -> str:
    """Fingerprint all review fields except the source-image bytes."""

    payload = dict(record)
    payload["image_sha256"] = "<image-bytes-compared-separately>"
    return content_fingerprint(payload)


def _read_review_image(path: Path) -> tuple[bytes, str]:
    """Read and validate the exact source bytes that will be copied."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ReviewAssetError(
            f"cannot read review-gated source image {path}: {error}"
        ) from error
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    except Exception as error:
        raise ReviewAssetError(
            f"cannot decode review-gated source image {path}: {error}"
        ) from error
    return payload, hashlib.sha256(payload).hexdigest()


def _render_review_image(payload: bytes, destination: Path) -> str:
    """Render JPEG from the same bytes used by the review hash."""

    temporary = destination.with_name(f".{destination.name}.review-incomplete")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.convert("RGB").save(temporary, format="JPEG", quality=92)
        return hashlib.sha256(temporary.read_bytes()).hexdigest()
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise ReviewAssetError(
            f"cannot render review-gated image {destination}: {error}"
        ) from error


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.review-incomplete")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _refresh_existing_review_image(
    *,
    directory: Path,
    meta_path: Path,
    existing_meta: dict,
    source_record: Mapping[str, object],
    source_image_bytes: bytes,
    source_image_sha256: str,
    review_id: str,
    review_required: bool = True,
) -> str:
    """Refresh only image provenance while preserving WAV/alignment assets."""

    existing_record = _review_record_from_speech_meta(existing_meta)
    if _review_core_fingerprint(existing_record) != _review_core_fingerprint(
        source_record
    ):
        raise StaleSpeechMetadataError(
            "existing speech metadata belongs to different dialogue/UID/image-path/"
            "task/generator content; refusing to overwrite WAV or alignment"
        )

    image_path = directory / "image.jpg"
    temporary_image = image_path.with_name(f".{image_path.name}.review-incomplete")
    asset_sha256 = _render_review_image(source_image_bytes, image_path)
    needs_refresh = (
        existing_meta.get("review_required") is not review_required
        or existing_meta.get("review_id") != review_id
        or existing_meta.get("image_sha256") != source_image_sha256
        or existing_meta.get("image_asset_sha256") != asset_sha256
        or not image_path.is_file()
        or hashlib.sha256(image_path.read_bytes()).hexdigest() != asset_sha256
    )
    if not needs_refresh:
        temporary_image.unlink(missing_ok=True)
        return "skip"

    updated_meta = dict(existing_meta)
    updated_meta.update(
        {
            "review_required": review_required,
            "review_id": review_id,
            "image_sha256": source_image_sha256,
            "image_asset_sha256": asset_sha256,
        }
    )
    # Invalidate the old approval before publishing the replacement image. A
    # crash between the atomic renames cannot expose a new image under the old ID.
    _write_json_atomic(meta_path, updated_meta)
    temporary_image.replace(image_path)
    return "review_updated"


def build_one(
    rec: dict,
    voices: dict,
    cfg: dict,
    outdir: Path,
    device: str,
    allow_no_image: bool = False,
    review_store=None,
) -> str:
    source_uid = str(rec["uid"])
    uid = speech_sample_uid(rec)
    d = outdir / uid
    wav_path, meta_path = d / "stereo_dialogue.wav", d / "dialogue.json"
    dialogue = [t for t in rec["dialogue"] if t.get("text")]
    if len(dialogue) < 4 or not all(is_valid_text(t["text"]) for t in dialogue):
        return "invalid"
    image_value = rec.get("image")
    img_src = Path(image_value) if image_value else None
    review_required = _requires_review(rec)
    reviews = review_store or default_review_store(ROOT)
    decisions = reviews.latest_decisions()
    tracked = review_required or bool(rec.get("human_revision")) or any(
        str(event.get("uid")) == source_uid for event in decisions.values())
    if img_src is None:
        # p_concat 用の非画像対話は、画像を捏造せず音声と発話メタデータだけを作る。
        # 明示フラグと kind の両方を要求し、画像対話の欠損を誤って許容しない。
        if review_required:
            raise ReviewAssetError(
                f"review-gated sample {source_uid!r} has no source image"
            )
        if not (allow_no_image and rec.get("kind") == "nonvisual"):
            return "noimage"
    elif not img_src.is_file():
        if review_required:
            raise ReviewAssetError(
                f"review-gated source image is missing or not a file: {img_src}"
            )
        return "noimage"

    source_image_bytes: bytes | None = None
    source_image_sha256 = rec.get("image_sha256")
    review_id = rec.get("review_id")
    review_source_record: dict | None = None
    if tracked:
        if img_src is not None:
            source_image_bytes, source_image_sha256 = _read_review_image(img_src)
        # Fingerprint the dialogue representation that speech metadata can
        # reconstruct exactly. Generator output normally already has this shape.
        review_dialogue = [
            {"speaker": turn.get("speaker"), "text": turn.get("text")}
            for turn in dialogue
        ]
        review_source_record = _review_record_from_source(
            rec,
            review_dialogue,
            source_uid=source_uid,
            image=str(image_value) if image_value is not None else None,
            image_sha256=source_image_sha256,
        )
        review_id = content_fingerprint(review_source_record)
        event = decisions.get(review_id)
        if (review_required and (not event or event.get("decision") != "approved")) or (event and event.get("decision") == "rejected"):
            return "review_blocked"
        base = (rec.get("human_revision") or {}).get("base_fingerprint", review_id)
        active = reviews.latest_revisions().get(base)
        if active and active["fingerprint"] != review_id:
            return "review_blocked"

    if wav_path.exists() and meta_path.exists():
        if not review_required and not rec.get("human_revision"):
            return "skip"
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StaleSpeechMetadataError(
                f"cannot validate existing speech metadata {meta_path}: {error}"
            ) from error
        if not isinstance(existing_meta, dict):
            raise StaleSpeechMetadataError(
                f"existing speech metadata is not an object: {meta_path}"
            )
        if img_src is None:
            if existing_meta.get("review_id") == review_id:
                return "skip"
            raise StaleSpeechMetadataError("existing nonvisual speech belongs to different dialogue")
        assert source_image_bytes is not None
        assert isinstance(source_image_sha256, str)
        assert isinstance(review_id, str)
        assert review_source_record is not None
        return _refresh_existing_review_image(
            directory=d,
            meta_path=meta_path,
            existing_meta=existing_meta,
            source_record=review_source_record,
            source_image_bytes=source_image_bytes,
            source_image_sha256=source_image_sha256,
            review_id=review_id,
            review_required=review_required,
        )

    rng = stable_rng(uid)
    assistant = voices[cfg["assistant"]["name"]]
    pool = cfg["user"]
    user_spec = rng.choices(pool, weights=[u.get("weight", 1.0) for u in pool])[0]
    user = voices[user_spec["name"]]
    policy, var = cfg["assistant_policy"], cfg["user_variation"]

    segs = []          # (speaker, text, start_sample, audio, meta)
    cursor = int(TARGET_SR * rng.uniform(0.3, 0.8))
    for turn in dialogue:
        is_a = turn["speaker"] == "アシスタント"
        text = turn["text"]
        if is_a:
            key, kw = assistant_params(text, policy, rng)
            voice, vname = assistant, assistant.name
        else:
            key, kw = "user", user_params(user, var, rng)
            voice, vname = user, user.name
        sr, audio = voice.synth(text, **kw)
        if sr != TARGET_SR:
            g = gcd(TARGET_SR, sr)
            audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)
        audio = trim_silence(audio, TARGET_SR)
        peak = float(np.abs(audio).max()) or 1.0
        audio = audio / peak * 0.8
        if segs:
            cursor += int(TARGET_SR * sample_gap_ms(rng) / 1000)
            cursor = max(cursor, segs[-1][2] + int(TARGET_SR * 0.3))  # 前発話の開始+0.3s より前には置かない
        segs.append(("ASSISTANT" if is_a else "USER", text, cursor, audio,
                     {"voice": vname, "policy": key, "params": kw}))
        cursor += len(audio)

    total = max(s[2] + len(s[3]) for s in segs) + int(TARGET_SR * rng.uniform(0.3, 0.8))
    stereo = np.zeros((total, 2), dtype=np.float32)
    meta = []
    for spk, text, start, audio, m in segs:
        ch = 0 if spk == "ASSISTANT" else 1
        stereo[start:start + len(audio), ch] += audio
        meta.append({"speaker": spk, "text": text, "start": round(start / TARGET_SR, 3),
                     "end": round((start + len(audio)) / TARGET_SR, 3), **m})
    stereo = np.clip(stereo, -1, 1)

    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "stereo_dialogue.tmp.wav"
    sf.write(tmp, stereo, TARGET_SR, subtype="PCM_16")
    tmp.rename(wav_path)
    image_asset_sha256 = None
    if img_src is not None:
        if tracked:
            assert source_image_bytes is not None
            image_path = d / "image.jpg"
            image_asset_sha256 = _render_review_image(
                source_image_bytes, image_path
            )
            image_path.with_name(
                f".{image_path.name}.review-incomplete"
            ).replace(image_path)
        else:
            try:
                Image.open(img_src).convert("RGB").save(d / "image.jpg", quality=92)
            except Exception:
                shutil.copy(img_src, d / "image.jpg")
    metadata = {
        "uid": uid, "source_uid": source_uid,
        "task_mode": rec.get("task_mode", "normal"),
        "source_task_mode": rec.get("task_mode"),
        # Keep adaptive-task review provenance attached even when TTS runs
        # after approval. Both TTS and the training-index builder fail closed
        # for approval-required records; ordinary records remain opt-in.
        "normal_strategy": rec.get("normal_strategy"),
        "human_revision": rec.get("human_revision"),
        "dataset_collection": rec.get("dataset_collection") or (rec.get("human_revision") or {}).get("collection"),
        "auxiliary_task": rec.get("auxiliary_task"),
        "verification_status": rec.get("verification_status"),
        "review_required": review_required,
        "review_id": review_id,
        "source_fingerprint": (rec.get("human_revision") or {}).get("base_fingerprint", review_id),
        "human_certified": bool(decisions.get(review_id, {}).get("decision") == "approved"),
        "image_sha256": source_image_sha256,
        "image_asset_sha256": image_asset_sha256,
        "split_group_key": rec.get("split_group_key") or source_uid,
        "grounding": rec.get("grounding"),
        "split": rec.get("split"),
        # Preserve the source string exactly: it participates in the
        # content-bound review fingerprint created by the generator.
        "image": image_value if img_src is not None else None,
        "kind": rec.get("kind"), "generator": rec.get("generator"),
        "user_voice": user.name, "assistant_voice": assistant.name,
        "duration": round(total / TARGET_SR, 3), "segments": meta,
    }
    _write_json_atomic(meta_path, metadata)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--outdir", default="data/speech/data_stereo")
    ap.add_argument("--pool", default=str(Path(__file__).parent / "voice_pool.json"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--allow-no-image",
        action="store_true",
        help="kind=nonvisual のレコードを image なしで音声化する",
    )
    args = ap.parse_args()

    outdir = ROOT / args.outdir if not Path(args.outdir).is_absolute() else Path(args.outdir)
    recs = []
    with open(args.dialogues, encoding="utf-8") as f:
        for i, l in enumerate(f):
            # Dialogue generators append one JSON object per line.  When TTS
            # follows a live generator, the final read can observe the last
            # write before its newline is published.  Ignore only that trailing
            # partial record; the next retry pass will pick it up intact.
            if not l.endswith("\n"):
                print(
                    f"[shard {args.shard}] ignored trailing partial JSONL line {i + 1}",
                    flush=True,
                )
                break
            if l.strip() and i % args.num_shards == args.shard:
                recs.append(json.loads(l))
    if args.limit:
        recs = recs[:args.limit]
    reviews = default_review_store(ROOT)
    revisions = reviews.latest_revisions()
    if revisions:
        revised_uids = {str(event.get("uid")) for event in revisions.values()}
        effective = []
        for record in recs:
            if str(record.get("uid")) not in revised_uids:
                effective.append(record)
                continue
            image = record.get("image")
            current_hash = image_file_sha256(image) if image else None
            effective.append(reviews.resolve_record(record, image_sha256=current_hash))
        recs = effective
    # Review-gated records must re-enter build_one even when WAV/metadata exist,
    # because the source image may have changed since the earlier snapshot.
    todo = [r for r in recs if _requires_review(r) or not (
        (outdir / speech_sample_uid(r) / "stereo_dialogue.wav").exists()
        and (outdir / speech_sample_uid(r) / "dialogue.json").exists()
    )]
    # Do not load voices or create directories for unapproved visual answers.
    decisions = reviews.latest_decisions()
    approved_uids = {str(event.get("uid")) for event in decisions.values() if event.get("decision") == "approved"}
    permitted = []
    blocked = 0
    for record in todo:
        if _requires_review(record):
            image = record.get("image")
            if str(record.get("uid")) not in approved_uids or not image:
                blocked += 1
                continue
            current_hash = image_file_sha256(image)
            review_id = content_fingerprint(record, image_sha256=current_hash)
            if decisions.get(review_id, {}).get("decision") != "approved":
                blocked += 1
                continue
        permitted.append(record)
    todo = permitted
    print(f"[shard {args.shard}] {blocked} records waiting for human approval", flush=True)
    print(f"[shard {args.shard}/{args.num_shards}] {len(recs)} records, {len(todo)} to build, device={args.device}", flush=True)
    if not todo:
        return

    voices, cfg = load_pool(Path(args.pool), args.device)
    print(f"[shard {args.shard}] loaded {len(voices)} voices", flush=True)
    stats = {
        "ok": 0,
        "review_updated": 0,
        "review_blocked": 0,
        "skip": 0,
        "invalid": 0,
        "noimage": 0,
        "error": 0,
    }
    t0 = time.time()
    for i, rec in enumerate(todo, 1):
        try:
            stats[build_one(
                rec, voices, cfg, outdir, args.device,
                allow_no_image=args.allow_no_image,
                review_store=reviews,
            )] += 1
        except ReviewAssetError as e:
            stats["error"] += 1
            print(f"❌ {speech_sample_uid(rec)[:24]} review asset error: {e}", flush=True)
            raise
        except Exception as e:
            stats["error"] += 1
            print(f"⚠️ {speech_sample_uid(rec)[:24]} error: {e}", flush=True)
        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[shard {args.shard}] {i}/{len(todo)} {stats} {el/60:.1f}min ({el/i:.1f}s/dlg)", flush=True)
    print(f"[shard {args.shard}] done {stats}", flush=True)


if __name__ == "__main__":
    main()
