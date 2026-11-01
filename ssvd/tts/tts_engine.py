# Style-Bert-VITS2 / AIVM (AivisHub) 音声モデルの統一ローダ
# - AIVM は safetensors 互換なので config/style_vectors を展開して SBV2 ライブラリで直接推論
# - transformers 新版が BERT を fp16 でロードする問題に対応 (.float())
import json
from pathlib import Path

import numpy as np
import torch
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel

# Resolve voice-pool paths from the checkout containing this engine.  Keeping
# this in sync with evaluation.prepare_tts's workspace discovery also ensures
# that the assets loaded by Voice are the assets that prepare_tts fingerprints.
ROOT = Path(__file__).resolve().parents[2]
BERT_MODEL_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"
BERT_REVISION = "547b0e8b044fba3f9b84d0ab9f990440bd130c8b"
_BERT_LOADED = False

# SBV2 の発話ごとの INFO ログを抑制
try:
    from style_bert_vits2.logging import logger as _sbv2_logger
    _sbv2_logger.remove()
    import sys as _sys
    _sbv2_logger.add(_sys.stderr, level="WARNING")
except Exception:
    pass


def load_bert():
    global _BERT_LOADED
    if not _BERT_LOADED:
        bert_models.load_model(
            Languages.JP, BERT_MODEL_ID, revision=BERT_REVISION
        ).float()
        bert_models.load_tokenizer(
            Languages.JP, BERT_MODEL_ID, revision=BERT_REVISION
        )
        _BERT_LOADED = True


def prepare_aivm(aivm_path: Path) -> dict:
    """AIVM から config.json / style_vectors.npy を展開し、SBV2 用のパス一式と スタイル名を返す"""
    import aivmlib
    work = aivm_path.parent / aivm_path.stem
    work.mkdir(exist_ok=True)
    cfg, sv, link = work / "config.json", work / "style_vectors.npy", work / f"{aivm_path.stem}.safetensors"
    meta_path = work / "aivm_meta.json"
    if not (cfg.exists() and sv.exists() and link.exists() and meta_path.exists()):
        with aivm_path.open("rb") as f:
            meta = aivmlib.read_aivm_metadata(f)
        cfg.write_text(meta.hyper_parameters.model_dump_json(indent=2), encoding="utf-8")
        sv.write_bytes(meta.style_vectors)
        if not link.exists():
            link.symlink_to(aivm_path.resolve())
        meta_path.write_text(json.dumps({
            "speaker": meta.manifest.speakers[0].name,
            "styles": [s.name for s in meta.manifest.speakers[0].styles],
        }, ensure_ascii=False), encoding="utf-8")
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"model": link, "config": cfg, "style_vec": sv, "styles": m["styles"]}


def prepare_sbv2_dir(d: Path) -> dict:
    model = next(iter(sorted(d.glob("*.safetensors"))))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    styles = list(cfg.get("data", {}).get("style2id", {"Neutral": 0}).keys())
    return {"model": model, "config": d / "config.json", "style_vec": d / "style_vectors.npy", "styles": styles}


class Voice:
    def __init__(self, name: str, spec: dict, device: str):
        self.name = name
        p = ROOT / spec["path"]
        info = prepare_aivm(p) if spec["type"] == "aivm" else prepare_sbv2_dir(p)
        self.styles = info["styles"]
        self.default_style = self.styles[0]
        self.model = TTSModel(model_path=str(info["model"]), config_path=str(info["config"]),
                              style_vec_path=str(info["style_vec"]), device=device)
        self.model.load()
        self.sr = self.model.hyper_parameters.data.sampling_rate

    @torch.inference_mode()
    def synth(self, text: str, **kw) -> tuple[int, np.ndarray]:
        kw.setdefault("style", self.default_style)
        if kw["style"] not in self.styles:
            kw["style"] = self.default_style
        sr, audio = self.model.infer(text=text, language=Languages.JP, **kw)
        return sr, audio.astype(np.float32)


def load_pool(pool_path: Path, device: str, which: str = "all") -> dict:
    """voice_pool.json を読み込み {name: Voice} を返す。which: all|assistant|user"""
    load_bert()
    cfg = json.loads(pool_path.read_text(encoding="utf-8"))
    voices = {}
    if which in ("all", "assistant"):
        a = cfg["assistant"]
        voices[a["name"]] = Voice(a["name"], a, device)
    if which in ("all", "user"):
        for u in cfg["user"]:
            voices[u["name"]] = Voice(u["name"], u, device)
    return voices, cfg
