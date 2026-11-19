"""Write a private, offline human-comparison sheet for the paired dialogue pilot."""
import base64
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

from ssvd.dialogue.preview_regeneration import _load_source
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, content_fingerprint

_MODELS = ("gpt-6-luna", "gpt-6.1-terra")
_HEADER = """<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; form-action 'none'; base-uri 'none'">
<title>日本の日常画像 · 対話比較</title>
<style>
*{box-sizing:border-box} body{font:16px/1.6 system-ui,sans-serif;color:#152a30;background:#f6f8f9;max-width:1100px;padding:1rem;margin:auto}
article,header{background:white;border:1px solid #cdd8da;border-radius:.5rem;padding:1rem;margin-bottom:1rem}
h1,h2,h3,p{margin:.3rem 0 .8rem}.columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}
section{min-width:0;border-top:2px solid #376467} img{max-width:100%;height:auto;display:block}
figure{margin:0 0 1rem} summary,button{cursor:pointer} li{overflow-wrap:anywhere;margin-bottom:.45rem}
ol{padding-left:1.7rem} button,select,input,textarea{font:inherit} label{display:block;margin:.4rem 0} textarea{width:100%} select{max-width:100%}
@media(max-width:700px){.columns{grid-template-columns:1fr}}
</style><body><header><h1>日本の日常画像の対話比較</h1>
<p>画像を見ながら、質問と回答の全ターンを確認してください。個数や読みの誤り、読めるのに「分からない」と答えた箇所、
見えにくいものを断定した箇所、追質問や発音用の読みの不自然さを記録してください。</p>
<p>「判定を書き出す」で入力済みの評価を端末に保存できます。ページを閉じる前に保存してください。学習用データの承認は従来の管理画面で行います。</p>
<button type="button" id="export">判定を書き出す</button></header>"""

_SCRIPT = """<script>
document.getElementById('export').addEventListener('click', () => {
  const scores = [];
  for (const form of document.querySelectorAll('fieldset')) {
    const result = Object.fromEntries([...form.querySelectorAll('[name]')].map(x => [x.name, x.value]));
    if (Object.values(result).some(value => value !== '')) {
      scores.push({sample:form.dataset.sample, side:form.dataset.side, version:form.dataset.version, ...result});
    }
  }
  const file = new Blob([JSON.stringify({schema_version:1,scores}, null, 2)], {type:'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(file);
  link.download = 'pilot-human-scores.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
</script></body></html>"""


def _private_write(path, contents):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
        try:
            tmp.write(contents)
            tmp.flush()
            name.replace(path)
        finally:
            name.unlink(missing_ok=True)


def _images(url):
    # _load_source has checked the bytes against the manifest hash. Use the
    # production Japanese vision encoder's square, bicubic resize for inspection.
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))) as image:
        thumb = image.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
    with BytesIO() as preview:
        thumb.save(preview, format="PNG")
        encoded = base64.b64encode(preview.getvalue()).decode("ascii")
    return (f'<figure><img loading="lazy" width="448" height="448" src="data:image/png;base64,{encoded}" '
            'alt="学習入力サイズの画像"><figcaption>学習入力サイズ（448 × 448）</figcaption></figure>'
            '<details><summary>元画像を確認</summary>'
            f'<img loading="lazy" src="{escape(url, quote=True)}" alt="元画像"></details>')


def _dialogue(record, sample_id, side):
    turns = "".join(f'<li><b>{escape(turn["speaker"])}</b>　{escape(turn["text"])}</li>'
                    for turn in record["dialogue"])
    task = record.get("auxiliary_task") or {}
    annotations = []
    for item in [task, *task.get("additional_tasks", [])]:
        written = item.get("recognized_text") or item.get("visible_fragment")
        if written:
            annotations.append(f'<p>OCR原表記　{escape(written)}　／　読み　{escape(item.get("spoken_reading") or "未記録")}</p>')
    return (f'<ol>{turns}</ol>{"".join(annotations)}'
            f'<fieldset data-sample="{sample_id}" data-side="{side}" data-version="{content_fingerprint(record)}">'
            '<legend>人手評価</legend><label>採用判断 <select name="decision"><option value="">未評価</option>'
            '<option value="accept">無修正で使える</option><option value="edit">要修正</option>'
            '<option value="reject">採用困難</option></select></label>'
            '<label>実際に修正にかかった時間（秒） <input name="edit_seconds" type="number" min="0" step="1"></label>'
            '<label>誤りの箇所・判断の理由<textarea name="notes" rows="2"></textarea></label></fieldset>')


def write_review(root: Path, run_dir: Path, manifest: dict, outcomes: dict) -> Path:
    """Refresh the offline sheet from current untouched sources; never update formal reviews."""
    if not (root / DEFAULT_REVIEW_EVENTS).is_file():
        raise RuntimeError("Human decision history is unavailable")
    samples = manifest["samples"]
    uids = [sample["uid"] for sample in samples]
    if len(set(uids)) != len(uids):
        raise ValueError("Duplicate UID in pilot manifest")

    # Counterbalance sides across the complete frozen sample, even on partial jobs.
    order = sorted(uids, key=lambda uid: sha256(("pilot-ab-v1/" + uid).encode()).digest())
    sides = {uid: (_MODELS if index % 2 == 0 else _MODELS[::-1]) for index, uid in enumerate(order)}
    articles = []
    comparable = []
    mapping = {}
    for sample in samples:
        uid = sample["uid"]
        if any(outcomes.get((uid, model), {}).get("status") == "protected" for model in _MODELS):
            continue
        try:
            source, kind, image, base = _load_source(root, uid)
        except (ValueError, FileNotFoundError):
            # Decisions, revised source records, and vanished images retire this sample.
            continue
        if (base != sample["base_fingerprint"] or source["image_sha256"] != sample["image_sha256"]
                or kind != sample["kind"]):
            continue
        sample_id = sha256((uid + "/" + base).encode()).hexdigest()[:16]
        row = dict(uid=uid, base_fingerprint=base, models={})
        candidates = []
        for side, model in zip(("A", "B"), sides[uid]):
            outcome = outcomes.get((uid, model), {})
            status = outcome.get("status", "missing")
            candidate = outcome.get("candidate")
            if status == "pending_review":
                if (not isinstance(candidate, dict) or candidate.get("generator") != model
                        or candidate.get("uid") != uid or candidate.get("image_sha256") != source["image_sha256"]
                        or candidate.get("review_id") != content_fingerprint(candidate)):
                    status = "invalid"
            row["models"][side] = dict(model=model, status=status)
            if status == "pending_review":
                row["models"][side]["review_id"] = candidate["review_id"]
                body = _dialogue(candidate, sample_id, side)
            else:
                label = {"hold": "保留", "invalid": "照合エラー", "missing": "未生成"}.get(status, "生成未完了")
                body = f'<p>{label}</p>'
            candidates.append(f'<section><h3>候補 {side}</h3>{body}</section>')
        mapping[sample_id] = row
        number = len(articles) + 1
        if all(item["status"] == "pending_review" for item in row["models"].values()):
            comparable.append(f'<a href="#img-{number}">画像 {number}</a>')
        articles.append(f'<article id="img-{number}"><h2>画像 {number}</h2>{_images(image)}'
                        f'<div class="columns">{"".join(candidates)}</div>'
                        '<details><summary>既存のQwen対話と比較</summary>'
                        f'{_dialogue(source, sample_id, "Qwen")}</details></article>')

    run_dir.mkdir(parents=True, exist_ok=True)
    _private_write(run_dir / "blind_key.json", json.dumps(mapping, ensure_ascii=False, indent=2))
    path = run_dir / "review.html"
    count = f'<p>現在の確認対象 {len(articles)} / 抽出時 {len(samples)} 画像</p>'
    links = "　".join(comparable)
    paired = f'<nav aria-label="全文を比較できる画像">両候補の全文が揃った画像 {len(comparable)} 枚　{links}</nav>'
    _private_write(path, _HEADER + count + paired + "".join(articles) + _SCRIPT)
    return path
