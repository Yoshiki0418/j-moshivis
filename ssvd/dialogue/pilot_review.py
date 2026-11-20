"""Write a private, offline human-comparison sheet for the paired dialogue pilot."""
import base64
from collections import defaultdict
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
_WARNING_LABELS = {
    "repeated_question": "前と同じ質問を繰り返していないか",
    "repeated_answer": "前と同じ回答を繰り返していないか",
    "previously_answered": "すでに答えた内容を聞き直していないか",
    "answer_in_question": "質問の中で答えを先に言っていないか",
    "spoken_form": "英字・数字を自然に読み上げられる表記か",
    "missing_task_metadata": "数えた対象や読み取った文字の記録を確認",
    "invalid_task_metadata": "タスクの記録が実際の発話に合うか",
    "topic_not_covered": "個数・文字の記録なし。会話が画像に合うか",
    "invalid_task_mode": "個数・読み取り結果の確かさを確認",
    "missing_target": "数えたもの・読んだ表示を画像で確認",
    "uncertainty_needs_visual_reason": "数や文字を確かめきれない理由が画像にあるか",
    "unknown_task_contains_answer": "判断できないとした発話と数や文字の記録が合うか",
    "count_must_be_positive_integer": "記録された個数が妥当か",
    "unspecified_count_target": "何を数えたかが明確か",
    "missing_written_or_reading": "元の表記と音声で使う読みを確認",
    "ocr_text_contains_only_symbols": "読み取った記号や文字を画像と照合",
    "ocr_needs_spoken_kana_reading": "記録された読みが自然なかな表記か",
    "missing_task_exchange": "数や文字についての質問と回答が対応するか",
    "question_does_not_ask_count": "何を数える質問か",
    "count_target_missing_from_exchange": "数えた対象を特定できるか",
    "unknown_count_needs_non_numeric_answer": "数えきれないものを断定していないか",
    "count_answer_mismatch": "答えた数と画像の個数が合うか",
    "count_answer_uses_different_counter": "助数詞が対象に合うか",
    "unsupported_or_negated_count": "答えた数とその根拠が合うか",
    "count_answer_claims_exact_total": "見えている範囲で総数を断定できるか",
    "count_answer_loses_lower_bound": "見えている最小の数として伝わるか",
    "count_answer_loses_estimate": "概数であることが伝わるか",
    "count_answer_changes_certainty": "数えられる範囲と回答の確かさが合うか",
    "question_does_not_ask_to_read": "どの文字を読む質問か",
    "ocr_question_refers_to_different_target": "質問と回答が同じ表示を指しているか",
    "unknown_ocr_needs_unreadable_answer": "判読できない範囲を正しく伝えているか",
    "ocr_answer_needs_spoken_form": "看板の英字・数字を自然に読み上げられるか",
    "ocr_answer_reading_mismatch": "画像の文字と発話での読みが合うか",
    "ocr_answer_contains_unsupported_text": "画像にない文字を補っていないか",
    "negated_ocr_answer": "読める文字を誤って否定していないか",
    "ocr_answer_loses_partial_reading": "読める部分と読めない部分を区別できているか",
    "unconfirmed_ocr_answer": "読める文字を見落としていないか",
}
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
.review-warning{display:block;border-left:3px solid #bd810d;padding:.15rem .5rem;background:#fff6dd;color:#664508;font-size:.9rem}
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


def _task_evidence(task):
    labels = {"ocr": "OCR", "counting": "カウント"}
    kind = labels.get(task.get("kind"))
    if not kind:
        return ""
    fields = (
        ("ocr_target", "読んだ表示"), ("counted_target", "数えた対象"),
        ("recognized_text", "原表記"), ("visible_fragment", "読めた部分"),
        ("spoken_reading", "発話での読み"), ("answer_count", "個数"),
        ("minimum_count", "確認できる最小の数"), ("estimated_count", "おおよその数"),
        ("reason", "判断の理由"),
    )
    values = [f'<dt>{label}</dt><dd>{escape(str(task[key]))}</dd>'
              for key, label in fields if task.get(key) is not None and task[key] != ""]
    if not values:
        return ""
    return f'<details><summary>{kind}の記録を画像と照合</summary><dl>{"".join(values)}</dl></details>'


def _dialogue(record, sample_id, side):
    dialogue = record["dialogue"]
    warnings = defaultdict(list)
    for warning in record.get("review_warnings", []):
        turn = warning.get("turn")
        if type(turn) is not int or not 1 <= turn <= len(dialogue):
            turn = None
        label = _WARNING_LABELS.get(warning["code"], "画像・会話の流れと照合")
        warnings[turn].append(f'<span class="review-warning">注意　{label}</span>')
    global_warnings = "".join(warnings.get(None, []))
    annotations = defaultdict(list)
    task = record.get("auxiliary_task") or {}
    for item in [task, *task.get("additional_tasks", [])]:
        index = item.get("answer_turn_index")
        turn = index + 1 if type(index) is int and 0 <= index < len(dialogue) else None
        annotations[turn].append(_task_evidence(item))
    turns = "".join(f'<li><b>{escape(turn["speaker"])}</b>　{escape(turn["text"])}'
                    f'{"".join(warnings.get(index, []))}{"".join(annotations.get(index, []))}</li>'
                    for index, turn in enumerate(dialogue, 1))
    return (f'{global_warnings}<ol>{turns}</ol>{"".join(annotations.get(None, []))}'
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
    flagged = []
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
        number = len(articles) + 1
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
                if candidate.get("review_warnings"):
                    flagged.append(f'<a href="#img-{number}-{side}">画像 {number}・候補 {side}</a>')
            else:
                label = {"hold": "保留", "invalid": "照合エラー", "missing": "未生成"}.get(status, "生成未完了")
                body = f'<p>{label}</p>'
            candidates.append(f'<section id="img-{number}-{side}"><h3>候補 {side}</h3>{body}</section>')
        mapping[sample_id] = row
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
    warning_links = (f'<nav aria-label="注意付きの候補">注意付きの候補 {len(flagged)} 件　'
                     + "　".join(flagged) + "</nav>") if flagged else ""
    _private_write(path, _HEADER + count + paired + warning_links + "".join(articles) + _SCRIPT)
    return path
