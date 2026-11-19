"""Write a local, read-only listening page for the backchannel pilot."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _seconds(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(0.0, result) if math.isfinite(result) else default


def _player(sample: dict, index: int, *, snippet: bool, start: float) -> str:
    kind = "snippet" if snippet else "full"
    prefix = "snippet_" if snippet else ""
    group_id = f"sample-{index}-{kind}"
    seek = 0.0 if snippet else max(0.0, start - 3.0)
    seek_label = "冒頭から再生" if snippet else "挿入の3秒前から再生"
    players = []
    for variant, key, label in (
        ("a", "original_url", "A · 元音声"),
        ("b", "augmented_url", "B · 相づち追加"),
    ):
        selected = variant == "a"
        players.append(
            f'<div class="audio-panel" id="{group_id}-{variant}"'
            f' role="tabpanel" aria-labelledby="{group_id}-tab-{variant}"'
            f'{"" if selected else " hidden"}>'
            f'<label class="sr-only" for="{group_id}-audio-{variant}">'
            f'{index + 1}件目 {"短い比較音声" if snippet else "全体音声"} {label}</label>'
            f'<audio id="{group_id}-audio-{variant}" controls preload="metadata"'
            f' src="{_escape(sample.get(prefix + key, ""))}" data-variant="{variant}">'
            'お使いのブラウザーは音声再生に対応していません。</audio>'
            '</div>'
        )
    return (
        f'<div class="player-group" data-active="a" data-seek="{seek:.6f}">'
        '<div class="ab-tabs" role="tablist" aria-label="比較する音声">'
        f'<button type="button" role="tab" id="{group_id}-tab-a"'
        f' aria-controls="{group_id}-a" aria-selected="true" data-variant="a">'
        '<span class="letter">A</span> 元音声</button>'
        f'<button type="button" role="tab" id="{group_id}-tab-b"'
        f' aria-controls="{group_id}-b" aria-selected="false" tabindex="-1" data-variant="b">'
        '<span class="letter">B</span> 相づち追加</button></div>'
        + "".join(players)
        + f'<button type="button" class="seek-button">▶ {seek_label}</button>'
        '<p class="player-status" aria-live="polite"></p></div>'
    )


def write_review(out_dir: Path, manifest: dict) -> Path:
    """Create ``index.html`` beside the manifest's relative media files.

    Audio files are only referenced, never opened or modified. The resulting page
    has no external dependencies, network API, or approval/persistence controls.
    """
    samples = manifest.get("samples", [])
    cards = []
    for index, sample in enumerate(samples):
        event = sample.get("event") or {}
        start = _seconds(event.get("start"))
        end = max(start, _seconds(event.get("end"), start))
        validation = json.dumps(
            sample.get("validation", {}), ensure_ascii=False, indent=2, default=str
        )
        title = _escape(sample.get("title") or sample.get("uid") or f"サンプル {index + 1}")
        cards.append(
            f'<article class="sample" aria-labelledby="sample-{index}-title">'
            '<div class="sample-heading">'
            f'<span class="sample-number">{index + 1:02d}</span>'
            f'<div><h2 id="sample-{index}-title">{title}</h2>'
            f'<p class="uid">{_escape(sample.get("uid", ""))}</p></div></div>'
            '<div class="sample-grid"><div class="context">'
            f'<img src="{_escape(sample.get("image_url", ""))}" alt="{title}の参照画像"'
            ' loading="lazy">'
            '<h3>対象のユーザー発話</h3>'
            f'<blockquote>{_escape(sample.get("user_text", ""))}</blockquote>'
            '</div><div class="listening">'
            '<div class="insertion"><span class="eyebrow">追加する相づち</span>'
            f'<strong class="event-text">「{_escape(event.get("text", ""))}」</strong>'
            f'<span class="event-time">{start:.2f}–{end:.2f} 秒'
            '<span class="time-hint">（全体音声の時刻）</span></span></div>'
            '<dl class="context-details"><div><dt>挿入位置の目印</dt>'
            f'<dd>{_escape(sample.get("anchor_prefix", ""))}</dd></div>'
            '<div><dt>選定理由</dt>'
            f'<dd>{_escape(sample.get("reason", ""))}</dd></div></dl>'
            '<h3>挿入前後を聴き比べる</h3>'
            '<p class="muted">同じ位置の短い音声です。再生中に A / B を切り替えられます。</p>'
            + _player(sample, index, snippet=True, start=start)
            + '<details class="full-audio"><summary>会話全体で確認する</summary>'
            '<p class="muted">全体の流れと前後の間を確認できます。</p>'
            + _player(sample, index, snippet=False, start=start)
            + '</details></div></div>'
            '<details class="validation"><summary>自動検証結果（読み取り専用）</summary>'
            '<p class="muted">数値・機械的なチェックです。自然さの判断には試聴が必要です。</p>'
            f'<pre>{_escape(validation)}</pre></details></article>'
        )

    page = _TEMPLATE.replace("__SAMPLE_COUNT__", str(len(samples)))
    # Replace this placeholder last: sample text is never treated as a template
    # or embedded inside executable JavaScript.
    page = page.replace("__SAMPLE_CARDS__", "\n".join(cards))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "index.html"
    destination.write_text(page, encoding="utf-8")
    return destination


_TEMPLATE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>相づち挿入の試聴レビュー</title>
<style>
:root{color-scheme:light;--ink:#243138;--muted:#626e74;--line:#dce3e4;--accent:#176b68;--paper:#fff;--wash:#f3f6f5}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.7 system-ui,-apple-system,"Noto Sans JP",sans-serif}
main{width:min(1120px,100%);margin:0 auto;padding:36px 24px 56px}h1,h2,h3,p{margin:0}h1{font-size:clamp(24px,4vw,34px);line-height:1.4;font-weight:650;letter-spacing:.02em}
header{margin-bottom:28px}.intro{margin-top:12px;max-width:850px}.status{display:inline-block;font-size:12px;letter-spacing:.04em;color:#5a4820;background:#f4edd9;border:1px solid #e4d7af;border-radius:6px;padding:3px 10px;margin-bottom:14px}
.guide{display:flex;flex-wrap:wrap;gap:8px 24px;margin-top:17px;padding:12px 16px;background:#e7eeeb;border-radius:8px;font-size:13px}.guide strong{color:var(--accent)}
.sample{background:var(--paper);border:1px solid var(--line);border-radius:12px;margin:20px 0;overflow:hidden}.sample-heading{display:flex;align-items:center;gap:13px;padding:21px 24px;border-bottom:1px solid var(--line)}
.sample-number{color:var(--accent);background:#edf4f1;font-size:15px;padding:5px 10px;border-radius:6px;font-variant-numeric:tabular-nums}h2{font-size:19px;line-height:1.45}h3{font-size:14px;margin:16px 0 6px}.uid{font:11px/1.6 ui-monospace,monospace;color:var(--muted);overflow-wrap:anywhere;margin-top:3px}
.sample-grid{display:grid;grid-template-columns:minmax(0,.85fr) minmax(0,1.35fr);gap:28px;padding:24px}.context img{display:block;width:100%;max-height:330px;aspect-ratio:4/3;object-fit:contain;background:#f5f5f2;border-radius:7px;border:1px solid #edf0ed}
blockquote{margin:0;padding:10px 14px;background:#f6f8f7;border-left:3px solid #c1d6ce;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}
.insertion{background:#f0f6f3;border-radius:8px;padding:12px 16px}.eyebrow{display:block;font-size:11px;color:var(--muted)}.event-text{font-size:23px;display:inline-block;margin-right:14px;overflow-wrap:anywhere}.event-time{display:inline-block;font-size:14px;white-space:nowrap;font-variant-numeric:tabular-nums}.time-hint{font-size:11px;color:var(--muted);margin-left:4px}
.context-details{font-size:12px;margin:12px 0 18px}.context-details>div{display:grid;grid-template-columns:110px minmax(0,1fr);gap:10px;margin:5px 0}dt{color:var(--muted)}dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.muted{font-size:12px;color:var(--muted);margin-bottom:10px}
.player-group{margin:10px 0 5px}.ab-tabs{display:flex;gap:6px;margin-bottom:8px}button{font:inherit;cursor:pointer;touch-action:manipulation}button:focus-visible,summary:focus-visible,audio:focus-visible{outline:3px solid #7caeb5;outline-offset:3px}
.ab-tabs button{flex:1;border:1px solid #ccd7d5;background:#fff;border-radius:7px;padding:8px 10px;font-size:13px;color:#536462;transition:background .12s}.ab-tabs button[aria-selected="true"]{color:#fff;background:var(--accent);border-color:var(--accent)}.letter{font-weight:750;display:inline-block;margin-right:7px}
audio{display:block;width:100%;height:42px}.audio-panel[hidden]{display:none}.seek-button{border:0;border-radius:5px;background:#edf2ef;color:#215955;padding:7px 10px;font-size:12px;margin-top:9px}.seek-button:hover{background:#dce9e2}.player-status{font-size:12px;color:#8f512c;margin-top:5px;min-height:0}.player-status:empty{display:none}
details{font-size:13px}summary{cursor:pointer;font-weight:550;padding:12px 0}summary::marker{color:#708c84}.full-audio{border-top:1px solid var(--line);margin-top:16px}.validation{border-top:1px solid var(--line);background:#fafcfb;padding:0 24px}.validation .muted{margin-top:4px}pre{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:0 0 18px;padding:14px;border:1px solid var(--line);border-radius:6px;background:white}
footer{font-size:12px;color:var(--muted);margin-top:24px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:700px){main{padding:22px 14px 36px}.sample-heading{padding:16px}.sample-grid{grid-template-columns:1fr;gap:19px;padding:16px}.context img{max-height:260px}.validation{padding:0 16px}.guide{display:block}.guide span{display:block;margin:3px 0}.event-text{font-size:21px}.context-details>div{grid-template-columns:92px minmax(0,1fr);gap:8px}}
</style>
</head>
<body>
<main>
<header>
<span class="status">人による試聴待ち · 学習投入なし</span>
<h1>相づち挿入の試聴レビュー</h1>
<p class="intro">__SAMPLE_COUNT__ 件の試作音声を、元音声と聴き比べます。相づちの位置・自然さ・ユーザー発話への重なりを確認してください。原本は変更していません。</p>
<div class="guide"><span><strong>左：アシスタント ／ 右：ユーザー</strong></span><span>ヘッドホン推奨</span><span>A / B 切替時は再生位置を引き継ぎます</span></div>
</header>
__SAMPLE_CARDS__
<footer>このページは試聴用プレビューです。承認・保存・学習投入の操作はありません。自動検証結果だけでは試聴済みとは扱いません。</footer>
<noscript><p>JavaScript を有効にすると A / B の切り替えと指定位置からの再生ができます。</p></noscript>
</main>
<script>
"use strict";
const allAudio = Array.from(document.querySelectorAll("audio"));
// Only one recording may play, including recordings in another sample.
document.addEventListener("play", (event) => {
  if (!(event.target instanceof HTMLAudioElement)) return;
  allAudio.forEach((audio) => { if (audio !== event.target) audio.pause(); });
}, true);

function metadata(audio) {
  if (audio.readyState >= 1) return Promise.resolve();
  if (audio.error) return Promise.reject(new Error("audio unavailable"));
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      audio.removeEventListener("loadedmetadata", loaded);
      audio.removeEventListener("error", failed);
    };
    const loaded = () => { cleanup(); resolve(); };
    const failed = () => { cleanup(); reject(new Error("audio unavailable")); };
    const timer = setTimeout(failed, 15000);
    audio.addEventListener("loadedmetadata", loaded);
    audio.addEventListener("error", failed);
    if (audio.networkState === HTMLMediaElement.NETWORK_EMPTY) audio.load();
  });
}

document.querySelectorAll(".player-group").forEach((group) => {
  const tabs = Array.from(group.querySelectorAll('[role="tab"]'));
  const status = group.querySelector(".player-status");
  let operation = 0;
  const activeAudio = () => group.querySelector('audio[data-variant="' + group.dataset.active + '"]');

  async function seek(audio, position, resume, token) {
    try {
      await metadata(audio);
      if (token !== operation) return;
      const duration = Number.isFinite(audio.duration) ? audio.duration : position;
      audio.currentTime = Math.max(0, Math.min(position, duration));
      if (resume) await audio.play();
    } catch (error) {
      if (token !== operation || error.name === "AbortError") return;
      status.textContent = audio.error
        ? "音声を読み込めません。音声ファイルの配置を確認してください。"
        : "音声の再生ボタンを押して再開してください。";
    }
  }

  async function select(tab) {
    if (tab.dataset.variant === group.dataset.active) return;
    const previous = activeAudio();
    const position = previous.currentTime || 0;
    const resume = !previous.paused && !previous.ended;
    previous.pause();
    const token = ++operation;
    group.dataset.active = tab.dataset.variant;
    status.textContent = "";
    tabs.forEach((button) => {
      const selected = button === tab;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
      document.getElementById(button.getAttribute("aria-controls")).hidden = !selected;
    });
    await seek(activeAudio(), position, resume, token);
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? tabs[0]
        : event.key === "End" ? tabs[tabs.length - 1]
        : tabs[(index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length];
      next.focus();
      select(next);
    });
  });
  group.querySelector(".seek-button").addEventListener("click", () => {
    const audio = activeAudio();
    allAudio.forEach((other) => other.pause());
    status.textContent = "";
    seek(audio, Number(group.dataset.seek), true, ++operation);
  });
  group.querySelectorAll("audio").forEach((audio) => {
    audio.addEventListener("error", () => {
      if (audio === activeAudio()) status.textContent = "音声を読み込めません。音声ファイルの配置を確認してください。";
    });
  });
});
</script>
</body>
</html>
'''
