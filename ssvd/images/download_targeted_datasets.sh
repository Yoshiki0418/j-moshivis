#!/bin/bash
# 数え上げ・日本語OCR用データを公式配布元から再開可能な形で取得する。
set -euo pipefail

cd /gpu-server/yamamoto/j-moshivis || exit 1

MODE=${1:-core}
PY=.venv-data/bin/python
LOG_ROOT=data/images/targeted_download_logs
mkdir -p "$LOG_ROOT"

download() {
  local url=$1
  local dest=$2
  local expected_bytes=${3:-0}
  local can_resume=${4:-1}
  mkdir -p "$(dirname "$dest")"
  if [ "$expected_bytes" -gt 0 ] && [ -f "$dest" ] && \
      [ "$(stat -c %s "$dest")" -eq "$expected_bytes" ]; then
    echo "already complete: $dest"
    return
  fi
  if [ "$can_resume" -eq 1 ] && [ -s "$dest" ]; then
    curl -fL --retry 8 --retry-all-errors --connect-timeout 30 \
      --continue-at - --output "$dest" "$url"
  else
    local restart="${dest}.restart"
    curl -fL --retry 8 --retry-all-errors --connect-timeout 30 \
      --output "$restart" "$url"
    mv -f "$restart" "$dest"
  fi
  if [ "$expected_bytes" -gt 0 ]; then
    local actual
    actual=$(stat -c %s "$dest")
    if [ "$actual" -ne "$expected_bytes" ]; then
      echo "size mismatch: $dest expected=$expected_bytes actual=$actual" >&2
      exit 2
    fi
  fi
}

extract_zip() {
  local archive=$1
  local outdir=$2
  unzip -tq "$archive" >/dev/null
  mkdir -p "$outdir"
  unzip -q -o "$archive" -d "$outdir"
}

collect_core() {
  local coco=data/images/counting/coco2017
  local tally=data/images/counting/tallyqa
  local tally_revision=46cdc649ec79c3dcc2720ff227ad07d7ee51da6f

  download http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
    "$coco/annotations_trainval2017.zip" 252907541 0
  download "https://raw.githubusercontent.com/manoja328/TallyQA_dataset/$tally_revision/tallyqa.zip" \
    "$tally/tallyqa.zip" 5691198
  printf '%s\n' "$tally_revision" > "$tally/source_revision.txt"
  echo "f4bbac642086de4f52a3fdda2de5fa2c  $coco/annotations_trainval2017.zip" | md5sum -c -
  extract_zip "$coco/annotations_trainval2017.zip" "$coco"
  extract_zip "$tally/tallyqa.zip" "$tally"
  sha256sum \
    "$coco/annotations_trainval2017.zip" \
    "$tally/tallyqa.zip" \
    > "$LOG_ROOT/core_archives.sha256"
}

collect_ocr() {
  local jssoda=data/images/ocr/jssoda_hf

  "$PY" ssvd/images/download_hf_dataset.py llm-jp/JSSODa "$jssoda" \
    --allow-pattern README.md --allow-pattern 'data/*'
}

collect_visual_genome() {
  local vg=data/images/counting/visual_genome
  download https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip \
    "$vg/images.zip" 9731705982
  download https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip \
    "$vg/images2.zip" 5471658058
  extract_zip "$vg/images.zip" "$vg"
  extract_zip "$vg/images2.zip" "$vg"
  sha256sum "$vg/images.zip" "$vg/images2.zip" \
    > "$LOG_ROOT/visual_genome_archives.sha256"
}

case "$MODE" in
  core) collect_core ;;
  ocr) collect_ocr ;;
  visual-genome) collect_visual_genome ;;
  all)
    collect_core
    collect_ocr
    collect_visual_genome
    ;;
  *)
    echo "usage: $0 {core|ocr|visual-genome|all}" >&2
    exit 64
    ;;
esac

echo "$MODE collection complete at $(date)"
