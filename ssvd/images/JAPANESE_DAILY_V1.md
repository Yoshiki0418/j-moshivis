# Japanese Daily v1 image collection

This is an image-only collection stage for 125,000 unique candidate files of
real Japanese everyday scenes. It does not create dialogues, audio, or
alignments. Every candidate is reviewed by the automated license, quality, and
overlap checks; all eligible images are retained, with no fixed 50,000-image
acceptance cap.

## Target mix

| Sampling bucket | Candidate allocation weight | Purpose |
|---|---:|---|
| `japanese_text_rich` | 50,000 | Shops, signs, prices, stations, menus, product displays |
| `japanese_everyday` | 50,000 | General everyday scenes across multiple Japanese cities |
| `count_friendly_everyday` | 25,000 | Scenes likely to contain multiple people or objects |

These are collection quotas, not ground-truth labels. The downstream adaptive
VLM may choose OCR, counting, or ordinary visual conversation for each image.
If it invents a count or transcription without a source annotation, the
existing `unverified_vlm_visual` review gate remains mandatory.

## Source and exclusions

- Wikimedia Commons is the initial source. Only per-file CC0, CC BY, CC BY-SA,
  or public-domain records are accepted. CC records without a license URL and
  records carrying explicit non-copyright rights restrictions are excluded.
  File page, creator, license name, license URL, source revision, source hash,
  and attribution requirement are retained.
- JaWildText and JAMMEval remain evaluation-only and are not harvested here.
- JSSODa is not counted toward the 50,000 real everyday photographs.
- DOST is not downloaded until an official training/redistribution permission
  can be recorded.
- Openverse is not the primary path because its anonymous API allowance is too
  small for a one-shot 50k collection. It can be added later as a supplemental
  source with the same per-file provenance rules.

## Pipeline

```bash
python3 ssvd/images/collect_japanese_daily_v1.py harvest
python3 ssvd/images/collect_japanese_daily_v1.py download
python3 ssvd/images/collect_japanese_daily_v1.py status
```

For the unattended metadata-to-image run used on the server:

```bash
bash ssvd/images/run_japanese_daily_v1.sh
```

Its combined log is `data/images/japanese_daily_v1/logs/collection.log`.

`harvest` stores page IDs only. `download` re-resolves the current file-level
license, downloads a 1536-pixel derivative, checks decoding/resolution/aspect
ratio/information content, and performs exact plus perceptual deduplication
against:

1. all 170,021 v3 images;
2. the targeted-v4 fingerprint cache (including 6,591 counting images);
3. images already accepted into Japanese Daily v1.

Outputs live in `data/images/japanese_daily_v1/`:

- `candidates.jsonl`: metadata-only crawl result;
- `accepted_images.jsonl`: accepted image manifest and attribution metadata;
- `ATTRIBUTION.csv`: creator, license, license URL, and source page for every
  accepted image;
- `download_audit.jsonl`: every terminal rejection/review decision and retry;
- `progress.json`: resumable progress totals for the dashboard;
- `images/<prefix>/<uid>.*`: downloaded image derivatives.

The candidate target is exactly 125,000 unique Commons file page IDs. The final
accepted count is intentionally not fixed: it is however many candidates remain
after license, rights-restriction, quality, and overlap filtering.
