# Counting and Japanese OCR image collection report

Date: 2026-09-27

The frozen v3 dataset was not modified. Everything described here is an
image-only candidate bank for a future v4 delta. No generated Japanese
dialogue or audio has been added, the number of dialogue turns is still
undecided, and the presence of a manifest does not make an image training
data.

## Current decision

- Use newly collected images rather than reusing the local COCO images.
- Freeze the image bank before deciding dialogue length or task mixture.
- After the image bank is fixed, generate new Japanese dialogue with Claude
  Sonnet 5 or a local VLM. Do not use a mechanical translation of the English
  TallyQA questions as the final training dialogue.
- Retain official count/transcription annotations as grounding evidence and
  reject generated dialogue that contradicts them.

Artifacts:

- `data/images/targeted_v4/image_pool_v1/counting_images.jsonl`
- `data/images/targeted_v4/image_pool_v1/ocr_images.jsonl`
- `data/images/targeted_v4/image_pool_v1/review_excluded_images.jsonl`
- `data/images/targeted_v4/image_pool_v1/summary.json`

Every accepted row has `training_ready=false`, `split=null`,
`dialogue_generation.generator=null`, and
`dialogue_generation.turn_count=null`.

## Counting: new Visual Genome photographs

The old COCO annotation-derived Japanese one-turn manifests remain available
for diagnostics, but they are not candidates for this v4 addition because the
underlying photographs were already present locally. The new pool uses only
Visual Genome photographs referenced by TallyQA train.

| Selection stage | Images | Rule |
|---|---:|---|
| TallyQA Visual Genome candidates | 33,367 | unique VG image IDs |
| Non-COCO origin | 6,991 | remove 26,376 rows whose official VG `coco_id` is set |
| Original-photo groups | 6,989 | merge two duplicate Flickr-photo pairs |
| v3 overlap clean | 6,980 | remove/review 9 recoloured or resized copies of v3 photographs |
| **Final image pool** | **6,591** | short side >= 300 px and aspect ratio <= 2.0 |

The final 6,591 photographs have 10,567 count facts: 3,696 from TallyQA AMT
and 6,871 from its imported Visual Genome annotations. 3,150 images have at
least one answer of two or more. These English source annotations are evidence
for later generation, not dialogue text.

The official Visual Genome `image_data.json.zip` is pinned by its archive and
content hashes. `flickr:{flickr_id}` is the primary `source_photo_uid` and
split key; the VG image ID is used only for the 83 photographs without a
Flickr ID. File SHA-256, decoded-pixel SHA-256, pHash, and dHash provide a
second independent overlap check against all 170,021 v3 images.

The earlier files below are retained as diagnostic/reference artifacts only:

- `data/images/counting/coco_counting_ja_train2017.jsonl`
- `data/images/counting/coco_counting_ja_val2017.jsonl`
- `data/images/counting/tallyqa/metadata_train.jsonl`

## Japanese OCR: new images

Decision updated: 2026-09-28.

The active image-only OCR pool now contains only 17,991 JSSODa train images.
Exact raw-file and decoded-pixel overlap with v3 is zero. No Japanese dialogue
has been generated from this pool.

| Dataset | Images | Type | Intended role |
|---|---:|---|---|
| JSSODa train | 17,991 | synthetic full pages | auxiliary candidate only |
| JSSODa validation | 2,246 | synthetic full pages | held out; not in the pool |

JSSODa's whitespace-stripped transcript length is 115--1,792 characters with a
median of 701; zero records are at most 100 characters. It contains synthetic
long documents rather than real shop signs. Because the public manifest has no
line/word geometry or wrapping metadata, derived short crops cannot be given
reliable exact targets. JSSODa will therefore be used only as an auxiliary
long-document/layout source and must not dominate the OCR mixture.

JPSC1400, NDL one-line, NDL PDM OCR part 1/2, CODH Modern Magazine and 37,244
derived NDL/CODH crops were removed from the active manifest and local data.
They focused on single characters or historical/document text rather than the
desired modern scene-text capability.

The replacement target is real 2--24-character Japanese text: shop names,
telephone numbers, prices, opening hours, station/facility names and guidance
words. DOST's official training split is the leading candidate because it was
captured in Osaka shopping streets and provides text regions and transcripts.
Its use/redistribution terms must be confirmed before collection; adjacent
video frames must be grouped and subsampled by sequence.

## Leakage and licensing rules

- Split by original photograph or scene-text video sequence; never split
  related rows, regions, or adjacent frames independently.
- Keep TallyQA test, COCO val count evaluation, JSSODa validation, JaWildText,
  JAMMEval, and the existing JA-VG clean suite out of training.
- Treat near-duplicate OCR hash matches as manual-review candidates because
  sparse black text on white backgrounds can cause perceptual-hash false
  positives.
- Preserve source revision/archive hashes, original identifiers and URLs,
  attribution, license, and crop/other modification notes in every derived
  manifest or its pinned source registry.
- TallyQA annotations are Apache-2.0; that does not by itself relicense source
  images. Visual Genome attribution and original Flickr identifiers are kept.
- JSSODa is CC BY 4.0. Confirm DOST's conditions before adding its files or
  derived crops.

## Remaining before dialogue generation

1. Confirm DOST training-data use and derived-data redistribution conditions.
2. Collect DOST training scenes, subsample near-identical frames by sequence,
   and retain exact region transcripts and provenance.
3. Audit capability balance across shop names, phone numbers, prices, hours,
   facilities/stations and guidance words; exclude private phone numbers.
4. Freeze image/sequence-level train/validation groups without choosing dialogue
   turn count yet.
5. Compare small Sonnet 5 and local-VLM generation pilots on the same frozen
   images, then decide turn count and dialogue mixture.
6. Validate every numerical answer or quoted OCR string against its canonical
   annotation before TTS and alignment.
