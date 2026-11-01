# Targeted counting and Japanese OCR data

The completed 2026-09-27 inventory, measured counts, generated manifests and
remaining pre-training work are recorded in
[`COLLECTION_REPORT_2026-09-27.md`](COLLECTION_REPORT_2026-09-27.md).

This directory documents the data sources collected for the two targeted
capabilities that are underrepresented in the generic image-dialogue corpus.
The download command is resumable:

```bash
bash ssvd/images/download_targeted_datasets.sh core
bash ssvd/images/download_targeted_datasets.sh ocr
bash ssvd/images/download_targeted_datasets.sh visual-genome
```

## Counting image bank

- **Visual Genome images referenced by TallyQA train** are the v4 candidates.
  The official Visual Genome image metadata removes every image with a COCO
  origin, and Flickr ID plus visual hashes remove alternate copies of the same
  photograph. After v3 overlap and image-quality filtering, the image-only
  bank contains 6,591 new photographs with 10,567 count facts.
- **TallyQA train annotations** are retained as numerical grounding evidence
  for attribute- and relation-conditioned counting. They are not translated
  into the final Japanese conversation. The annotation repository is
  Apache-2.0 and does not relicense its source images. Never put its
  test-simple or test-complex split into training.
- **COCO 2017 instance-derived tasks** are retained for diagnostics and held-out
  checks, but the existing local COCO photographs are no longer v4 training
  candidates.

CountBench is deliberately not collected as training data.  It is small,
license-ambiguous at the image level, and useful as a held-out benchmark.

## Japanese OCR training data

- **JSSODa** (`llm-jp/JSSODa`): 20,237 synthetic Japanese OCR images with exact
  transcriptions, horizontal/vertical direction and one-to-four-column layout;
  CC BY 4.0. Its train split (17,991 images) is the only active OCR training
  candidate. The official validation split (2,246 images) remains held out.

JPSC1400, NDL one-line, NDL PDM OCR part 1/2, CODH Modern Magazine, and their
derived crops were removed on 2026-09-28. Historical and single-character
material did not match the target capability: reading short modern text in
shops, signs and everyday scenes.

### Model-fit note for J-MoshiVis

J-MoshiVis resizes the visual input to 448 px. Across all JSSODa records, the
whitespace-stripped transcript length is 115--1,792 characters with a median
of 701; no record is 100 characters or shorter. Public metadata has no line or
word bounding boxes, so an exact short-text crop cannot be recovered safely.
JSSODa is therefore an auxiliary long-document/layout source, not the primary
source for spoken OCR answers.

The missing primary source must contain real modern scene text with exact
2--24-character transcriptions: shop names, telephone numbers, prices, opening
hours, station/facility names and guidance words. The leading candidate is the
official DOST training split (downtown Osaka). Before inclusion, confirm its
use and redistribution terms, retain source scene and text geometry, and
deduplicate or subsample neighbouring video frames by sequence. Telephone
numbers must be public business numbers or synthetic numbers, never private
residential contact data.

For counting, use the annotation-derived number to validate model-generated
dialogue. Keep one stable `source_photo_uid` for every original photograph and
split by that identifier, not by dialogue row, so alternate Visual Genome IDs
or several later conversations from one image cannot cross train/validation
boundaries.

Dialogue text has intentionally not been created at this stage. After image
collection and image-level splits are frozen, Claude Sonnet 5 or a local VLM
will generate new Japanese dialogue from the image plus canonical count/OCR
evidence. Generator choice, turn count, and mixture are still undecided; every
image-pool row is therefore marked `training_ready=false`.

JaWildText and JAMMEval CC-OCR-JA-Refined are reserved in their entirety for
evaluation.  They must not enter dialogue generation, TTS, or model training.
Do not train on `jawildtext_cropped`: it is derived from the same evaluation
benchmark. If `honmono-ocr-test` is retained for evaluation, its ICDAR MLT
portion also prevents MLT 2017/2019 train/validation from being used for
training.
The existing JA-VG evaluation images must also be excluded from TallyQA/Visual
Genome training candidates before dialogue generation.

Every derived manifest must retain the source dataset, immutable revision or
archive SHA-256, original image identifier/URL, and license.  Image files from
COCO, Visual Genome, or TallyQA should not be redistributed as if the
annotation repository's license covered them.
