# Safari Image Classifier — Plan

Revised 2026-09-15 14:53 UTC after review feedback: `min_box_area` removed, own/finetuned
vision model added, e2e-only testing, real sandbox network constraints recorded.

## Goal

Walk an SD-card directory tree of safari photos, detect + classify the animal(s) in each
image, and materialize each image into a per-label directory under `~/animal_pics`
(configurable) by **copy** (default) or **symlink** (`--link`). A GUI shows every image with
the label it received, its confidence, and the detection boxes, and lets you correct labels.

## Label taxonomy

| Label | Rule |
|---|---|
| `<species>` e.g. `lion`, `hippopotamus` | Exactly one animal, or one animal clearly dominant |
| `multiple` | ≥2 animals, none dominant |
| `landscape` | No animal detected, image is sharp |
| `junk` | No animal detected AND image fails sharpness check (blurry mess) |
| `unknown` | Animal detected but classifier confidence below threshold |

`unknown` exists so a low-confidence guess is never silently filed as a species. Nothing is
discarded.

**Dominance rule.** Each detected animal has a box area as a fraction of the frame. The
largest animal is dominant when its box area is `>= dominance_ratio` (default `1.6`) times
the next-largest animal box area. Otherwise → `multiple`.

There is **no** `min_box_area` floor. The ratio already handles the distant-impala case: a
small background animal is far below `1/1.6` of the foreground subject's area, so the
foreground animal is dominant and the photo is filed as that species. Adding an absolute
area floor would only introduce a second, redundant threshold that could silently discard a
genuinely small-but-only subject (a bird high in frame, a distant lone rhino).

## Pipeline

```
scan → decode → detect → crop → classify → (bird refinement) → decide → materialize → catalog
```

1. **scan** — recursive walk, extension allowlist, skip hidden/`.Trash`/`.thumbnails`.
   SD card is opened **read-only**; the source is never written to.
2. **decode** — Pillow (+ `pillow-heif` for HEIC, optional `rawpy` for CR2/NEF/ARW/DNG).
   EXIF orientation applied; EXIF timestamp/GPS carried into the catalog.
3. **detect** — MegaDetector v5a: boxes with class `animal | person | vehicle`. This is the
   segmentation/bounding-box stage. Weights load from the GitHub release asset, which is
   reachable, so detection is exercised for real in tests.
4. **crop** — each animal box cropped with a small margin for the classifier.
5. **classify** — our own species classifier (see "Species model" below) run per crop,
   emitting `{common_name, scientific_name, taxon_rank, confidence}`.
6. **bird refinement** — if the top label rolls up to class `Aves`, re-run the crop through
   the bird-specialist head / provider and keep the better-supported answer.
7. **decide** — dominance rule + confidence thresholds → one final label.
8. **materialize** — `~/animal_pics/<label>/<original-name>`; on collision append a short
   content-hash suffix. Copy preserves mtime; `--link` symlinks, `--hardlink` also offered.
   Writes are atomic (temp + rename).
9. **catalog** — SQLite at `~/animal_pics/.catalog.db`: source path, sha256, final label, all
   candidate labels + scores, boxes, provider + model versions, EXIF, and human overrides.

## Species model (build / finetune our own)

The classifier is ours, behind a `Classifier` protocol so backends stay swappable.

- **Architecture**: CNN/ViT backbone + linear head over a curated safari label set
  (big cats, ungulates, elephant/rhino/hippo/giraffe, primates, small carnivores, common
  raptors and other safari birds), plus a bird-specialist head for finer bird species.
- **Transfer learning** is the intended path: freeze backbone → train head → unfreeze top
  blocks at a low LR. Standard augmentation (flip, crop jitter, colour jitter), class-balanced
  sampling for the long tail, label smoothing.
- **Training subsystem**: dataset manifest format, loader, trainer with checkpoint/resume,
  eval (top-1/top-5, per-class recall, confusion matrix), calibration for honest confidence,
  and export to a versioned artifact the pipeline loads by path.
- **Sandbox constraint**: ImageNet backbone weights (download.pytorch.org, HF) and every
  standard wildlife dataset (LILA, HF) are **blocked** here. The workflow's first job is to
  find what *is* reachable via PyPI/GitHub for backbone weights and labelled imagery. If
  nothing adequate is reachable, the training subsystem is validated end-to-end on a
  generated synthetic dataset with a from-scratch small CNN — proving the whole train → eval
  → export → inference path — and real training runs on your machine, where HF and the
  datasets are reachable, via `animal-classifier train`.
- Crops from MegaDetector make good training input, so the tool can bootstrap its own
  training data from your labelled output directories (`animal-classifier export-trainset`).

## Birds

**Merlin has no public API.** Merlin Photo ID runs on-device in the app; the public eBird
API 2.0 covers taxonomy/observations/hotspots, not image identification. iNaturalist's
vision endpoint is access-restricted. Both eBird and iNaturalist are additionally blocked
from this sandbox.

Bird handling is therefore a pluggable provider interface:

- `own_bird_head` (default, offline) — our bird-specialist classifier head.
- `ebird_enrich` (online, free key, optional) — not a classifier: canonicalizes the predicted
  name against eBird taxonomy and down-ranks species that do not occur at the photo's EXIF
  GPS location. The genuinely useful Merlin-adjacent integration; wired but disabled by
  default and unreachable from this sandbox.
- `hosted_bird_api` (online, paid) — stub that fails with a clear "needs credentials" error.

## Idempotency & scale

- Keyed on sha256 of content → re-runs skip processed images; a duplicate photo is classified
  once. `--reclassify` forces re-inference; `--dry-run` plans without writing.
- Decode workers separate from batched inference; resumable after interruption.

## GUI

Local web app (FastAPI + vanilla JS), `animal-classifier gui` → `http://127.0.0.1:8765`.

- Thumbnail grid grouped by label, per-label counts in a sidebar.
- Detection boxes overlaid, top-3 candidates with confidences.
- Filter by label / confidence band / date; full-size viewer.
- **Re-tag**: choose the correct label → file moves to the right folder, override recorded in
  the catalog so later runs respect the correction (and it becomes training data).
- Live progress while a classify run is in flight.

## Layout

```
animal_classifier/
  cli.py                 # typer CLI: classify | gui | train | eval | export-trainset | verify
  config.py              # TOML + env + flag layering
  scan.py                # recursive scan, dedupe by hash
  images.py              # decode, EXIF, blur (variance-of-Laplacian)
  detect/                # base.py, megadetector.py, stub.py
  classify/              # base.py, own_model.py, birds/, stub.py
  training/              # manifest, dataset, transforms, trainer, eval, export
  decide.py              # dominance rule, thresholds, taxonomy
  materialize.py         # copy / symlink / hardlink, atomic, collisions
  catalog.py             # SQLite schema, queries, overrides
  gui/                   # FastAPI app + static assets
tests/e2e/               # end-to-end tests only
```

## Testing — e2e only

No unit tests. Every test drives a real entry point and asserts observable outcomes:

- `classify` over a fixture SD-card tree → assert exact output tree, symlink vs copy modes,
  collision naming, catalog rows, resume/idempotency on a second run.
- Dominance / `multiple` / `landscape` / `junk` / `unknown` outcomes asserted through the CLI
  on fixtures built to trigger each, using a scripted stub detector for deterministic boxes,
  plus at least one run against **real MegaDetector weights**.
- `train` → `eval` → `export` → `classify` with the exported model, on a synthetic dataset.
- GUI via FastAPI TestClient / Playwright: grid renders, filter works, re-tag moves the file
  and updates the catalog.

## Progress log

`docs/PROGRESS.md` — every workflow step appends a UTC-timestamped entry describing what it
did, what it verified, and what it found blocked.
