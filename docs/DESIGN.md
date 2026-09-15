# Safari Image Classifier — Technical Design

Status: **iteration 3**, revised 2026-09-15 after `docs/design-review.json` iteration 2 (verdict
`CHANGES_REQUESTED`, 1 HIGH / 13 MEDIUM / 6 NIT). Every finding of both review rounds is resolved;
§14 maps iteration 1 and §15 maps iteration 2 to where each was addressed. Grounded in
`docs/PLAN.md` (agreed behaviour) and `docs/RECON.md` (what is actually reachable). Every asset
named here has been downloaded, loaded and executed in-sandbox — no blocked host is assumed
available anywhere in this document.

Facts re-measured for **this** iteration (iteration 3), used by §11.1's now data-driven frozen lists:

| Measurement | Result | How |
|---|---|---|
| Multi-animal COCO images, val bucket, by GT area ratio | ratio **> 3.0 → 22**; ratio **> 4.0 → 19**; ratio **< 1.3 → 29** | `instances_val2017.json`, non-crowd, `split_for(file_name)` |
| `ceil(0.8 × len(list))` for those lengths | 22 → **18**; 19 → **16**; 29 → **24** | the E4 assertion, derived not hard-coded |
| `coco_species.json` candidates (single non-crowd animal, `area_frac ≥ 0.20`, class in E7's seven) | **24** | same |
| Dominance ground truth at the 1.6 gate | 471 multi-animal images → **235 dominant / 236 multiple** | same (unchanged) |

Facts measured for iteration 2 and re-confirmed by the iteration-2 review (not inherited blindly):

| Measurement | Result | How |
|---|---|---|
| Pinned dependency set resolves | **109 packages**, `opencv-python`/`roboflow`/`sahi` absent, versions identical to the RECON-verified venv | `uv pip compile` with the §2.1 overrides, `--python-version 3.12` |
| MegaDetector loads and runs **without** `roboflow` and `sahi` | `names=['animal','person','vehicle']`, `stride=[8,16,32,64]`, forward → `(1, 25500, 8)` | import-blocking probe against `.venv` |
| `pillow-heif` encodes HEIF on py3.12 + pillow 12.3.0 | `pillow_heif 1.7.0`, `libheif 1.23.3`, `save(format="HEIF")` OK, re-open OK | encode/decode round-trip probe |
| COCO animal instances after the `iscrowd` filter | 2,700 annotations − 34 crowd = **2,666** over 1,016 images | `instances_val2017.json` |
| Sub-2px animal boxes in COCO | **7**, smallest `area_frac` = 0.00001243 | same |
| Dominance ground truth at the 1.6 gate | 471 multi-animal images → **235 dominant / 236 multiple** | same |
| Frozen-list availability in the val bucket | GT ratio > 3.0: 104 total, **22 in val**; GT ratio < 1.3: 149 total, **29 in val** | same, with §7.2's `split_for` |
| E7 7-class crop counts | **1,625 train / 349 val**; majority-class baseline 0.229, chance 0.143 | same |
| Host | 8 cores, 30 GB RAM, `torch.cuda.is_available() == False` | `nproc`, torch |

## 1. Overview

`animal-classifier` is a Python CLI + library that walks a read-only SD-card tree of safari
photographs, detects animals with MegaDetector v5a, crops each detection, classifies the crop
with **our own finetuned species model**, resolves one label per image with the dominance rule,
and materializes the image into `~/animal_pics/<label>/` by copy (default), symlink or hardlink.
Every decision is recorded in a SQLite catalog keyed by content sha256, so runs are idempotent and
resumable. A local FastAPI + vanilla-JS GUI on `127.0.0.1:8765` shows each image with its detection
boxes and label and lets the user re-tag — which moves the file to the correct label directory and
records a human override that later runs respect.

The pipeline is a straight line with one branch:

```
scan → decode → detect (MegaDetector v5a) → crop ─┬→ species head ──┐
                                                  └→ bird head ─────┴→ decide → materialize → catalog
        (no animal boxes) ────────────→ blur metric → landscape | junk
```

Two models are ours and are produced by the training subsystem in this repo:

| Artifact | Label space | Trained on (real, in-sandbox) | Backbone |
|---|---|---|---|
| `models/species.acmodel` | 10 animal classes → canonical species names | COCO val2017 animal crops from ground-truth boxes (**2,666** non-crowd instances over 1,016 images) | timm `efficientnet_b0`, ImageNet-pretrained, finetuned |
| `models/birds.acmodel` | 200 bird species | CUB-200-2011 (11,788 images, per-class dirs, boxes, official split) | timm `efficientnet_b0`, ImageNet-pretrained, finetuned |

Per-class COCO instance counts (non-crowd) are `bird 427, cow 372, sheep 354, horse 272, zebra 266,
elephant 252, giraffe 232, dog 218, cat 202, bear 71`. **`bear` (71 instances, 11 of them in the
val split) is the class that limits any per-class recall claim** — §7.4 reports per-class recall but
§11 never asserts a threshold on `bear`.

`RECON.md` §"Explicit decision on training" applies: a pretrained backbone **and** real labelled
data are both present and verified, so the synthetic-data contingency in the build spec is **not**
used as the proof of the training subsystem. Training is proven by genuinely finetuning a real
ImageNet backbone on real labelled crops and asserting the accuracy reached (E7). A from-scratch
`tinycnn` on generated synthetic data is retained only as a fast deterministic smoke test of the
train → eval → export → infer path (§7.6, E10), never as a substitute for the real run.

## 2. Technology stack (locked once approved)

| Concern | Choice | Why / constraint |
|---|---|---|
| Package + env manager | **`uv` exclusively** — `uv venv`, `uv pip install`, `uv run`, `uv lock`, `uv sync --frozen` | Never `pip`, `pip3`, `virtualenv`, `python -m venv`. |
| Python | **3.12**, pinned by a `.python-version` file containing `3.12`; `requires-python = ">=3.12,<3.14"` | The whole verified stack was resolved and executed on CPython 3.12.13; the default `python3` on this host is 3.9.25, so the pin is load-bearing, not cosmetic. |
| CLI | `typer==0.27.2` | The version installed in the RECON-verified venv. `roboflow` (the transitive dep that caps `typer<0.26`) is removed from the resolution entirely — §2.1. |
| Detection | `yolov5==7.0.14` + MegaDetector v5a checkpoint | The checkpoint is a yolov5-v7 pickle; only this distribution can unpickle it (§5.4). |
| Image I/O | `pillow==12.3.0`, **`pillow-heif==1.7.0`** (HEIC decode **and** encode), `rawpy==0.27.1` (optional `raw` extra) | `pi-heif` **cannot encode** HEIF (`KeyError: 'HEIF'`), which makes E16's fixture unbuildable; `pillow-heif` 1.7.0 / libheif 1.23.3 encodes and re-reads, verified in-sandbox on py3.12. |
| CV metric | `opencv-python-headless==5.0.0.93` | **Never `opencv-python`** — this image has no `libGL.so.1` (RECON). |
| Modelling | `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `numpy==2.5.3` | Exactly the versions in which MegaDetector loading, MegaDetector inference and both backbone loads were verified. The pypi CUDA build is the only route; this host has no CUDA device, so inference and training are CPU-only here. |
| Backbone weights | local files in `models/backbones/` | `timm.create_model(arch, pretrained=False)` + explicit `load_state_dict`. **Never `pretrained=True`** (hits blocked huggingface.co). |
| GUI backend | `fastapi==0.141.1` + `uvicorn[standard]==0.39.0` | Verified in the same resolution. |
| GUI frontend | vanilla JS + CSS served as static files | No npm, no bundler, no build step. |
| HTTP client | `httpx==0.28.1` | GUI e2e tests, and the optional eBird provider. |
| Tests | `pytest==9.1.1`, `tests/e2e` only | No unit tests anywhere. |
| Build backend | `hatchling`, `src/` layout | Already established. |

### 2.1 The exact dependency contract, and why it is shaped this way

Three transitive dependencies of `yolov5` are hostile to this environment:

1. **`opencv-python`** (via `yolov5`, `ultralytics`, `sahi`) imports `libGL.so.1`, which this image
   does not have — it cannot even import.
2. **`roboflow`** (via `yolov5`) pins `typer<0.26` in 1.4.2 and pins `opencv-python-headless==4.10.0.84`
   exactly in 1.3.8. Either version therefore drags the environment away from the verified one, and
   the two constraints cannot both be satisfied together with the verified `opencv-python-headless
   5.0.0.93` — measured: `roboflow==1.3.8` + `opencv-python-headless==5.0.0.93` is *unsatisfiable*.
3. **`sahi`** requires `opencv-python>=4.12.0.88`, re-introducing (1).

None of the three is used by our code path. **Verified**: with `roboflow` and `sahi` blocked at
import time, `import yolov5`, `yolov5.utils.augmentations.letterbox`,
`yolov5.utils.general.non_max_suppression`, `yolov5.utils.general.scale_boxes`, the
`models`/`utils` alias shim, the MegaDetector checkpoint load and a 640×640 forward pass all
succeed (`names=['animal','person','vehicle']`, output `(1, 25500, 8)`). So they are removed from
the resolution with `uv` overrides rather than worked around at runtime:

```toml
[project]
name = "animal-classifier"
requires-python = ">=3.12,<3.14"
dependencies = [
  "typer==0.27.2",
  "torch==2.14.0",
  "torchvision==0.29.0",
  "timm==1.0.29",
  "numpy==2.5.3",
  "yolov5==7.0.14",
  "opencv-python-headless==5.0.0.93",
  "pillow==12.3.0",
  "pillow-heif==1.7.0",
  "fastapi==0.141.1",
  "uvicorn[standard]==0.39.0",
  "httpx==0.28.1",
  "setuptools==80.10.2",          # yolov5 7.0.14 still imports pkg_resources, removed in setuptools 81
]

[project.optional-dependencies]
raw = ["rawpy==0.27.1"]
dev = ["pytest==9.1.1"]

[tool.uv]
override-dependencies = [
  "opencv-python; python_version < '3.0'",   # never true → requirement dropped
  "roboflow; python_version < '3.0'",
  "sahi; python_version < '3.0'",
]
```

Measured result of exactly this set (`uv pip compile --python-version 3.12` with those overrides):
**109 packages**, containing `opencv-python-headless==5.0.0.93`, `typer==0.27.2`,
`setuptools==80.10.2`, `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `pillow==12.3.0`,
`pillow-heif==1.7.0`, `numpy==2.5.3`, `ultralytics==8.4.153`, and **no `opencv-python`, no
`roboflow`, no `sahi`**. `rawpy==0.27.1` resolves cleanly as an extra.

Rules that follow, and are not optional:

- `uv.lock` is **committed**, and every command in docs, tests and CI uses **`uv sync --frozen`**
  (or `uv run --frozen`). `uv run` without `--frozen` re-syncs the venv from `pyproject.toml` +
  `uv.lock` and has been observed replacing an installed package mid-session, so an unpinned
  resolution can silently evict the verified `torch`/`timm`/`opencv-headless` build.
- Every runtime dependency lives in `pyproject.toml`. Anything installed ad hoc with
  `uv pip install` will be evicted by the next sync.
- **If a resolution ever moves `torch`, `torchvision`, `timm`, `numpy` or `yolov5`, the three RECON
  probes (`scripts/probe_md_checkpoint.py`, `scripts/probe_md_inference.py`,
  `scripts/probe_backbones.py`) must be re-run and `docs/RECON.md` updated before any other work
  continues.** The pins exist to keep the verified environment; a moved pin invalidates the
  verification, not the other way round.
- The earlier claim that the resolution *pins* `typer` to 0.25.x is **withdrawn**: it was true only
  of a resolution that includes `roboflow` 1.4.2, and this design removes `roboflow`. The CLI
  targets typer 0.27.2 and the pin is what makes that deterministic.

## 3. Configuration

Layered, highest wins: **CLI flag → environment variable (`ANIMAL_CLASSIFIER_*`) → TOML file →
built-in default**. The resolved config is dumped into the `runs` catalog row so any result can be
explained later.

TOML is read from `--config`, else `$XDG_CONFIG_HOME/animal-classifier/config.toml`, else
`~/.config/animal-classifier/config.toml`. A `--config` path that does not exist is **fatal** (the
user asked for a specific file); an absent default path is normal and silently skipped.

```toml
output_root       = "~/animal_pics"
mode              = "copy"          # copy | link | hardlink
dominance_ratio   = 1.6
min_species_confidence = 0.45
detector_confidence    = 0.20
detector_iou           = 0.45
detector_image_size    = 1280
crop_margin       = 0.08
blur_threshold    = 100.0
max_file_bytes    = 536870912       # 512 MiB — scan-time byte cap, see §5.1
species_model     = "models/species.acmodel"
bird_model        = "models/birds.acmodel"
detector_weights  = "models/md_v5a.0.0.pt"
bird_provider     = "own_bird_head"  # own_bird_head | ebird_enrich | hosted_bird_api
jobs              = 7
device            = "auto"           # auto | cpu | cuda
formats           = ["jpeg", "png", "tiff", "heic"]
```

`config.py` exposes a frozen `Config` dataclass built by `Config.resolve(...)`. Validation is total
and happens once, before any file is touched (§10.2). **There is no `min_box_area`, no
`min_box_area_frac`, no `min_animal_area`, and no absolute box-area floor under any other name.**
An unknown key in the TOML file is fatal, which is what stops such a key from ever being silently
honoured.

`max_file_bytes` is **not** a box-area floor and must not be read as one: it is a whole-file byte cap
applied by `scan.py` before anything is decoded (§5.1), it never inspects, compares or discards a
detection box, and it cannot change an image's label — an over-cap file is *skipped and recorded*,
exactly like a video. Detections inside an accepted image are never filtered by size anywhere
(`dominance_ratio` is the only size gate, §5.7 / I2).

### 3.1 `formats` — exact semantics

`formats` is the **eligible-family selector**. Each entry must be one of `jpeg | png | tiff | heic`;
any other value (including `raw`) is fatal (exit 3) with the message listing the valid set. It maps
to extensions as:

| Family | Extensions |
|---|---|
| `jpeg` | `.jpg`, `.jpeg` |
| `png` | `.png` |
| `tiff` | `.tif`, `.tiff` |
| `heic` | `.heic`, `.heif` |

A file whose family is recognised but **not listed** is skipped with the enumerated reason
`format_disabled` (distinct from `unsupported_extension`, which means the extension belongs to no
family at all). RAW is **not** a family and is governed solely by `--raw`; putting `raw` in
`formats` is the fatal error above, so the two mechanisms can never be confused. `classify` exposes
a repeatable `--formats` flag (`--formats jpeg --formats png`) which replaces the configured list
entirely rather than adding to it.

## 4. Module layout

```
src/animal_classifier/
  cli.py                 # typer app: classify | gui | train | eval | export-trainset | verify
  config.py              # layering, validation, frozen Config, split/format policy
  errors.py              # ConfigError, AssetError, DecodeError, MaterializeError, CatalogError
  scan.py                # read-only recursive walk, format policy, skip reasons
  images.py              # decode, EXIF orient/time/GPS, sha256, blur metric, crop
  detect/
    base.py              # Detector protocol, Box dataclass
    megadetector.py      # yolov5 alias shim, letterbox, NMS, box scaling
    scripted.py          # deterministic boxes from a JSON sidecar (test detector)
  classify/
    base.py              # Classifier protocol, Prediction / Candidate dataclasses
    artifact.py          # .acmodel save/load, format_version, label space, model_id
    own_model.py         # timm backbone + linear head inference
    birds/
      base.py            # BirdProvider protocol, BirdResult, GpsPoint
      own_head.py        # DEFAULT — our CUB-200 head
      ebird_enrich.py    # optional online enrichment, degrades gracefully
      hosted_api.py      # documented stub
  taxonomy/
    labels.py            # static table loader, slug(), LABEL_RE, RESERVED_LABELS,
                         #   common ↔ scientific, rank, class rollup (Aves)
    data/coco_animals.csv, data/cub200.csv
    data/ebird_aliases.csv   # cub_key,ebird_com_name,ebird_sci_name — the ONLY name bridge
                             #   between our label space and eBird's orthography (§5.6)
  training/
    manifest.py          # JSONL manifest read/write/validate, split_for()
    dataset.py           # ManifestDataset, crop-on-load, deterministic split
    transforms.py        # train/eval transforms
    trainer.py           # two-stage finetune, checkpoint/resume
    evaluate.py          # top-1/top-5, per-class recall, confusion matrix, calibration
    export.py            # → .acmodel
    tinycnn.py           # from-scratch CNN for the synthetic smoke test
    synthetic.py         # deterministic seeded shape dataset
  decide.py              # dominance rule, thresholds, final label
  materialize.py         # atomic copy/symlink/hardlink, collisions, re-tag moves
  catalog.py             # SQLite schema, migrations, queries, overrides
  gui/
    app.py               # FastAPI app factory
    static/index.html, app.js, app.css
scripts/
  build_coco_manifest.py    # COCO val2017 → species manifest (real training data)
  build_cub_manifest.py     # CUB-200-2011 → bird manifest
  build_coco_dominance.py   # freezes tests/e2e/data/coco_dominance.json (§11 E4)
  make_e2e_fixtures.py      # builds the fixture SD-card tree from data/raw
tests/e2e/                  # end-to-end tests ONLY
tests/e2e/data/coco_dominance.json   # committed, frozen expectation list for E4
```

## 5. Pipeline stages

### 5.1 Scan (`scan.py`) — the source tree is read-only

`os.walk(source, followlinks=False)`, sorted for deterministic order. Yields `Candidate(path, size,
mtime)` or `Skipped(path, reason)`. Skip reasons are enumerated and recorded, never silently
dropped, and **each one has exactly one defining rule**:

| Reason | Rule (scan-time, before any decode) |
|---|---|
| `hidden` | basename starts with `.` |
| `system_dir` | inside `.Trash*`, `.thumbnails`, `.Spotlight-V100`, `__MACOSX` |
| `unsupported_extension` | extension belongs to no known family (§3.1), and is not RAW or video |
| `format_disabled` | family known but not listed in `formats` (§3.1) |
| `video` | extension in `.mp4 .mov .avi .m4v .mts .mpg .3gp` — always skipped |
| `raw_not_enabled` | RAW extension without `--raw` |
| `zero_bytes` | `st_size == 0` |
| `unreadable` | `PermissionError`/`OSError` on `os.stat` or on descending a directory |
| `too_large` | `st_size > max_file_bytes` (default 512 MiB) — a **file-size** cap, distinct from decode's 400 MP pixel cap, which records `too_large_pixels` |
| `symlink` | the entry is a symlink to a regular file and `--follow-source-symlinks` was not passed |
| `symlink_escape` | the entry is a symlink and `--follow-source-symlinks` **was** passed, but `os.path.realpath(entry)` is not under the resolved `source` — refusing it keeps `sources.path` honest provenance and keeps the read-only guarantee scoped to the card |

`symlink_loop` from iteration 2 is **deleted**: `os.walk(followlinks=False)` never descends a
symlinked directory, so it was unreachable. Symlinked **directories** are never descended under any
flag. Symlinked **files** are skipped (`symlink`) by default and only ingested with
`--follow-source-symlinks`, which then still rejects escapes (`symlink_escape`). Decode's pixel-count
rejection is a *separate* reason (`too_large_pixels`, §5.2), so a scan-time byte cap and a decode-time
pixel cap can never be confused in the catalog. E16 covers `too_large`, `symlink` and
`symlink_escape`.

Extension policy is derived from `formats` (§3.1) — default JPEG/PNG/TIFF/HEIC. RAW
(`.cr2 .cr3 .nef .arw .dng .raf .orf .rw2`) is skipped with reason `raw_not_enabled` unless `--raw`
is passed. Video (`.mp4 .mov .avi .m4v .mts .mpg .3gp`) is **always** skipped and recorded as
`video`, per spec.

**Read-only enforcement.** Three independent mechanisms, because this is the one irreversible risk:

1. Every source read goes through `images.open_source(path)`, which opens with mode `"rb"`. No
   module outside `materialize.py` performs any write, and `materialize.py` refuses any path that is
   not under `output_root`.
2. `Config.resolve` fails fatally if `output_root` is inside `source`, or `source` is inside
   `output_root`, or they are the same directory (resolved, symlinks followed).
3. An e2e test snapshots `(relative path, size, mtime_ns, sha256)` for the whole fixture SD-card
   tree before a run and asserts byte-for-byte equality after `classify`, a GUI re-tag,
   `export-trainset` and `verify --fix` (E1). This is the assertion that actually holds the
   invariant.

### 5.2 Decode (`images.py`)

`PIL.Image.open` → `ImageOps.exif_transpose` → `convert("RGB")`.
`pillow_heif.register_heif_opener()` is called once at import (decode **and** encode support; the
encoder is what lets `make_e2e_fixtures.py` build the HEIC fixture). RAW decode uses
`rawpy.imread(...).postprocess()` and is only reachable behind `--raw`. `Image.MAX_IMAGE_PIXELS` is
raised to 400 MP; anything larger is a `DecodeError` → `skipped(reason="too_large_pixels")` (the
scan-time byte cap is the separate `too_large`, §5.1).

**The one coordinate frame, stated once and binding everywhere.** All persisted geometry is in the
**EXIF-transposed** frame — the frame the pixels are actually in after `ImageOps.exif_transpose`,
which is also the frame the detector sees:

- `images.width` / `images.height` are `exif_transpose(im).size`, **not** the raw file's stored
  dimensions. For an orientation-6 file whose stored raster is 4000×3000, the catalog records
  3000×4000.
- Every `boxes` coordinate (`x0,y0,x1,y1`) and `area_frac` is in that frame. `area_frac` divides by
  the transposed `W*H`, so it is orientation-invariant.
- `/api/images/{sha}/thumb` is generated from the transposed image (§6), so the canvas overlay, the
  stored dimensions and the thumbnail all agree by construction.
- `/api/images/{sha}/full` serves the **original bytes** with their EXIF intact, so the browser must
  do the same rotation. The frontend therefore sets `image-orientation: from-image` explicitly on
  the detail `<img>` (the CSS default, made explicit so a future reset stylesheet cannot silently
  rotate every box overlay 90°).

E18 carries an orientation-6 fixture that pins this: a physically landscape raster stored with
orientation 6 must produce `images.width < images.height`, and a box that is valid only in the
transposed frame.

sha256 is computed by streaming the **raw file bytes** in 1 MiB chunks (not the decoded pixels), so
it is stable across Pillow versions and identifies duplicates exactly.

EXIF extracted: `DateTimeOriginal` → ISO-8601, GPS lat/lon → signed decimal degrees. Both are
**optional** in the contract; absent EXIF stores `NULL` and is never fabricated. GPS is only used by
the opt-in `ebird_enrich` provider (§5.6).

**Blur metric.** Variance of the Laplacian on the grayscale image. The image is **downscaled only
when its long edge exceeds 512 px** — never upscaled, because interpolating a small image upward
depresses Laplacian variance and would push small sharp animal-free photos toward `junk`. The edge
length actually measured is stored as `blur_ref_edge` next to `blur_score`, so two scores are only
ever compared with their reference edge in view. Default `blur_threshold = 100.0`; lower is
blurrier. The score is computed and stored for **every** image but can only *change* a label in the
no-animal branch (§5.7).

### 5.3 Crop

For each animal box, expand by `crop_margin` (0.08) of the box's own width/height on each side, clip
to the frame, and crop. Boxes narrower or shorter than 2 px after clipping cannot be resized
meaningfully — they are recorded with `species_common = NULL`, `species_conf = NULL` and
`species_status = "degenerate"`, and are excluded from classification, but **they still count as
animals for the dominance rule**, because excluding them would be an area floor by the back door.
A degenerate box that wins dominance yields the label `unknown` (§5.7). This is not hypothetical:
COCO val2017 contains 7 non-crowd animal boxes with a sub-2px side, the smallest at
`area_frac = 0.00001243`. Crops are resized to the artifact's declared `input_size` with bilinear
resampling, then normalized with the artifact's declared mean/std.

### 5.4 Detect (`detect/megadetector.py`)

Loading, exactly as proven in `scripts/probe_md_checkpoint.py` and re-proven for this iteration with
`roboflow`/`sahi` absent:

```python
sys.modules.setdefault("models", importlib.import_module("yolov5.models"))
sys.modules.setdefault("utils",  importlib.import_module("yolov5.utils"))
ckpt = torch.load(weights, map_location="cpu", weights_only=False)
model = (ckpt.get("ema") or ckpt["model"]).float().eval()
```

`weights_only=False` is required (the checkpoint pickles a live `models.yolo.DetectionModel`) and is
acceptable **only** because the file's provenance and exact byte length are pinned in `RECON.md`
(280,766,885 B, sha256 `94e88fe97c8050f2e3d0cc4cb4f64729d639d74312dcbe2f74f8eecd3b01b276`).
`MegaDetector.load()` verifies size and sha256 against those constants and fails fatally on
mismatch, with the download URL in the message. The `ema` weights are preferred over `model` when
present, which is standard for yolov5 checkpoints.

Inference per image, using yolov5's own helpers (module paths verified):
`yolov5.utils.augmentations.letterbox(im, new_shape=1280, stride=64, auto=False)` → CHW float32
`/255` → `model(x)[0]` → `yolov5.utils.general.non_max_suppression(pred, conf_thres=0.20,
iou_thres=0.45, max_det=100)` → `yolov5.utils.general.scale_boxes(letterboxed_shape, boxes,
original_shape)`. `stride=64` matches the checkpoint's reported `model.stride = [8,16,32,64]`.

`torch.set_num_threads(jobs)`, `torch.inference_mode()`. Budget from measured recon numbers: ~0.55
s/frame at 640 px, so **~2 s/image at 1280 px on 8 CPU cores**.

Output `Box(cls: "animal"|"person"|"vehicle", conf, x0, y0, x1, y1)` in **EXIF-transposed
full-resolution pixel coordinates** (§5.2's single frame — `scale_boxes` maps the letterboxed
inference tensor back onto `exif_transpose(im).size`, never onto the raw stored raster), plus
`area_frac = ((x1-x0)*(y1-y0)) / (W*H)` against the same transposed `W`/`H`, stored for
explainability in the GUI.

**Only `cls == "animal"` boxes participate in labelling.** `person` and `vehicle` boxes are stored
and drawn in the GUI but do not create labels; an image containing only people is therefore
`landscape` (or `junk` if blurry). This is a deliberate consequence of the closed label taxonomy in
`PLAN.md`, and because it will surprise a user whose card holds family photos, it is documented
user-facing in **both** `README.md` and `classify --help`: *"Images containing only people or
vehicles are filed as `landscape` (or `junk` if blurry); their person/vehicle boxes are still kept
in the catalog and drawn in the GUI."*

`detect/scripted.py` implements the same protocol by reading a `<image>.boxes.json` sidecar from a
fixture directory, giving the e2e suite exact control over box geometry for dominance cases. It is
selected only by `--detector scripted`, documented as a testing affordance.

### 5.5 Species classification (`classify/own_model.py`)

Our model, loaded from a `.acmodel` artifact (§7.1). Inference: batch the crops of one image
(batch ≤ 8), forward, `softmax(logits / artifact["temperature"])`, take top-5. The temperature is
read **without a fallback**: `train` always writes an explicit `1.0` and a missing key is a fatal
`AssetError` (§7.1), because `artifact.get("temperature", 1.0)` would be exactly the silent default
for a required value that I7 forbids.

```python
@dataclass(frozen=True)
class Candidate:
    rank: int          # 1..5
    common: str
    scientific: str | None
    score: float       # calibrated softmax probability

@dataclass(frozen=True)
class Prediction:
    common: str
    scientific: str | None
    rank_level: str            # "species" | "genus" | "family" | "order" | "class"
    score: float
    top5: tuple[Candidate, ...]
    model_id: str
```

`Candidate` is exactly the shape of a `candidates` row (§5.9), so persistence is a direct mapping.
`rank_level` comes from the artifact's label entry (§7.1) and is stored as `boxes.species_rank`, so
a coarse identification (`bird`, `cow` → `rank_level="class"`/`"species"`) is distinguishable from a
fine one everywhere, including the GUI caption.

Confidence gate: if `top1.score < min_species_confidence` (0.45) the box's species is `unknown` —
recorded with its top-5 so the GUI can still suggest, but never filed as a species.

### 5.6 Bird refinement (`classify/birds/`)

```python
@dataclass(frozen=True)
class GpsPoint:
    lat: float          # signed decimal degrees, [-90, 90]
    lon: float          # signed decimal degrees, [-180, 180]

@dataclass(frozen=True)
class BirdResult:
    provider: str                       # "own_bird_head" | "ebird_enrich" | "hosted_bird_api"
    status: str                         # "refined" | "kept_coarse" | "unreachable" | "no_gps"
    prediction: Prediction | None       # non-None iff status == "refined"

class BirdProvider(Protocol):
    name: str
    def refine(self, crop: Image.Image, coarse: Prediction, gps: GpsPoint | None) -> BirdResult: ...
```

**Merge rule** (the only way a `BirdResult` reaches the catalog):

- `status == "refined"` → `boxes.species_common/scientific/conf/rank` and the box's `candidates`
  rows are replaced by `prediction` and its top-5; `images.model_id` becomes
  `f"{species_model_id}+{bird_model_id}"`; `images.provider_status = "refined"`.
- any other status → the coarse row stands unchanged and only
  `images.bird_provider` + `images.provider_status` are written. Nothing else is mutated.

**Trigger.** Refinement fires when **any of the coarse top-3 candidates rolls up to class `Aves`**,
or when the coarse top-1 is below `min_species_confidence` and any top-3 candidate rolls up to
`Aves`. Keying on top-1 alone (iteration 1) made the path unreachable whenever the coarse head
misranked a bird, which is precisely the case refinement exists for. `--force-bird-head` is a
documented diagnostic flag that runs the bird head on every animal crop regardless of the coarse
prediction; it exists so the bird path can be exercised independently of coarse-head quality
(E8), and it is recorded in `runs.config_json`.

**`--force-bird-head` is diagnostic only, and says so at runtime.** Because the merge rule replaces
the coarse row whenever the bird head clears the gate, forcing the head onto *every* animal crop can
file a zebra as a CUB bird species. The flag therefore logs exactly one `WARNING` per run —
*"--force-bird-head: the CUB bird head is being run on every animal crop; non-bird crops may be
relabelled as bird species. This run is recorded in runs.config_json so its labels are
identifiable."* — and `runs.config_json` records it (already specified), so affected labels can be
found and re-run later. It is never a documented production setting.

- **`own_bird_head` (DEFAULT).** Our CUB-200 head from `models/birds.acmodel`. Fully offline. Returns
  `status="refined"` when its calibrated top-1 ≥ `min_species_confidence`; otherwise
  `status="kept_coarse"` (the label then stays the coarse `bird` if that cleared the gate, else
  `unknown`).
- **`ebird_enrich` (opt-in, `--bird-provider ebird_enrich`).** Not a classifier; a re-ranker.

  **The alias table is the whole contract.** `taxonomy/data/ebird_aliases.csv` (§4) is the only
  bridge between our label space and eBird's orthography, because eBird returns *its* names
  (`"Zebra Dove"`, `"Streptopelia chinensis"`) which need not equal a CUB label, and CUB ships
  `scientific = NULL` for many classes (§13.3). Schema — three columns, header row, one row per
  aliased label, checked at load:

  | Column | Meaning | Empty allowed |
  |---|---|---|
  | `cub_key` | our artifact label `key`; must exist in the loaded bird artifact's label space, else fatal exit 3 naming the row | no |
  | `ebird_com_name` | eBird `comName` for that species | yes (if the sci name is present) |
  | `ebird_sci_name` | eBird `sciName` for that species | yes (if the com name is present) |

  A row with **both** name columns empty is a fatal config error (it could never match anything, so
  it is a typo, not data). The table is a *partial* map by design: it covers the CUB classes for
  which an eBird name was authored offline, and nothing else.

  **Matching rule** (`candidate ↔ observation`), evaluated per candidate against the observation list:
  1. Look up the candidate's `cub_key` in the alias table. **No alias row → the candidate is exempt:
     score untouched, no canonicalisation, one `DEBUG` line.** Absence from *our* table is a gap in
     our data, not evidence about the bird's range, so it must never cause a down-rank.
  2. Otherwise it **matches** an observation when `ebird_sci_name` case-folded equals the
     observation's `sciName` case-folded; else when `casefold_alnum(ebird_com_name)` equals
     `casefold_alnum(comName)`, where `casefold_alnum(s)` lowercases and strips every `[^a-z0-9]`
     (so `"Brewer's Blackbird"` and `"brewers blackbird"` match).
  3. An **aliased, unmatched** candidate — present in our table, absent from the response — has its
     score multiplied by **0.25** (§13.5 explains the constant).

  - **No EXIF GPS** → canonicalize the names of aliased candidates against the table only, ranking
    **unchanged**, `status="no_gps"`, `provider_status="no_gps"`. No network call is attempted.
  - **With GPS** → one cached call to
    `https://api.ebird.org/v2/data/obs/geo/recent?lat=<lat>&lng=<lon>&dist=50&back=30` (4 s connect
    / 8 s read timeout, one attempt, no retry), responses cached in `<output_root>/.ebird-cache.json`
    keyed by `(round(lat,2), round(lon,2))`. The multiplier is applied per the matching rule above;
    the top-5 is then re-sorted and `min_species_confidence` is re-applied — so a demoted top-1 can
    legitimately become `unknown`. `status="refined"` if the order or the gated outcome changed, else
    `"kept_coarse"`.
  - **Unreachable** (`api.ebird.org` answers `000` from this sandbox) → exactly **one** `WARNING`
    per run, the `own_bird_head` result returned unmodified, `status="unreachable"`,
    `provider_status="unreachable"`. This is the single sanctioned graceful degradation in the
    design, and it is sanctioned because the enrichment is *optional*: no required value is being
    substituted and the catalog records that enrichment did not happen.
  - Selecting the provider **without** `ANIMAL_CLASSIFIER_EBIRD_API_KEY` is fatal at config time
    (§10.2) — there the missing value *is* required. A 401/403 from the API is likewise fatal.
- **`hosted_bird_api` (documented stub).** Raises `AssetError` at config time with the exact
  environment variables and endpoint it would need. It never silently no-ops.

Merlin itself remains unusable as an API by design, not by sandbox accident: Merlin Photo ID is an
on-device model with no public photo-ID endpoint (`PLAN.md`, `RECON.md`). `ebird_enrich` is the
genuine Merlin-adjacent integration.

### 5.7 Decide (`decide.py`) — the dominance rule

Input: the list of **animal** boxes with their `area_frac` and per-box predictions, plus the image's
blur score. Output: exactly one label from the closed set
`{<species>, multiple, landscape, junk, unknown}`.

```python
def species_or_unknown(box) -> str:
    # A box that could not be classified, or was not classified confidently, is `unknown`.
    if box.species_status == "degenerate" or box.species_conf is None:
        return "unknown"
    if box.species_conf < config.min_species_confidence:
        return "unknown"
    return box.species_slug     # pre-validated at artifact.load() by slug(); e.g. "plains_zebra"

animals = sorted(animal_boxes, key=lambda b: b.area_frac, reverse=True)

if not animals:
    return "junk" if blur_score < blur_threshold else "landscape"

if len(animals) == 1:
    return species_or_unknown(animals[0])

# >= 2 animals: the ONLY size gate in the system.
if animals[0].area_frac >= dominance_ratio * animals[1].area_frac:
    return species_or_unknown(animals[0])
return "multiple"
```

`species_or_unknown` is total: every branch returns a member of the closed set, `None` is never
compared with a float, and a degenerate winner is `unknown` rather than a crash or an invented
species (E26). It cannot raise on a bad label either, because `box.species_slug` was produced and
validated by `slug()` when the artifact was loaded (§5.8, §7.1) — the decision function never
slugifies at decision time, so a label-space problem is always a startup failure, never a mid-run one.

Deliberate properties:

- **There is no `min_box_area` and no absolute box-area floor anywhere** — not in `decide.py`, not in
  `Config`, not in the detector, not in the crop step, not in the catalog. `dominance_ratio` is the
  only size comparison in the system, and it is purely *relative*. A distant impala in the corner of
  a lion portrait is handled exactly as intended: the lion's box area is far more than 1.6× the
  impala's, so the image is filed as `lion`. Conversely a bird occupying 0.1% of a frame with
  nothing else detected is filed as that bird's species — an area floor would have thrown it away,
  which is the reason the floor is excluded.
- The comparison is a **multiplication**, never a division, so a degenerate zero-area second box
  makes the first dominant instead of raising `ZeroDivisionError`.
- Comparison is `>=`, so an exact-ratio tie resolves to dominant; equal areas (ratio 1.0 < 1.6)
  resolve to `multiple`.
- If the dominant box's species is below the confidence gate the label is `unknown`, **not**
  `multiple` — the dominance question was answered; only the identity is uncertain.
- The blur metric can only produce `junk` in the zero-animal branch. A blurry photo with a detected
  animal is still classified; MegaDetector firing is stronger evidence than a hand-tuned sharpness
  threshold.

### 5.8 Materialize (`materialize.py`)

Destination `<output_root>/<label>/<original_filename>`. **Label directories match
`^[a-z0-9][a-z0-9_-]{0,63}$`; every other entry directly under `output_root` is tool-internal and is
ignored by label enumeration, by the GUI and by `verify`'s orphan scan** — that covers
`.catalog.db`, `.catalog.db-wal`, `.catalog.db-shm`, `.thumbs/`, `.ebird-cache.json` and any stale
`.ac-tmp-*`.

**`taxonomy.labels.slug()`, published here beside the regex it must satisfy.** Real label sources are
not tame: humanised CUB-200 names carry apostrophes and periods (`Brewer's Blackbird`,
`Le Conte's Sparrow`), and `--allow-new-labels` lets a user type anything. Lowercasing and mapping
spaces alone would emit `brewer's_blackbird`, which is not a legal directory name here. The function
and its failure mode are therefore fixed:

```python
LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RESERVED_LABELS = frozenset({"multiple", "landscape", "junk", "unknown"})
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

def slug(common: str) -> str:
    """Canonical directory component for a species name. Never returns an invalid name."""
    folded = unicodedata.normalize("NFKD", common).encode("ascii", "ignore").decode()
    s = _NON_ALNUM.sub("_", folded.lower()).strip("_")[:64].rstrip("_")
    if not LABEL_RE.fullmatch(s):
        raise ConfigError(
            f"species label {common!r} does not slugify to a valid label directory "
            f"(got {s!r}, must match {LABEL_RE.pattern}); fix the label in the model artifact "
            f"or the taxonomy table"
        )
    return s
```

`slug()` **raises**; it never falls back to a sanitized guess, because the destination directory is a
required value (I7). Validation happens **up front, not mid-run**: `artifact.load()` slugs every
label entry in the artifact's label space at load time and exits 3 on the first failure (§7.1), so a
bad label space can never reach `materialize` after files have already been written.

**Reserved names.** `multiple`, `landscape`, `junk` and `unknown` are ordinary directories on disk
but reserved in the label space: a species slug or a user-supplied label equal to any of
`RESERVED_LABELS` is rejected — fatal exit 3 in `artifact.load()`, HTTP 422 in the GUI's label
validator (§6) — so a model class can never shadow a pipeline outcome and make `unknown/` ambiguous.

All three modes are atomic via **temp-then-`os.replace`, with the temp file created in the
destination directory** (guaranteeing the same filesystem, so `os.replace` is a true atomic rename):

| Mode | Implementation |
|---|---|
| `copy` (default) | `tempfile.mkstemp(dir=dest_dir, prefix=".ac-tmp-")` → stream copy → `shutil.copystat` (preserves mtime) → `os.replace(tmp, dest)` |
| `--link` | `os.symlink(abs_source, tmp)` → `os.replace(tmp, dest)` |
| `--hardlink` | `os.link(source, tmp)` → `os.replace(tmp, dest)` |

On any failure the temp file is removed in a `finally`, so a crash never leaves a partial image at
the real destination. Stale `.ac-tmp-*` files from a killed process are swept at the start of the
next run.

**Collisions on `<label>/<name>`:**

- *copy / hardlink mode* — if the existing path's content sha256 equals ours, the work is already
  done → no write, counted as `already_present` (this is what makes re-runs cheap).
- *link mode* — the existing destination is **not** hashed first, because dereferencing a symlink
  whose target is gone (card unplugged) is exactly the case that must not raise. `os.readlink(dest)`
  is compared to the intended absolute source: equal → `already_present`, no write. Different, or a
  dangling link → the destination is replaced through the same atomic temp+`os.replace` path and
  counted as `relinked`. Only a *regular file* at the destination is hashed.
- If the content differs, the destination becomes `<stem>-<sha256[:8]><suffix>`. If *that* also
  exists with different content — effectively impossible — it is a fatal `MaterializeError` rather
  than a guessed third name.

`--hardlink` across filesystems raises `OSError EXDEV`; that is **fatal and actionable**: "SD card
and ~/animal_pics are on different filesystems; hardlinks cannot cross filesystems — use --link for
symlinks or omit the flag to copy."

**Materialized files are never opened for writing — the precondition that makes I1 true in
`--hardlink` mode.** In hardlink mode a destination under `~/animal_pics` *is* the card's inode (E3
asserts equal `st_ino`), so I1's path-based guard would not stop an in-place write through the
destination path from modifying the SD card. That hole is closed structurally rather than by
convention: **no code path anywhere opens a materialized file with a writing mode, truncates it,
`copystat`s onto it or edits it in place. The only operations permitted on an *existing* destination
are `os.replace` (rename) and `os.unlink` (remove), both of which affect the directory entry only and
leave the inode's bytes untouched.** New content is always created as a fresh `.ac-tmp-*` inode and
renamed in. GUI re-tag (below) and `verify --fix` (§8) both comply, and E1 runs a `--hardlink` leg so
the property is tested, not just asserted.

`--dry-run` runs the whole pipeline, writes the planned destination into the catalog with
`status = "planned"`, and performs no destination writes at all.

**Re-tag (from the GUI) is a pure output-tree rename that never opens the source.** This is the
correction of iteration 2's biggest error: re-tag was defined by reference to the three materialize
modes, *all* of which read the SD card (stream copy from source, `os.symlink(abs_source)`,
`os.link(source)`) — yet the GUI's whole purpose is reviewing a card that has usually already been
unplugged. Re-tag is therefore its own operation, `materialize.retag()`, and the source path is never
opened by it (this is what E1's re-tag leg asserts):

```python
def retag(sha256: str, new_label: str, note: str | None) -> str:
    row = catalog.image(sha256)                    # 404 if unknown
    old = Path(row.dest_path or "")                # 409 if empty, not under output_root,
    require_under_output_root(old)                 #   or missing on disk -> row marked 'failed'
    new_dir = output_root / new_label              # mkdir(parents=True, exist_ok=True), 0o755
    new = collision_resolve(new_dir / old.name, sha256, mode=row.mode)   # §5.8 rules, below

    with catalog.write_tx():                       # 1. intent first, per I4
        catalog.insert_override(sha256, row.label, new_label, note)
        catalog.update_image(sha256, label=new_label, label_source="human",
                             status="materializing", dest_path=str(new))
    os.replace(old, new)                           # 2. one rename; same filesystem by construction
    with catalog.write_tx():                       # 3. commit the outcome
        catalog.update_image(sha256, status="done")
    return str(new)
```

Why `os.replace` alone covers all three modes: it renames a **directory entry**. A regular file moves,
a hardlink moves and keeps its inode (so the card's data is untouched — see the no-write rule above),
and a symlink moves **as a symlink**, without being dereferenced, so a link dangling because the card
is gone re-tags fine. Nothing is ever copied, and no new bytes are read from anywhere.

Collision at the new destination reuses the rules above, in the same order: identical content — or, in
link mode, identical `os.readlink` — means the file is already filed there, so the *old* entry is
`os.unlink`ed and the result is counted `already_present`; different content gets
`<stem>-<sha256[:8]><suffix>`; that name taken with different content again is a fatal
`MaterializeError`.

**The crash window, and who reconciles it.** Because intent is recorded before the rename (I4), the
only interleaving a crash can leave is *row says `new_label` / `status='materializing'` with the file
still at the old path* — a **pending move**, never a lost or duplicated image. `verify` detects it
(`dest_path` missing while the old-label path holds the row's content) and `verify --fix` **completes
the rename forward**; it never invents an intent, because the `overrides` row already states it. The
iteration-2 claim that a crash leaves the image "present in two label dirs" is **withdrawn** — a
rename has no such window. `classify` skips `status='done'` hashes and does not reconcile.

One interaction worth stating so it is not discovered in the field: if a `classify` run happens to
process that pending-move hash first (rows left `materializing` are re-processed, §5.9), it will
materialize the human label's destination **from the source** — the human label wins, so the outcome
is the same file in the same place — and the stale copy under the old label then remains as
`verify --fix`'s second case below. Either order converges on one file under the human label.

### 5.9 Catalog (`catalog.py`)

SQLite at `<output_root>/.catalog.db`, `PRAGMA journal_mode=WAL`, `foreign_keys=ON`,
`busy_timeout=10000`. Schema version in a `meta` table; on a newer version the tool refuses to run
rather than corrupt data.

```sql
images(sha256 TEXT PRIMARY KEY, bytes INT, width INT, height INT,
       exif_datetime TEXT, gps_lat REAL, gps_lon REAL,
       blur_score REAL, blur_ref_edge INT,
       label TEXT, label_source TEXT CHECK(label_source IN ('model','human')),
       confidence REAL, species_common TEXT, species_scientific TEXT, species_rank TEXT,
       status TEXT CHECK(status IN ('planned','materializing','done','skipped','failed')),
       dest_path TEXT, mode TEXT, model_id TEXT, bird_provider TEXT,
       provider_status TEXT, run_id TEXT, first_seen TEXT, last_updated TEXT)
sources(path TEXT PRIMARY KEY, sha256 TEXT, mtime_ns INT, FOREIGN KEY(sha256) REFERENCES images)
boxes(id INTEGER PRIMARY KEY, sha256 TEXT, idx INT, cls TEXT, conf REAL,
      x0 REAL, y0 REAL, x1 REAL, y1 REAL, area_frac REAL,
      species_common TEXT, species_scientific TEXT, species_conf REAL, species_rank TEXT,
      species_status TEXT, is_dominant INT, FOREIGN KEY(sha256) REFERENCES images)
candidates(box_id INT, rank INT, common TEXT, scientific TEXT, score REAL,
           PRIMARY KEY(box_id, rank), FOREIGN KEY(box_id) REFERENCES boxes)
overrides(id INTEGER PRIMARY KEY, sha256 TEXT, old_label TEXT, new_label TEXT,
          created_at TEXT, note TEXT, FOREIGN KEY(sha256) REFERENCES images)
skipped(path TEXT PRIMARY KEY, reason TEXT, detail TEXT, run_id TEXT, seen_at TEXT)
runs(run_id TEXT PRIMARY KEY, source_root TEXT NOT NULL, started_at TEXT, finished_at TEXT,
     argv TEXT, config_json TEXT, n_total INT, n_done INT, n_skipped INT, n_failed INT, state TEXT)

CREATE UNIQUE INDEX idx_boxes_sha_idx ON boxes(sha256, idx);   -- one row per (image, box index)
CREATE INDEX        idx_boxes_sha     ON boxes(sha256);        -- GUI's per-image box fetch
CREATE INDEX        idx_images_label  ON images(label);        -- /api/labels + /api/images?label=
CREATE INDEX        idx_sources_sha   ON sources(sha256);      -- provenance lookup for one hash
```

`candidates` is keyed `PRIMARY KEY(box_id, rank)` and `boxes` carries a **unique** `(sha256, idx)`
index, so a duplicated insert is an `IntegrityError` at the moment it happens rather than a silently
doubled box set (finding 2's failure mode becomes impossible at the schema level, not just avoided by
the write order below).

**`run_id` format.** `run_id = f"{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"` — sortable by
construction (UTC, second resolution) with a random tail for collisions inside one second. `GET
/api/run` and `verify`'s source-root lookup both define "newest" explicitly as
`ORDER BY started_at DESC, run_id DESC LIMIT 1`, never as "the last row inserted".

**`images.confidence` is NULL for exactly one, enumerated set of labels**: `landscape`, `junk`,
`multiple`, and any `unknown` whose image has no classifiable box (all boxes degenerate, or none at
all). It is non-NULL exactly when the label came from a scored winning box — i.e. a species label, or
an `unknown` produced by a *scored* box falling below `min_species_confidence` (there the score is
real and worth showing). NULL means "this label has no confidence by construction", never "we forgot
to write it", and §6's `include_unscored` exists so a confidence filter cannot make those rows
invisible.

`sources` is a separate table because one content hash legitimately has many source paths (the same
photo copied twice on the card) — it is classified once and filed once. A path whose **content
changed** in place is an upsert, not a conflict:

```sql
INSERT INTO sources(path, sha256, mtime_ns) VALUES(?,?,?)
ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, mtime_ns=excluded.mtime_ns;
```

The previous `images` row and its destination file are **retained**, because content — not path — is
the identity in this tool; the old photo was really filed, and the new one is a new image.

`runs.source_root` is written at run start (`NOT NULL`) and is the **only** provenance of the source
tree for later commands. **`verify`** takes no `SOURCE` argument and reads it from the newest `runs`
row for its nesting check (§8). The **GUI** never reads bytes from it and never resolves paths against
it; it uses the value for one cosmetic purpose only — displaying the card path in the page header, so
a user with two cards can tell which run they are looking at (§6, `GET /api/run`). Iteration 2's
"`gui` and `verify` read it" was ambiguous about that and is narrowed here.

**Idempotency, resume and `--limit`.** `classify` computes sha256 first and skips any hash already at
`status='done'` unless `--reclassify`. Rows left at `materializing` or `planned` by an interrupted run
are re-processed, which is safe because materialization is content-addressed and collision-aware.

**`--limit N` is a budget on *inference*, not on statuses** — the reading iteration 2 left ambiguous:

- Without `--reclassify`: a `status='done'` hash is skipped **before the budget is consulted**, so it
  costs nothing and repeated `--limit` runs advance through the card instead of re-examining its first
  N files forever.
- With `--reclassify`: every image actually processed consumes budget, **including** previously `done`
  ones — because under `--reclassify` those hashes are precisely what is submitted to inference.
  Candidates are ordered `images.last_updated ASC` (oldest first, `NULL` first for never-classified
  rows), so repeated `--reclassify --limit N` runs sweep the whole card exactly once per pass instead
  of re-doing the same N images.

E13 asserts both legs: two `--limit 2` runs over a 5-image card leave 4 `done` rows, and two
`--reclassify --limit 2` runs give **4 distinct hashes** a new `model_id`/`last_updated`.

**Re-inference replaces, never appends.** A re-classified hash must not accumulate boxes, so the
per-image write is one transaction with a fixed order:

```sql
BEGIN IMMEDIATE;
DELETE FROM candidates WHERE box_id IN (SELECT id FROM boxes WHERE sha256 = :sha);
DELETE FROM boxes      WHERE sha256 = :sha;
-- INSERT the new boxes, then their candidates
UPDATE images SET label=:label, confidence=:conf, species_common=:c, species_scientific=:s,
                  species_rank=:r, model_id=:mid, bird_provider=:bp, provider_status=:ps,
                  status=:st, dest_path=:dp, run_id=:rid, last_updated=:now
 WHERE sha256 = :sha;
COMMIT;
```

`overrides` rows are **never** touched by this (they are append-only, I6), and `first_seen` is never
rewritten. E13's `--reclassify` leg asserts `COUNT(*)` over `boxes` and `candidates` is unchanged
after a second run — the assertion that would have caught the doubling.

**`--reclassify` and overrides.** `--reclassify` re-runs inference but **keeps `label_source='human'`
labels** unless `--ignore-overrides` is also given. `--ignore-overrides` **never deletes `overrides`
rows**: it suppresses them for the current run and sets `label_source='model'`, so a later run
*without* the flag re-applies the newest override and moves the file back. That is intended and
idempotent, and it means a flag named "ignore" cannot destroy a human correction. Permanently
discarding a correction is done by re-tagging in the GUI (which appends a newer override), not by a
flag (E20).

**Single-writer discipline, without letting the GUI kill a run.** `classify` is the only long-lived
writer. The GUI opens **read-only** connections for all `GET` routes
(`sqlite3.connect("file:...?mode=ro", uri=True)`), so browsing during a 2.8 h run cannot contend at
all. A re-tag `POST` uses a short-lived write connection opened with **`busy_timeout = 250 ms`** (the
`classify` writer's is 10 s) and `BEGIN IMMEDIATE`, so contention surfaces as a fast, deterministic 409
instead of a browser request hanging for ten seconds — and E19 provokes it reliably by holding a write
transaction open for ~1 s rather than racing the writer. On contention it changes **nothing on
disk** and returns `409 {"error": "a classify run is writing the catalog; retry"}`. Inside
`classify`, lock contention retries with 0.5/1/2/4/8 s backoff and then records that **one image** as
`failed` (run exit code 4) — never a fatal exit 1, which would have contradicted the rule that a
per-image failure never aborts the run.

## 6. GUI (`gui/`)

`animal-classifier gui` starts `uvicorn` on `127.0.0.1:8765` (`--host` is not offered; the address is
hardcoded to loopback). The app is created by `create_app(config)` so the e2e suite can also drive it
in-process.

| Route | Behaviour |
|---|---|
| `GET /` | Single HTML page; no framework, no build step |
| `GET /static/*` | `app.js`, `app.css` |
| `GET /api/labels` | **Catalog is authoritative**: `SELECT label, COUNT(*) FROM images WHERE status IN ('done','planned') GROUP BY label ORDER BY label`. Each entry is `{label, count, files_on_disk}`, where `files_on_disk` is the count of non-tool-internal entries in `output_root/<label>/` (§5.8) — a visible cross-check, not a second source of truth |
| `GET /api/images` | Query: `label`, `min_conf`, `max_conf`, `include_unscored` (default `false`), `date_from`, `date_to`, `q`, `limit` (≤200, default 60), `offset`. Response `{items: [...], total, unscored_excluded}`. Rows include boxes, top-5 candidates, `species_rank`, blur score + `blur_ref_edge`, provider status |
| `GET /api/images/{sha256}/thumb` | 320 px JPEG generated **from `dest_path` only**, `exif_transpose` applied (§5.2), cached at `<output_root>/.thumbs/<sha256[:2]>/<sha256>.jpg`; 409 when there is no readable materialized file |
| `GET /api/images/{sha256}/full` | Full image bytes from `dest_path` under `output_root`; `Content-Type` from the real format; 409 on the same no-readable-file conditions |
| `POST /api/images/{sha256}/label` | `{"label": "...", "note": "..."}` → `materialize.retag()` (§5.8): one `os.replace` inside the output tree, the source is never opened; writes the override; returns the new `dest_path` |
| `GET /api/run` | Newest `runs` row (`ORDER BY started_at DESC, run_id DESC LIMIT 1`), including `source_root` for the header's card-path display → live progress while a `classify` run is in flight |

**Two numbers per label, on purpose.** Counting labels from the filesystem while listing images from
the catalog (iteration 2) gave one fact two owners, and they diverge in states this design creates:
after `--dry-run` every row is `planned` with no directories at all, so a filesystem sidebar would be
empty beside a populated grid (exactly E17's state); a `dest_path` deleted behind the tool's back
inflates rows over files with nothing to show it. So the catalog owns `count`, the filesystem
contributes `files_on_disk`, and the frontend renders `count` with a warning marker whenever
`files_on_disk != count` (with `run verify` as the suggested next step). E18 asserts both numbers,
including a dry-run leg where every label has `files_on_disk == 0` and a non-zero `count`.

**Unscored rows are never silently filtered away.** `images.confidence` is NULL for `landscape`,
`junk`, `multiple` and box-less `unknown` (§5.9), and SQL's `confidence >= :min` silently drops NULLs —
which would hide an entire class of images the moment a user touched the confidence slider, the exact
opposite of the "nothing is discarded" property the closed label set exists to give. Therefore: a
request carrying `min_conf` or `max_conf` matches NULL-confidence rows **only** when
`include_unscored=true`, and **every** `/api/images` response carries `unscored_excluded: <count>` (0
when no confidence filter is active) so the UI can say *"N images have no confidence score — show
them"* and offer the toggle. E18 asserts that count.

**No readable materialized file → 409, both byte routes.** `/thumb` and `/full` are `dest_path`-only:
the GUI never reads the source tree, never resolves a path from the request, and never puts an SD-card
read behind a grid tile. The three no-file conditions are `status='planned'` (a `--dry-run` catalog),
a NULL/empty `dest_path`, and a destination that cannot be opened — including **a dangling symlink**,
which is the *normal* state for `--link` users once the card is unplugged. All three return
`409 {"error": "no readable materialized file for this image", "reason": "planned" | "missing" | "dangling_symlink"}`,
and the grid renders a placeholder tile with the reason instead of a broken image. E17 covers the
planned case, E18 the dangling-symlink case.

**Byte-serving safety.** Paths are never taken from the request. The handler looks up `dest_path` in
the catalog, resolves it, and asserts it is under `output_root` before opening it read-only. The GUI
**never serves bytes from the source tree** — the iteration-1 "read-only source mode" is removed,
because nothing supplied a source root to it. A `sha256` path parameter that is not 64 hex characters
is a 422; an unknown hash is a 404; a hash with no readable materialized file is the 409 above.

Frontend: a thumbnail grid grouped by label with per-label counts in a sidebar, label/confidence/date
filters, and a detail view drawing boxes on a `<canvas>` overlay scaled from the stored pixel
coordinates and the stored `width`/`height` — both in §5.2's EXIF-transposed frame, and the detail
`<img>` carries `style="image-orientation: from-image"` so the browser's rotation of `/full` matches
that frame. Each box caption shows `cls`, `area_frac`, its top-1 species **and its rank** (`species` /
`class` / …), so a coarse `bird` is visibly coarse. The detail view shows the top-3 candidates and a
label picker; picking a label `POST`s and updates the grid in place.

New labels: `POST` accepts any label already present in the taxonomy or catalog. An unrecognized
label is rejected 422 unless the GUI was started with `--allow-new-labels`, which is how a user adds
a species the model does not know yet. In **every** case the submitted label must satisfy `LABEL_RE`
and must not be in `RESERVED_LABELS` (§5.8) — `--allow-new-labels` widens *which* labels are known, it
never relaxes the syntax or lets a user create a second `unknown/`; violations are 422 with the reason.

## 7. Species model and training subsystem

### 7.1 The `.acmodel` artifact (`classify/artifact.py`)

A single `torch.save` dict — self-describing so inference never has to guess:

```python
{"format_version": 1, "model_id": "species-effb0-20260915-a1b2c3",
 "arch": "efficientnet_b0", "input_size": 224,
 "normalize": {"mean": [...], "std": [...]},
 "labels": [{"key": "zebra", "common": "Zebra", "scientific": "Equus quagga",
             "class": "Mammalia", "rank": "species"},        # rank: species|genus|family|order|class
            {"key": "bird",  "common": "Bird",  "scientific": None,
             "class": "Aves", "rank": "class"}, ...],
 "state_dict": {...},              # backbone + head, one flat dict
 "temperature": 1.0,               # ALWAYS present; 1.0 until `eval --calibrate` fits one
 "train": {"dataset": "coco-animals", "manifest_sha256": "...", "epochs": 8,
           "val_top1": 0.91, "val_top5": 0.99,
           "backbone_weights": "efficientnet_b0_ra-3dd342df.pth",
           "calibrated_from": None, "created_at": "..."}}
```

(The `val_top1` above is an illustrative sample value, not a target; the asserted threshold is E7's.)

**`temperature` is always written, never defaulted at read time.** `train` writes
`"temperature": 1.0` with `train.calibrated_from = None` — an uncalibrated model is an *identity*
temperature, which is a real value, not a missing one. Only `eval --calibrate` ever writes a value
≠ 1.0, and only into a **new** file (§7.4, I8). On the read side `artifact.load()` treats a **missing**
`temperature` key as a fatal `AssetError`: *"artifact <path> has no 'temperature'; it was written by an
older/foreign writer — re-run `animal-classifier train`, or fit one with `animal-classifier eval
--calibrate --out <new.acmodel>`"*. This is deliberate: `artifact.get("temperature", 1.0)` is precisely
the silent default for a required inference value that I7 forbids, and E7 asserts a freshly trained
artifact carries `temperature == 1.0` and `calibrated_from is None` so `train`'s output is provably
loadable by `classify`.

Loading validates `format_version` (mismatch → fatal `AssetError` naming the expected version and
`animal-classifier train` as the fix), rebuilds `timm.create_model(arch, pretrained=False,
num_classes=len(labels))`, and loads the state dict with `strict=True`. Every label entry carries
`rank`, which flows into `Prediction.rank_level`, `boxes.species_rank` and the GUI caption.

**The whole label space is validated at load, before any file is written.** `artifact.load()` walks
every label entry and, for each one, computes `slug(common)` (§5.8) and checks it against `LABEL_RE`
and `RESERVED_LABELS`. Any label that fails to slugify, or that slugifies onto `multiple`,
`landscape`, `junk` or `unknown`, is a **fatal exit 3** naming the offending label and the artifact
path. Doing this at load — rather than at the moment a directory is created — is the point: a model
with one unusable class name fails immediately instead of half-way through a 5,000-image run with
files already on disk. The slugs are cached on the loaded artifact, so `decide.py` and
`materialize.py` use the same validated strings.

**`model_id` is immutable and traceable.** It is written into every `images` row, so any historical
label can be traced to the exact model that produced it. Consequently **an artifact is never mutated
once its `model_id` has been recorded**: `eval --calibrate` must write a *new* file (§7.4).

### 7.2 Manifests, datasets and the one true split

Manifest = JSONL, one line per training sample; `path` is absolute or manifest-relative, `box` is
optional and crops on load:

```json
{"path": "data/coco/val2017/000000397133.jpg", "label": "zebra", "box": [12.0, 40.5, 310.2, 288.9], "split": "train"}
```

`manifest.py` validates every line and **fails on the first bad one** with the line number and the
offending field — a training run must never quietly train on a truncated manifest. Missing files are
fatal, not skipped.

**The split function, published here as the single source of truth** (`training/manifest.py`):

```python
def split_for(path: str) -> str:
    """'val' for one image in five, deterministically, machine-independently."""
    key = os.path.basename(path)                     # not the full path
    return "val" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 5 == 0 else "train"
```

Keying on the **basename** rather than the whole path is deliberate: it makes the split identical no
matter where `data/raw` was unpacked or whether paths are absolute, and — because every crop of one
photograph shares that photograph's basename — it keeps all crops of an image on the same side of the
split, which is what actually prevents leakage. An explicit `split` field in the manifest always
wins (CUB ships an official split); otherwise `split_for` assigns it. Measured on COCO's 1,016 animal
images: **813 train / 203 val**.

**Leakage rule, binding on the test suite:** the species model trains on COCO val2017 and E4/E7
evaluate on COCO val2017, so every image an e2e test asserts against **must** satisfy
`split_for(path) == "val"`, and every training manifest a test feeds to `train` **must** be filtered
to `split == "train"`. E4's frozen list and E7's eval set are both built under that rule (§11).

Two builders produce real training data from the archives already in `data/raw/`:

- `scripts/build_coco_manifest.py` reads `annotations/instances_val2017.json`, extracts the 10 animal
  categories (ids 16–25: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe), drops
  `iscrowd=1` instances (34 of them), and emits one sample per instance with its `bbox` converted to
  `x0,y0,x1,y1` — **2,666 samples** over 1,016 images. `--classes` selects a subset (E7 uses seven).
- `scripts/build_cub_manifest.py` extracts CUB-200-2011 and emits one sample per image with the
  species from its directory name and the box from `bounding_boxes.txt`, honouring the official
  `train_test_split.txt`.

`dataset.py` crops with the same margin as inference (0.08) so train and inference see the same
framing, and uses `WeightedRandomSampler(1/sqrt(class_count))` for the long tail.

### 7.3 Transforms and training loop

**Input resolution: 224 is the shipped default; E7's 128 is a test-budget trade-off, stated as such.**
`timm`'s `efficientnet_b0` pretrained config is 224 px, and running it at 128 px uses those features
off-resolution — architecturally fine (the head sits behind global pooling) but a measurable accuracy
risk. The default `--input-size` is therefore **224** for every real training run and for both shipped
artifacts; E7 alone passes `--input-size 128`, purely to fit a real finetune inside a CPU-only 20-minute
e2e budget, and its `val_top1 >= 0.55` threshold is calibrated for that handicapped setting. If E7's
threshold ever proves too tight, raising E7's input size toward 224 is the first sanctioned remedy
(§11.2) — never lowering the threshold silently.

Train: `RandomResizedCrop(input_size, scale=(0.65, 1.0))`, `RandomHorizontalFlip`,
`ColorJitter(0.2, 0.2, 0.2, 0.05)`, normalize. Eval: `Resize(input_size * 1.14)` →
`CenterCrop(input_size)` → normalize. Horizontal flip only — vertical flips do not occur in wildlife
photography and would waste capacity.

Two-stage transfer learning (`trainer.py`), the path `PLAN.md` specifies:

1. **Head only** — backbone frozen (`requires_grad=False`, `eval()` so BatchNorm stats hold),
   `AdamW(lr=3e-3, wd=1e-4)`, cosine schedule, `epochs_head` (default 3).
2. **Finetune** — unfreeze the last `unfreeze_blocks` (default 2) stages, `lr=3e-4` for those and
   `3e-5` elsewhere, `epochs_finetune` (default 5).

Loss `CrossEntropyLoss(label_smoothing=0.1)`. `torch.set_num_threads(jobs)`.

**Device resolution (`--device`, `classify` and `train` alike):**

| Value | Behaviour |
|---|---|
| `auto` (default) | `cuda` if `torch.cuda.is_available()` else `cpu`; the resolved value is logged at `INFO` and stored in `runs.config_json` |
| `cpu` | always valid |
| `cuda` when `torch.cuda.is_available()` is false | **fatal, exit 3**: `"--device cuda requested but torch reports no available CUDA device; omit the flag or pass --device cpu"` |

Silently substituting CPU for an explicit `--device cuda` would violate invariant I7, so it does not
happen. (This host: `torch.cuda.is_available() == False`, so every in-sandbox run is CPU.)

Checkpoint after every epoch to `<out>.ckpt` (model, optimizer, scheduler, epoch, RNG states, best
val top-1, manifest sha256). `--resume` restores all of it and refuses to resume across a different
manifest sha256 or arch. Best-val weights are what get exported.

### 7.4 Eval and calibration (`training/evaluate.py`)

`animal-classifier eval --model <artifact> --manifest <m> --split val` writes `metrics.json` (top-1,
top-5, macro/per-class recall, support) and `confusion_matrix.csv`, and prints a summary table.
Confidence honesty matters because `min_species_confidence` gates the `unknown` label, so eval can
also fit a single temperature by LBFGS on the val split's NLL.

**Calibration produces a new artifact, never an edit.** `eval --calibrate --out <new.acmodel>`
writes a copy with `temperature` set, `model_id = f"{old_model_id}+cal{n}"` (n = 1 + the number of
existing `+calN` suffixes) and `train.calibrated_from = old_model_id`. Writing calibration into an
existing artifact path is **refused, fatal exit 3**. Without this, two behaviourally different models
would share one `model_id` and every pre-calibration `images` row would point at a model that no
longer exists.

### 7.5 `export-trainset`

`animal-classifier export-trainset -o ~/animal_pics --destination trainset.jsonl` walks the catalog
for `label_source='human'` (and optionally high-confidence model labels via
`--include-model-labels --min-conf`) and emits a training manifest, closing the loop: GUI corrections
become the next training set. Source images are read read-only.

**Which labels are exportable, and what `box` means for each.** Human overrides come from the *whole*
closed label set, so most of them are not species at all — and emitting `label: "junk"` into a species
manifest would train the head on a non-species class (§7.2 makes `box` optional, so nothing downstream
would reject it). The filter is therefore explicit:

| Catalog label | Default | With `--include-non-species` | `box` emitted |
|---|---|---|---|
| a species slug | **exported** | exported | the image's dominant box (`boxes.is_dominant = 1`); **omitted** if the image has no boxes, which is reachable only via a human override |
| `landscape` | skipped, counted | exported as class `landscape` (a scene-filter class) | never — these images have no boxes |
| `junk` | skipped, counted | exported as class `junk` | never |
| `multiple` | skipped, counted | **still skipped** — by definition no single box owns the image, so there is no honest sample to emit | — |
| `unknown` | skipped, counted | **still skipped** — the label records absence of an identity; exporting it would teach the model a class named "we don't know" | — |

Every run prints a counted summary to stdout — `exported=<n> skipped_multiple=<n> skipped_unknown=<n>
skipped_landscape=<n> skipped_junk=<n> no_box=<n>` — so a user who re-tagged 200 images and got 40
manifest lines can see exactly where the other 160 went instead of assuming data loss. `--include-non-
species` writes `landscape`/`junk` lines for users training a scene filter, and the mixed manifest is
legal because `train` derives its class list from the manifest itself. E25 asserts that a `junk`
override produces **no** manifest line by default and is reported in the counted summary.

### 7.6 The synthetic smoke path

`training/tinycnn.py` is a small from-scratch CNN (4 conv blocks, ~180 k params) and
`training/synthetic.py` generates a deterministic seeded set of coloured-shape images
(`--dataset synthetic`). This exists so the e2e suite has a training test that finishes in seconds
without touching the 1.1 GB CUB archive. It is explicitly **not** the proof of the training
subsystem — E7 is, by finetuning `efficientnet_b0` on real COCO crops.

## 8. CLI surface (`cli.py`)

`typer` 0.27.2. Global: `--config`, `--verbose/-v`, `--quiet`.

```
classify SOURCE
  -o/--output PATH            (default ~/animal_pics)
  --link | --hardlink         (mutually exclusive; default copy)
  --dry-run  --reclassify  --ignore-overrides
  --raw                       enable RAW decode (requires the `raw` extra)
  --formats [jpeg|png|tiff|heic]        repeatable; replaces the configured list
  --max-file-bytes INT        scan-time file-size cap (default 536870912 = 512 MiB)
  --follow-source-symlinks    ingest symlinked files inside the card (default: skip them)
  --dominance-ratio FLOAT     (default 1.6)
  --min-confidence FLOAT      (default 0.45)
  --blur-threshold FLOAT      (default 100.0)
  --detector [megadetector|scripted]   --detector-weights PATH  --detector-confidence FLOAT
  --species-model PATH  --bird-model PATH
  --bird-provider [own_bird_head|ebird_enrich|hosted_bird_api]  --force-bird-head
  --jobs INT  --limit INT  --device [auto|cpu|cuda]
gui   -o/--output PATH  --port INT (default 8765)  --allow-new-labels
train --manifest PATH  --out PATH  --arch TEXT  --backbone-weights PATH
      --dataset [manifest|synthetic]  --classes TEXT  --epochs-head INT  --epochs-finetune INT
      --batch-size INT  --input-size INT  --unfreeze-blocks INT  --resume  --jobs INT
      --device [auto|cpu|cuda]  --seed INT
eval  --model PATH  --manifest PATH  --split [train|val]  --report-dir PATH
      --calibrate  --out PATH        (--calibrate requires --out; never edits in place)
export-trainset -o/--output PATH  --destination PATH  --include-model-labels  --min-conf FLOAT
                --include-non-species
verify  -o/--output PATH  --json  --fix
```

`classify --help` carries the person/vehicle → `landscape` note from §5.4.

**`--link` / `--hardlink` mutual exclusion is hand-rolled**, because typer/click do not enforce it for
you: both are `bool` options and the `classify` callback raises `typer.BadParameter` when both are set,
which typer maps to **exit 2** (§10.2's usage class). Likewise `--formats` is a repeatable
`list[Format]` option with `Format` an `enum.StrEnum`, so an invalid family is a typer usage error
listing the valid set. Neither pattern is exotic, but both are stated so the implementer does not
reach for a `click.Group`-level hack.

**`verify`** checks and reports. Its exit codes are the documented ones, not a generic "non-zero":

| Outcome | Exit | Meaning |
|---|---|---|
| everything present and consistent | `0` | the report lists each check as `pass`/`skipped(...)` |
| a required **asset or config** is missing, unloadable or invalid — detector weights, an artifact, an unreadable catalog, a too-new schema, a non-writable `output_root`, a bad bird-provider config | `3` | same class as every other fatal asset/config error; `--fix` cannot repair it and does not try |
| assets fine, but the **catalog and filesystem disagree** — missing `dest_path`, an orphan file, a file under a directory that contradicts its row, a pending re-tag | `4` | reportable state, the `--fix`-able class; `verify --fix` exits `0` when it resolved every finding, `4` when any finding remains |
| an unexpected IO error while checking | `1` | as everywhere else |

E24 asserts these **specific** codes (3 for a deleted artifact, 4 for a missing `dest_path` and for a
pending re-tag, 0 after `--fix`), not merely "non-zero". The checks are:
detector weights present with the expected size + sha256 and loadable; species/bird artifacts present,
loadable, `format_version` accepted; backbone weights present; catalog openable at a known schema
version; `output_root` writable; **`output_root` not nested with the source root read from the newest
`runs` row** — with no `runs` row this specific check is reported as `skipped(no_runs)`, not passed and
not failed; catalog/filesystem reconciliation (rows whose `dest_path` is missing, files with no row,
images in a directory that disagrees with their row's label, ignoring tool-internal entries per §5.8);
bird provider configuration; and an *informational* reachability probe of the optional online providers
(a failure there is reported, never fatal).

**`verify --fix`** is the only reconciler, and it is deliberately narrow — it repairs exactly two
states, both of which the design can actually produce, and it obeys §5.8's no-write rule (it only ever
calls `os.replace` or `os.unlink`, never opening a materialized file):

1. **Pending re-tag (the §5.8 crash window).** `images.status = 'materializing'` with an `overrides` row
   whose `new_label` equals `images.label`, `dest_path` (under the new label) absent, and the row's
   content still present at the old label's path. `--fix` **completes the rename forward**
   (`os.replace(old, dest_path)`, then `status='done'`) and logs it at `INFO`. The human intent is read
   from the `overrides` row, never inferred from where files happen to sit.
2. **A duplicate the catalog explains.** The same content present under both an old and the current
   label directory, with an `overrides` row justifying the current label — reachable from a re-tag
   interrupted by an *older* build, or a user's manual copy. `--fix` unlinks the copy under the **old**
   label directory only.

Anything else — an orphan file the catalog has no row for, a row whose `dest_path` is simply gone, a
label directory a human created by hand — is **reported and left alone** (exit 4). `--fix` never
touches anything under the source root, never deletes a file the catalog cannot explain, never invents
a label, and never adopts the filesystem as intent (which would manufacture fake human overrides out of
a stray `mv`). Without `--fix`, both states above are reported and the exit code is 4.

Exit codes: `0` success · `1` unexpected/IO failure · `2` typer usage · `3` missing or invalid required
asset/config · `4` completed with per-image failures or skips.

## 9. Concurrency and throughput

Decode + sha256 + blur run in a `ThreadPoolExecutor(jobs)` (Pillow and hashlib release the GIL);
detection and classification run on the main thread with `torch.set_num_threads(jobs)`, because torch
already parallelizes internally and competing pools would thrash 8 cores. Images are processed in a
bounded pipeline (queue depth `2 * jobs`) so a 64 GB card never loads into RAM. Catalog writes are
batched per image in one transaction from the main thread only.

`classify` is the single writer; the GUI reads through read-only connections and writes only the short
re-tag transaction, with the 409-on-contention rule in §5.9. At ~2 s/image, 5,000 images ≈ 2.8 h;
`--limit` (a budget on inference, spent only on images actually submitted to it — §5.9) and resume make that tolerable, and
progress is written to the `runs` row so the GUI can show it live.

## 10. Error handling and validation

### 10.1 Per-operation failure table

Rule, applied everywhere: **a missing or unexpected *required* value fails loudly with an actionable
message; it is never replaced by a default.** A per-image failure does not abort the run — it is
recorded as `failed`/`skipped` with its reason and the run's exit code becomes 4.

| Operation | Failure | Class | Caller receives | Log |
|---|---|---|---|---|
| Load config | unknown TOML key, bad type, out-of-range value | fatal | `ConfigError`, exit 3, key + valid range | `ERROR` |
| Load config | `--config` path missing | fatal | exit 3, the path | `ERROR` |
| Load config | `output_root` nested with `source` | fatal | exit 3, both resolved paths | `ERROR` |
| Load config | `formats` entry not in `{jpeg,png,tiff,heic}` (e.g. `raw`) | fatal | exit 3, the offending value + the valid set | `ERROR` |
| Load config | `--device cuda` with no available CUDA device | fatal | exit 3, "omit the flag or pass --device cpu" | `ERROR` |
| Load config | `ebird_enrich` without API key; `hosted_bird_api` selected | fatal | exit 3, the env var names needed | `ERROR` |
| Detector weights | file absent / wrong size / sha256 mismatch | fatal | `AssetError`, exit 3, expected bytes + sha256 + download URL | `ERROR` |
| Detector weights | unpickle fails (`ModuleNotFoundError: models`) | fatal | exit 3, "re-sync with `uv sync --frozen`; yolov5 must be installed" | `ERROR` |
| Model artifact | absent, unreadable, `format_version` too new, `strict` key mismatch | fatal | `AssetError`, exit 3, pointing at `animal-classifier train` | `ERROR` |
| Scan | source missing / not a dir / not readable | fatal | exit 3 | `ERROR` |
| Scan | permission denied on a subdir | recoverable | `skipped(reason="unreadable")` | `WARNING` |
| Scan | family not in `formats` | not an error | `skipped(reason="format_disabled")` | `DEBUG` |
| Scan | `st_size > max_file_bytes` | not an error | `skipped(reason="too_large")` | `DEBUG` |
| Scan | symlinked file without `--follow-source-symlinks` | not an error | `skipped(reason="symlink")` | `DEBUG` |
| Scan | symlinked file with `--follow-source-symlinks` whose `realpath` escapes `source` | recoverable | `skipped(reason="symlink_escape")`, exit 4 | `WARNING` |
| Decode | truncated/corrupt image, zero bytes | recoverable | `skipped(reason=…)`, exit 4 | `WARNING` |
| Decode | more than 400 MP | recoverable | `skipped(reason="too_large_pixels")`, exit 4 | `WARNING` |
| Decode | `--raw` without the `raw` extra | fatal | exit 3, `uv pip install -e '.[raw]'` | `ERROR` |
| Detect | inference raises (OOM, malformed tensor) | recoverable | `failed`, exit 4 | `ERROR` + traceback at `-v` |
| Classify | crop degenerate (< 2 px) | recoverable | box kept for dominance, `species_status='degenerate'`, label `unknown` if it wins | `DEBUG` |
| Bird provider | `ebird_enrich` unreachable / timeout / 429 / bad JSON | **degrade** (optional enrichment) | own-head result unchanged, `provider_status='unreachable'` | one `WARNING` per run |
| Bird provider | `ebird_enrich` with no EXIF GPS | not an error | names canonicalized, ranking unchanged, `provider_status='no_gps'` | `DEBUG` |
| Bird provider | `ebird_enrich` 401/403 (bad key) | fatal | exit 3 — a rejected credential is a real misconfiguration | `ERROR` |
| Materialize | destination not writable / out of space (`ENOSPC`) | fatal | `MaterializeError`, exit 1, temp cleaned up | `ERROR` |
| Materialize | `--hardlink` across filesystems (`EXDEV`) | fatal | exit 3, suggests `--link` or copy | `ERROR` |
| Materialize | collision, same content (or, in link mode, `readlink` equals the intended source) | not an error | `already_present` | `DEBUG` |
| Materialize | link mode, destination symlink differs or dangles | not an error | atomically replaced, counted `relinked` | `DEBUG` |
| Materialize | collision, different content, hash-suffixed name also taken | fatal | exit 1, both paths | `ERROR` |
| Catalog | schema version newer than this build | fatal | exit 3, refuses to touch the DB | `ERROR` |
| Catalog | `database is locked` in `classify` past `busy_timeout` | recoverable after 0.5/1/2/4/8 s backoff | that image recorded `failed`, exit 4 — **never** a fatal abort | `WARNING`, then `ERROR` for the image |
| Catalog | `database is locked` on a GUI re-tag | rejected, nothing written | HTTP 409 `{"error": "a classify run is writing the catalog; retry"}` | `WARNING` |
| GUI `/full`, `/thumb` | `status='planned'`, `dest_path` NULL/missing, or a dangling symlink destination | rejected | HTTP 409 `{"error": "no readable materialized file for this image", "reason": …}`; grid shows a placeholder | `DEBUG` |
| GUI re-tag | unknown label without `--allow-new-labels`; invalid label chars; a reserved label; bad sha256 | rejected | HTTP 422 with the reason | `WARNING` |
| GUI re-tag | `dest_path` vanished (or is not under `output_root`) before the `os.replace` | rejected | HTTP 409, row marked `failed`, nothing renamed | `ERROR` |
| GUI re-tag | crash between the override transaction and the `os.replace` | recoverable, next `verify` | row left `materializing`; `verify` reports it, `verify --fix` completes the rename (§8) | `ERROR` at next `verify` |
| Artifact load | a label fails `slug()`/`LABEL_RE`, or collides with a reserved label | fatal | `AssetError`, exit 3, naming the label and the artifact | `ERROR` |
| Artifact load | `temperature` key absent | fatal | `AssetError`, exit 3, naming `eval --calibrate` / `train` | `ERROR` |
| eBird aliases | `cub_key` not in the bird artifact's label space, or both name columns empty | fatal | `ConfigError`, exit 3, with the CSV line number | `ERROR` |
| Train | manifest bad line / missing image / empty class | fatal | exit 3, line number + field | `ERROR` |
| Train | `--resume` against a different manifest sha256 or arch | fatal | exit 3, both ids | `ERROR` |
| Train | backbone weights absent, or `pretrained=True` attempted | fatal | exit 3, local path expected + RECON note | `ERROR` |
| Eval | `--calibrate` without `--out`, or `--out` equal to an existing artifact path | fatal | exit 3, artifacts are immutable once recorded | `ERROR` |
| Verify | required asset missing/unloadable | fatal | exit **3**, the asset and how to produce it | `ERROR` |
| Verify | catalog/filesystem inconsistency, `--fix` not given (or a finding `--fix` will not touch) | reported | exit **4**, the offending paths | `ERROR` |

Logging: `logging` to stderr, `INFO` default, `DEBUG` at `-v`, `WARNING`+ only at `--quiet`. Never
`print` for diagnostics; stdout carries only user-facing results (`verify --json`).

### 10.2 Validation of every external input

| Input | Rule | On failure |
|---|---|---|
| `SOURCE` | required; exists; is a dir; readable | fatal, exit 3 |
| `--output` | `~` expanded; created (parents) if absent; writable; not nested with `SOURCE` | fatal, exit 3 |
| `--link` / `--hardlink` | mutually exclusive | fatal, exit 2 |
| `formats` / `--formats` | non-empty; every entry in `{jpeg,png,tiff,heic}` | fatal, exit 3 |
| `--device` | `auto` \| `cpu` \| `cuda`; `cuda` requires an available CUDA device | fatal, exit 3 |
| `dominance_ratio` | float, finite, `>= 1.0` | fatal, exit 3 (`< 1.0` would make the smaller box dominant) |
| `min_species_confidence` | float in `[0.0, 1.0]` | fatal, exit 3 |
| `detector_confidence`, `detector_iou` | float in `(0.0, 1.0]` | fatal, exit 3 |
| `blur_threshold` | float `> 0` | fatal, exit 3 |
| `crop_margin` | float in `[0.0, 0.5]` | fatal, exit 3 |
| `detector_image_size` | int, multiple of 64, `[320, 2048]` | fatal, exit 3 |
| `max_file_bytes` | int `>= 1024`; a whole-file byte cap only — never applied to a box (I2) | fatal, exit 3 |
| `jobs` | int `>= 1`, warn above `os.cpu_count()` | fatal, exit 3 |
| `--limit` | int `>= 1` when given | fatal, exit 3 |
| `--port` | int in `[1024, 65535]` | fatal, exit 3 |
| weights/model/manifest paths | exist; readable; expected magic/keys | fatal, exit 3 |
| EXIF datetime / GPS | genuinely optional; parsed defensively; unparseable → `NULL` + `DEBUG` | not an error |
| GPS values when present | `lat ∈ [-90, 90]`, `lon ∈ [-180, 180]`; out of range → `NULL` + `WARNING` | not an error |
| eBird API key | required, non-empty **iff** `ebird_enrich` selected | fatal, exit 3 |
| GUI `label` | matches `LABEL_RE`; **not** in `RESERVED_LABELS`; and known unless `--allow-new-labels` | HTTP 422 |
| GUI `sha256` | exactly 64 hex chars | HTTP 422 |
| GUI `limit`/`offset` | int, `limit` in `[1, 200]`, `offset >= 0` | HTTP 422 |
| GUI `min_conf`/`max_conf` | float in `[0, 1]`, `min <= max` | HTTP 422 |
| GUI `include_unscored` | bool, default `false`; when `false` a confidence filter excludes NULL-confidence rows and the response reports `unscored_excluded` | HTTP 422 on a non-bool |
| Artifact label space | every entry slugifies (§5.8) and is not reserved | fatal, exit 3, at load |
| `ebird_aliases.csv` | header present; `cub_key` non-empty and in the bird label space; at least one of the two name columns non-empty | fatal, exit 3, with the line number |
| Manifest line | object with `path` (str, resolvable) + `label` (non-empty str); `box` = 4 finite numbers, `x0<x1`, `y0<y1`; `split` ∈ {train,val} | fatal, exit 3, with line number |

### 10.3 Invariants and their owning layer

| # | Invariant | Owner | Why there |
|---|---|---|---|
| I1 | The source tree is never written, moved, renamed or deleted — including by `verify --fix` and GUI re-tag. **And because a `--hardlink` destination *is* the card's inode, no code path ever opens a materialized file for writing or truncates it: the only permitted operations on an existing destination are `os.replace` and `os.unlink`, which touch the directory entry only** | `config.py` (nesting guard) + `images.open_source` (`"rb"` only) + `materialize.py` (refuses paths outside `output_root`; creates new bytes only in a fresh `.ac-tmp-*` inode) | Writes exist in exactly one module, so the guard has one place to live. The path-based guard alone is not sufficient under `--hardlink` (aliased inode), so the no-write rule is part of the invariant, and E1 runs both a copy and a hardlink leg |
| I2 | **No absolute box-area floor exists**; `dominance_ratio` is the only size gate. `max_file_bytes` is a whole-file scan cap and is never applied to a box | `decide.py`, with `config.py` rejecting unknown keys and `scan.py` owning the file cap | Keeping the rule in one pure function makes its absence auditable at a glance; rejecting unknown config keys stops a floor being reintroduced by configuration; keeping the byte cap in `scan.py` keeps it structurally incapable of seeing a box |
| I3 | Every image gets exactly one label from the closed set | `decide.py` (`species_or_unknown` is total) | A single pure decision function, so no caller can invent a label |
| I4 | An image's `dest_path` parent directory always equals its catalog `label` | `materialize.py` + `catalog.py` in one write order: row → `materializing` → file → `done` | Crash-consistency needs the intent recorded *before* the filesystem changes; `verify --fix` reconciles the one interleaving that can still diverge |
| I5 | One destination file per (content, label) | `materialize.py` content-addressed collision check | Only that module knows both the hash and the destination |
| I6 | A human override outranks the model on later runs, and is never destroyed by a flag | `catalog.py` (`overrides` rows are append-only; `label_source`) read by `classify` | The durable record is the DB, so the DB owns precedence |
| I7 | No required value is ever silently defaulted (`--device cuda`, missing weights, missing API key, missing artifact) | `config.py` validation + `errors.py` (no bare `except`, no `.get(k, fallback)` for required keys) | Fail-loud must be structural, not a convention |
| I8 | An artifact's behaviour never changes after its `model_id` has been recorded | `training/evaluate.py` (calibration writes a new file) + `artifact.py` (no in-place save) | Traceability is a property of the artifact writer, not of its readers |
| I9 | All persisted geometry is in one frame — the EXIF-transposed one: `images.width/height`, every box coordinate, `area_frac`, and the thumbnail | `images.py` (`exif_transpose` immediately after open; the only place raw dimensions exist) + `gui/` (thumb from the transposed image, `image-orientation: from-image` on `/full`) | One conversion at the entry point means no other layer can be in a different frame; E18's orientation-6 fixture proves it end-to-end |
| I10 | Every label the tool writes to disk is a validated, non-reserved slug | `taxonomy.labels.slug()` (raises), enforced at `artifact.load()` and in the GUI validator | Validating at load, not at directory-creation time, keeps a bad label space from failing half-way through a run |

## 11. Testability — end-to-end tests only

`tests/e2e/`, run with `uv run --frozen pytest tests/e2e`. **No unit tests anywhere**, no
`tests/unit`, no in-process assertions on private helpers. Every test drives a real entry point — the
installed `animal-classifier` console script via `subprocess`, or the real ASGI app over HTTP — and
asserts observable outcomes: the output tree, the catalog contents, exit codes, stderr, HTTP
responses.

Fixtures are built by `scripts/make_e2e_fixtures.py` from the archives already in `data/raw/`
(session-scoped, cached under `tests/e2e/_fixtures/`, gitignored). If `data/raw` is missing, the
fixture builder **fails with the exact download commands** rather than skipping — a silently skipped
suite is worse than a red one. The same rule applies to the HEIC fixture: it is produced by
re-encoding a COCO JPEG through `pillow-heif`, and if HEIF **encoding** is unavailable the builder
fails loudly instead of skipping E16 (and never substitutes AVIF).

### 11.1 Frozen expectation lists

Two committed JSON files remove all guesswork from the real-image tests. Both are generated by
`scripts/build_coco_dominance.py` from `instances_val2017.json` (non-crowd animal instances only),
contain only images with `split_for(file_name) == "val"` (§7.2 leakage rule), and are sorted by COCO
`image_id`:

| File | Contents | Measured availability (re-verified this iteration) |
|---|---|---|
| `tests/e2e/data/coco_dominance.json` | `{"threshold_high": 3.0, "threshold_low": 1.3, "high": [...], "low": [...]}`; `high` = **every** val-bucket image whose GT area ratio exceeds `threshold_high`, `low` = **every** val-bucket image whose ratio is below `threshold_low`; each entry `{image_id, file_name, gt_ratio, top_class, second_class}` | ratio > 3.0 → **22**; ratio < 1.3 → **29**; (ratio > 4.0 → **19**, the remedy threshold) |
| `tests/e2e/data/coco_species.json` | 20 images with **exactly one** non-crowd animal instance, `area_frac >= 0.20`, GT class in E7's seven | 24 exist |

**List lengths are data-driven, not hard-coded — and so is E4's assertion.** Iteration 2 froze both
lists at exactly 20 entries, which quietly broke §11.2's escape hatch: at the stricter ratio > 4.0 only
**19** val-bucket images exist, so "re-freeze at > 4.0" could not produce a 20-entry list, leaving only
the two moves this design forbids (lower the threshold inside test code, or break the leakage rule).
Fixed by making the size follow the data: the builder emits *all* qualifying val-bucket images at the
recorded thresholds and E4 asserts **`>= ceil(0.8 * len(list))`** on each side, reading the lengths from
the JSON. Concretely today: `high` = 22 → assert **≥ 18**; `low` = 29 → assert **≥ 24**; and at the
remedy threshold `high` = 19 → assert **≥ 16**, with no edit to the test's arithmetic. Cost is 51
detections ≈ 100 s, which keeps E4 inside its budget.

### 11.2 The tests

| # | Test | Drives | Asserts |
|---|---|---|---|
| E1 | Source tree immutability, **in copy mode and in `--hardlink` mode** | `classify`, then a GUI re-tag, `export-trainset`, `verify --fix` — the whole sequence run twice, once per mode | Pre/post snapshot of `(path, size, mtime_ns, sha256)` over the whole fixture card is byte-identical. The hardlink leg is the one that matters for I1: the destination shares the card's inode, so it proves re-tag renamed the entry instead of rewriting bytes (source `mtime_ns` and `sha256` unchanged, `st_ino` still shared afterwards). The re-tag leg additionally asserts the source is never *opened*: the run is performed with the fixture card directory made mode `0o500` (traversable, non-writable) and, on the second pass, with the card path renamed away entirely — re-tag must still succeed, because §5.8 forbids it from touching the source |
| E2 | Happy path, copy mode | `classify` on a mixed fixture card | Exact output tree; one dir per label; `images`/`boxes`/`sources`/`runs` rows (incl. `runs.source_root`); exit 0 |
| E3 | `--link` and `--hardlink` | `classify` twice into separate roots | `Path.is_symlink()` + resolved target; `st_nlink == 2` and equal `st_ino`; both destinations readable |
| E4 | Dominance on **real** COCO images with **real** MegaDetector weights | `classify --detector megadetector` over `coco_dominance.json` (22 `high` + 29 `low` today) | Every label in the closed set; **≥ `ceil(0.8 * len(high))`** of `high` are not `multiple` (18 of 22 today); **≥ `ceil(0.8 * len(low))`** of `low` are `multiple` (24 of 29); the thresholds and list lengths are read from the JSON, never literals in the test. Both counts printed per-image so a failure is diagnosable. No species identity assertion here (that is E7), because MegaDetector's box set legitimately differs from the annotator's |
| E5 | Dominance boundary cases | `classify --detector scripted` | ratio exactly 1.6 → dominant; 1.59 → `multiple`; equal areas → `multiple`; zero-area second box → dominant, no exception |
| E6 | **No area floor** | `classify --detector scripted` with a single animal box covering 0.05% of the frame, and a 0.1%-vs-0.5% pair | The lone tiny animal is filed as its species (never dropped, never `landscape`); the tiny pair still resolves by ratio; `classify --help` and `verify --json` expose no `min_box_area`-like option |
| E7 | **Real training + species identity** | `build_coco_manifest.py --classes zebra,elephant,giraffe,bear,cow,sheep,bird` (filtered to `split == "train"`: 1,625 crops) → `train --arch efficientnet_b0 --backbone-weights models/backbones/efficientnet_b0_ra-3dd342df.pth --input-size 128 --epochs-head 2 --epochs-finetune 2 --batch-size 32` → `eval --split val` (349 crops) → `classify --species-model <artifact>` over `coco_species.json` | Artifact has `format_version`, 7 labels with `rank`, `temperature == 1.0`, `train.calibrated_from is None` (so `train`'s output is provably loadable by `classify` without inventing a temperature), and `train` metadata; **`val_top1 >= 0.55`** (chance 0.143, majority-class baseline 0.229); artifact `train.val_top1` equals `metrics.json` top-1 within `1e-6`; **≥ 14/20** of the frozen single-animal val images receive their GT species label through the real CLI; wall clock **< 20 min** on 8 CPU cores. Per-class recall is reported but not asserted for `bear` (11 val instances) |
| E8 | Bird path | E7's artifact (its label space contains `bird`) + `build_cub_manifest.py` → `train` a 5-species CUB head → `classify` a CUB test-split image; then a second run with `--force-bird-head` | Label is one of the 5 CUB species; `bird_provider='own_bird_head'`; `provider_status='refined'`; `images.model_id == "<species_id>+<bird_id>"`; box `species_rank='species'`. The `--force-bird-head` run shows the bird head invoked without relying on the coarse head's ranking |
| E9 | Training resume | `train` interrupted after epoch 1, then `--resume`; and `--resume` with a different manifest | Resumes at the right epoch and finishes; the mismatched resume exits 3 with both ids |
| E10 | Synthetic smoke | `train --dataset synthetic --arch tinycnn` (4 classes, 500 train / 100 val) → `eval` → `classify` | Full train→eval→export→infer path; **`val_top1 >= 0.90`**; wall clock **< 90 s** |
| E11 | `landscape` / `junk` | `classify` on a sharp animal-free photo and a heavily blurred one | `landscape` vs `junk`; `blur_score` **and `blur_ref_edge`** stored for both; a small (≤ 512 px) sharp animal-free photo is `landscape`, proving no upscaling bias |
| E12 | `unknown` (low confidence) | `classify --min-confidence 0.999` | Label `unknown`; top-5 candidates still recorded; file still materialized |
| E13 | Idempotency, resume, `--limit`, `--reclassify` | `classify` twice; kill mid-run and rerun; two `classify --limit 2` runs over a 5-image fixture; then `classify --reclassify` twice; then two `classify --reclassify --limit 2` runs | Second run writes nothing new, `already_present` counted, no duplicate rows; killed run resumes and completes; after the two `--limit 2` runs there are **4** `status='done'` rows (budget spent on new work only). **`--reclassify` replaces rather than appends**: `COUNT(*)` over `boxes` and over `candidates` is identical after the second `--reclassify` run, and `overrides` rows are untouched. **`--reclassify` consumes budget**: the two `--reclassify --limit 2` runs give **4 distinct hashes** a fresh `last_updated`/`model_id`, in `last_updated ASC` order, so the pass sweeps forward instead of re-doing the same 2 images |
| E14 | Duplicate content | Same photo twice under different names | Classified once, one destination file, two `sources` rows |
| E15 | Collision, different content | Two different images sharing a filename in one label | Second becomes `<stem>-<sha8><suffix>`; both intact |
| E16 | Formats and skips, byte cap, symlink policy | Card with JPEG/PNG/TIFF/HEIC, an `.mp4`, a `.cr2`, a 0-byte file, a truncated JPEG, a sparse file just over `--max-file-bytes`, a symlink to a JPEG **inside** the card and a symlink to a JPEG **outside** it; plus a `--formats jpeg` run and a `--follow-source-symlinks` run | Four formats processed by default; `.mp4` → `skipped(video)`; `.cr2` → `raw_not_enabled`; 0-byte and truncated → `skipped`; over-cap file → `skipped(too_large)` **with its bytes never read**; both symlinks → `skipped(symlink)` by default; with `--follow-source-symlinks` the in-card link is ingested while the escaping link is `skipped(symlink_escape)`; exit 4. With `--formats jpeg` the PNG/TIFF/HEIC files are `skipped(format_disabled)` and the JPEGs still process. `--formats raw` exits 3 listing the valid set |
| E17 | `--dry-run` | `classify --dry-run`, then `GET /api/images/{sha}/full`, `/thumb`, `/api/labels` | Destination tree has no image files; rows are `planned`; exit 0; `/full` **and** `/thumb` return **409** with `reason="planned"`; `/api/labels` still lists every label with `count > 0` and `files_on_disk == 0` (the catalog-authoritative sidebar) |
| E18 | GUI browse, coordinate frame, unscored rows, dangling links | real uvicorn subprocess + `httpx`, over a fixture including an **EXIF-orientation-6** photo and a `--link` root whose card was moved away | `/` serves HTML; `/api/labels` returns catalog `count` **and** `files_on_disk`, both matching, excluding `.thumbs`/`.catalog.db`/`.ebird-cache.json`; `/api/images?label=…` filters; thumb and full bytes decode as images; box coordinates lie inside the stored dimensions; captions carry `species_rank`. **Frame**: for the orientation-6 file `images.width < images.height` although its stored raster is landscape, and its box is inside the transposed frame only; the served thumb's dimensions match `width`/`height`. **Unscored**: a `min_conf` query excludes the `landscape`/`junk`/`multiple` rows and reports the exact `unscored_excluded` count; the same query with `include_unscored=true` returns them. **Dangling**: `/thumb` on the link-mode row whose target is gone returns 409 `reason="dangling_symlink"`, not 500 |
| E19 | GUI re-tag renames inside the output tree, in all three modes, and 409s under contention | `POST /api/images/{sha}/label` against a copy-mode root, a `--link` root **whose source card has been renamed away**, and a `--hardlink` root; then the same POST while a competing writer holds a `BEGIN IMMEDIATE` transaction for ~1 s | File gone from the old label dir, present in the new one, content unchanged; `overrides` row written; `label_source='human'`; `status='done'`. Mode-specific: the symlink moves **as a symlink** with `os.readlink` unchanged and still dangling (no dereference, no error), and the hardlink keeps its `st_ino`. Under contention: HTTP **409** within ~250 ms (the GUI's `busy_timeout`), and **nothing on disk or in the DB changed** |
| E20 | Override lifecycle | E19 → `classify --reclassify` → `--reclassify --ignore-overrides` → `classify --reclassify` again | Human label kept; then model label restored *and* the `overrides` row still present; then the human label and destination **return** — proving `--ignore-overrides` suppresses without destroying |
| E21 | GUI validation | bad label (`Brewer's Blackbird`), a **reserved** label (`unknown`) — both with and without `--allow-new-labels` — bad sha256, out-of-range `limit`, non-bool `include_unscored`, unknown hash | 422 for each invalid input (the reserved label 422s even under `--allow-new-labels`), 404 for the unknown hash; nothing on disk changed |
| E22 | Fail-loud config | missing weights; unknown TOML key; `dominance_ratio=0.5`; `hosted_bird_api`; `ebird_enrich` without a key; output nested inside source; `--formats raw`; `--device cuda` on this CUDA-less host; `eval --calibrate` without `--out` | Exit 3 each, with the offending name in stderr, and **no** destination directory created |
| E23 | `ebird_enrich` degrades, and the alias table governs matching | `classify --bird-provider ebird_enrich` with a dummy key (host answers `000` here) on a GPS-tagged fixture and on a no-GPS fixture; plus a **stubbed-transport** leg (`httpx.MockTransport` injected through the provider's client factory — no network, and the only place the suite fakes a response) returning two observations, run against a bird artifact whose label space contains one aliased-and-observed class, one aliased-but-unobserved class and one class with **no** alias row | Both live runs succeed with labels equal to the `own_bird_head` run; exactly one warning in the GPS case with `provider_status='unreachable'`; the no-GPS case makes **no** network call and records `provider_status='no_gps'`. Stubbed leg: the observed candidate keeps its score, the aliased-unobserved candidate's score is exactly `0.25 ×` its original, and the **alias-less candidate is exempt** — score bit-identical — proving absence from our table is not treated as range evidence |
| E24 | `verify` and `verify --fix`, with **specific** exit codes | `verify --json` on a good tree; after deleting a materialized file; after moving the species artifact away; then after simulating a **pending re-tag** (row `materializing` + an `overrides` row + the file still at the old path); then after simulating a duplicate (same content under two label dirs + an `overrides` row) | Exit **0** with an all-present report; **4** naming the missing `dest_path`; **3** for the absent artifact (and `--fix` does not attempt it); **4** for the pending re-tag, then `verify --fix` exits **0** having completed the rename (file only under the new label, row `done`); **4** for the duplicate, then `--fix` exits **0** with exactly **one** copy left (the one the override justifies). The source tree is byte-identical throughout, and an orphan file with no catalog row is reported but **left in place** |
| E25 | `export-trainset` round trip, and the label filter | E19 (species override) plus a `junk` override and a `multiple` override → `export-trainset` → `train --manifest`; then `export-trainset --include-non-species` | Manifest lines validate, include the human species label and the dominant box, and `train` consumes them. The `junk` and `multiple` overrides produce **no** manifest line by default and are reported in the counted summary (`skipped_junk=1 skipped_multiple=1`). With `--include-non-species` the `junk` image appears as class `junk` **without** a `box`, while `multiple` is still absent |
| E26 | Degenerate dominant box | `classify --detector scripted` with a 1 px animal box as the only detection | Label `unknown`; `boxes.species_status='degenerate'`; the file is materialized under `unknown/`; exit 0 (no crash, no invented species) |

What is *hard* to e2e-test, and the deliberate choice made: browser-driven GUI testing needs a
Playwright browser download from a host not probed in RECON, so the GUI is tested at the HTTP +
filesystem boundary (E17–E21) instead. That is where the risky behaviour lives — the re-tag **rename**,
the 409 paths and the validation — so the coverage loss is cosmetic rendering only. Full-dataset
training accuracy is likewise not asserted: E7 asserts a real CPU-sized finetune clears a stated
threshold, and full runs are the user's `animal-classifier train` on their own machine.

**Two sanctioned test seams, and nothing else.** Every test drives a real entry point; the only
injected doubles are `--detector scripted` (a *shipped* detector implementation, selected by a real
CLI flag, so the geometry of dominance cases can be stated exactly) and E23's `httpx.MockTransport`
for the eBird HTTP boundary, which exists because `api.ebird.org` answers `000` in this sandbox and the
matching rule is otherwise untestable at all. Both replace an **external boundary** while the entire
program under test runs for real, through its installed console script. Neither is a unit test: no test
imports a private helper, asserts on an internal function, or lives outside `tests/e2e/`.

**If E4's aggregate thresholds fail on real MegaDetector output**, the sanctioned remedy is to
**re-freeze at a stricter GT ratio and let the list length follow the data** — `threshold_high = 4.0`
yields 19 val-bucket images, and E4's `>= ceil(0.8 * len(high))` then asserts ≥ 16/19 with **no change
to the test's arithmetic**. Record the re-freeze here and in `PROGRESS.md`. The forbidden moves remain
forbidden: never lower the ratio inside test code, never pad a list with train-bucket images (that
breaks §7.2's leakage rule), and never hard-code a list length again — that is precisely what made the
iteration-2 remedy impossible. If E7's `val_top1 >= 0.55` fails, the first remedy is raising E7's
`--input-size` back toward the shipped 224 (§7.3), not lowering the threshold.

## 12. Sequenced build order

1. **Dependency contract first**: write `pyproject.toml` exactly as §2.1, add `.python-version`
   (`3.12`), `uv lock`, commit `uv.lock`, then re-run `scripts/probe_md_checkpoint.py`,
   `probe_md_inference.py`, `probe_backbones.py` against the synced venv. Nothing else starts until
   those three pass on the locked environment.
2. `errors.py`, `config.py`, `catalog.py` (schema **with** the unique/lookup indexes and
   `PRIMARY KEY(box_id, rank)` from §5.9), `taxonomy/` + static label tables + `slug()`/`LABEL_RE`/
   `RESERVED_LABELS`.
3. `scan.py`, `images.py`, `materialize.py` → `classify` end-to-end with `detect/scripted.py` and a
   pass-through classifier. Lands E1, E2, E3, E5, E6, E11, E13–E17, E22, E26.
4. `detect/megadetector.py` (weights verification, alias shim, letterbox/NMS) +
   `scripts/build_coco_dominance.py`. Lands E4.
5. `training/` + `scripts/build_*_manifest.py` + `train`/`eval`. Lands E7, E9, E10.
6. `classify/own_model.py`, `classify/birds/*` + `taxonomy/data/ebird_aliases.csv`, wire `decide`
   thresholds. Lands E8, E12, E23.
7. `gui/`. Lands E18–E21.
8. `export-trainset`, `verify` (+ `--fix`), README (including the person/vehicle note). Lands E24, E25.

## 13. Open assumptions

1. `~/animal_pics` and the SD card are on the same filesystem often enough that `--hardlink` is
   useful; when they are not, the `EXDEV` message tells the user what to do instead.
2. The COCO-trained head is a genuine but honestly-scoped safari species model (zebra, elephant,
   giraffe, bear, bird, plus horse/sheep/cow/cat/dog) — the best real labelled data reachable
   in-sandbox. `train` + `export-trainset` are the documented path to a wider label set on the user's
   machine, where LILA/Snapshot Serengeti and HuggingFace are reachable. Species outside the set come
   back as `unknown` rather than a confident wrong guess, which is the intended failure mode.
3. Scientific names ship as a static in-repo table (no taxonomy API is reachable). The 10 COCO classes
   get hand-authored binomials with an explicit `rank` (`bird` is `rank="class"`, not a species);
   CUB-200 ships common names with `scientific = NULL` where no reliable offline source exists —
   honest absence of an optional field, not a substituted default.
4. `taxonomy/data/ebird_aliases.csv` is hand-authored offline and is deliberately **partial**: it covers
   the CUB classes for which an eBird name could be written down without reaching the API. A candidate
   with no alias row is exempt from the down-rank (§5.6), so growing the table can only ever make
   enrichment sharper, never retroactively wrong — and `provider_status` always records whether
   enrichment ran at all.
5. `ebird_enrich`'s 0.25 down-rank multiplier is a deliberate, documented constant, not a fitted one:
   nothing in-sandbox can reach eBird to tune it. It is strong enough to demote an off-range species
   below a plausible in-range alternative and weak enough that a correct-but-unreported species can
   still win from a high score. A user who wants different behaviour changes the constant, and the
   catalog's `provider_status` always says whether enrichment ran.

## 14. Responses to design review iteration 1

Verdict was `CHANGES_REQUESTED` with 25 findings. All 25 are resolved below. Three fixes are
implemented differently from the reviewer's suggestion because the suggestion was measurably
unbuildable; those are called out explicitly.

| # | Sev | Resolution | Where |
|---|---|---|---|
| 1 | HIGH | **Adapted — the suggested pin set is unsatisfiable.** Measured: `roboflow==1.3.8` pins `opencv-python-headless==4.10.0.84` exactly, and `roboflow` 1.4.2 pins `typer<0.26`, so "pin `roboflow==1.3.8` + `opencv-python-headless==5.0.0.93` + `typer==0.27.2`" cannot resolve. Instead the design pins the RECON-verified versions **and overrides `roboflow` and `sahi` out of the resolution** alongside `opencv-python` — verified: 109 packages, none of the three present, and MegaDetector loads + forwards with both blocked. The `typer<0.26` claim is withdrawn; `uv.lock` is committed and `uv sync --frozen` is mandatory; a moved `torch`/`timm`/`numpy`/`yolov5` pin forces a re-run of the three probes | §2, §2.1, §12.1 |
| 2 | HIGH | `species_or_unknown` is now defined in full and is total: degenerate or `None` confidence → `unknown`, below gate → `unknown`, else `slug(common)`. New test E26 asserts label `unknown`, `species_status='degenerate'`, file under `unknown/`, exit 0 | §5.7, E26 |
| 3 | MED | `GpsPoint`, `Candidate(rank, common, scientific, score)`, `Prediction(…, top5, rank_level)` and `BirdResult(provider, status, prediction)` are all defined, with the explicit merge rule for `refined` vs everything else (including `model_id = "<species>+<bird>"`) | §5.5, §5.6 |
| 4 | MED | Trigger widened to *any of the coarse top-3 rolls up to Aves, or top-1 below the gate with any Aves in top-3*; `--force-bird-head` added as a documented diagnostic; E7's head now includes **`bird`** (7 classes) so E8 can reuse it and the natural trigger can fire | §5.6, §8, E7, E8 |
| 5 | MED | E4 now asserts against a committed frozen list (`tests/e2e/data/coco_dominance.json`) with decidable aggregates — every label in the closed set, ≥ 16/20 high-ratio not `multiple`, ≥ 16/20 low-ratio `multiple`, counts printed. Species identity moved to E7. **Adapted:** lists are 20 + 20, not 40 + 40, because only 22 val-bucket images have GT ratio > 3.0 | §11.1, E4 |
| 6 | MED | `split_for()` published as the single source of truth; e2e assertions must use `split == "val"` images and training manifests must be filtered to `split == "train"`. **Refined:** the key is the file *basename*, not the full path, so the split is machine-independent and all crops of one photo stay on one side | §7.2, §11.1 |
| 7 | MED | E7: 7 classes, train-split crops only, `--input-size 128 --epochs-head 2 --epochs-finetune 2 --batch-size 32`, `val_top1 >= 0.55`, artifact/metrics agreement within `1e-6`, < 20 min, plus ≥ 14/20 species-identity hits. E10: 4 synthetic classes, 500/100, `val_top1 >= 0.90`, < 90 s | E7, E10 |
| 8 | MED | `pillow-heif==1.7.0` is a core dependency (verified: encodes HEIF and re-reads on py3.12 + pillow 12.3.0, libheif 1.23.3); `make_e2e_fixtures.py` writes the HEIC fixture by re-encoding a COCO JPEG and fails loudly if HEIF encoding is unavailable. No AVIF substitute | §2, §5.2, §11 |
| 9 | MED | `formats` fully specified as the eligible-family selector with a family→extension table, fatal on unknown values (including `raw`), new skip reason `format_disabled`, repeatable `--formats` flag that replaces the list, and E16 cases for `--formats jpeg` and `--formats raw` | §3.1, §5.1, §10.1, §10.2, E16 |
| 10 | MED | `runs.source_root TEXT NOT NULL` added and written at run start; `gui`/`verify` read it from the newest `runs` row; with no runs row the nesting check reports `skipped(no_runs)`. The GUI's source-serving mode is **removed** — `/full` serves only from `output_root` and returns 409 for `planned` rows | §5.9, §6, §8, E17 |
| 11 | MED | GUI `GET`s use `mode=ro` connections; the re-tag `POST` returns 409 on contention with nothing written; `classify` retries lock contention 0.5/1/2/4/8 s then records **one image** as `failed` (exit 4) instead of aborting. E19 asserts the 409 path | §5.9, §9, §10.1, E19 |
| 12 | MED | `--device` resolution table added: `auto` logs and stores the resolved device; explicit `cuda` without an available CUDA device is fatal exit 3 with the exact remedy; `cpu` always valid. E22 covers it on this CUDA-less host | §7.3, §10.1, §10.2, E22 |
| 13 | MED | No GPS → canonicalize only, `provider_status='no_gps'`, ranking unchanged, no network call. With GPS → one cached `/v2/data/obs/geo/recent?…dist=50&back=30` call, absent candidates' scores × **0.25**, re-sort, re-apply the confidence gate (a demoted top-1 may become `unknown`). E23 asserts the no-GPS case | §5.6, §10.1, E23 |
| 14 | MED | `eval --calibrate --out <new.acmodel>` writes a copy with `model_id = "<old>+calN"` and `train.calibrated_from`; calibrating into an existing artifact path is fatal exit 3. New invariant I8: artifacts are immutable once their `model_id` is recorded | §7.4, §10.1, §10.3 |
| 15 | MED | `--ignore-overrides` never deletes `overrides` rows — it suppresses them for one run and sets `label_source='model'`; a later plain `--reclassify` re-applies the newest override and moves the file back. Permanent discard is a GUI re-tag. E20 extended with the third run | §5.9, E20 |
| 16 | MED | `verify --fix` added as the **only** reconciler, scoped to deleting a stale copy under an old label directory when an `overrides` row justifies the current label; never touches the source; report-only without `--fix`. The "classify reconciles" claim is withdrawn. E24 extended. **Superseded in iteration 3** by §15 findings 1 and 16: re-tag is now a rename, so `--fix`'s primary case is *completing a pending move*, and the exit code is the specific `4` (`3` for assets), not "non-zero" | §5.8, §8, E24, §15.1 |
| 17 | MED | `--limit N` caps images **newly submitted to inference**; `done` hashes are skipped without consuming budget. E13 asserts two `--limit 2` runs over a 5-image card yield 4 done rows. **Extended in iteration 3** (§15 finding 3) with the `--reclassify` case the rule did not cover: there every processed image consumes budget, ordered `last_updated ASC` | §5.9, §9, E13 |
| 18 | NIT | `rank` added to every artifact label entry, `Prediction.rank_level`, `boxes.species_rank` and `images.species_rank`; shown in the GUI detail caption | §5.5, §5.9, §6, §7.1 |
| 19 | NIT | Stated once: label directories match `^[a-z0-9][a-z0-9_-]{0,63}$`; every other entry under `output_root` is tool-internal and ignored by `/api/labels` and by `verify`'s orphan scan. E18 asserts it | §5.8, §6, §8, E18 |
| 20 | NIT | `sources` upsert specified as `ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, mtime_ns=excluded.mtime_ns`, with the note that the previous `images` row and destination file are retained because content is the identity | §5.9 |
| 21 | NIT | Blur metric **downscales only** when the long edge exceeds 512 px; `blur_ref_edge` stored beside `blur_score` and asserted in E11 | §5.2, §5.9, E11 |
| 22 | NIT | Link mode compares `os.readlink(dest)` with the intended absolute source before any hashing; a differing or dangling link is replaced atomically and counted as `relinked`; only regular files are hashed | §5.8, §10.1 |
| 23 | NIT | The unreachable "missing HEIF library" fatal row is deleted; `pillow-heif` is core | §10.1 |
| 24 | NIT | "~2,700" corrected to **2,666** (2,700 − 34 `iscrowd`) over 1,016 images, with per-class counts, and the explicit note that `bear` (71 total, 11 val) limits per-class recall claims — so no threshold is asserted on it | §1, §7.2, E7 |
| 25 | NIT | People/vehicle-only images filing as `landscape` (or `junk` if blurry) is documented in `README.md` **and** `classify --help`, noting their boxes are still catalogued and drawn | §5.4, §8, §12.8 |




## 15. Responses to design review iteration 2

Verdict was `CHANGES_REQUESTED` with 20 findings (1 HIGH / 13 MEDIUM / 6 NIT) and no blocking
condition tripped. **All 20 are resolved.** Two are implemented differently from the reviewer's
suggested wording — both are noted as *adapted* with the reason, and both end up stricter than the
suggestion rather than looser. Nothing is backlogged and nothing is ignored.

| # | Sev | Resolution | Where |
|---|---|---|---|
| 1 | HIGH | **Adapted, and stricter.** Re-tag is now its own operation, `materialize.retag()`, which never opens the source: `collision_resolve` the new name, then **one `os.replace`** inside the output tree — which moves a regular file, keeps a hardlink's inode, and moves a symlink *as* a symlink without dereferencing (so the card-unplugged case, the normal one, works). The "present in two label dirs" sentence is **deleted**. Adapted on ordering: the reviewer put the override row *after* the rename; this design writes it **before**, because I4 requires intent to be recorded before the filesystem changes. The only crash window is therefore a *pending move* (row `materializing`, file still at the old path), which `verify --fix` completes forward from the `overrides` row — never a duplicate, never a lost image, never an intent inferred from the filesystem. E1 gains a hardlink leg plus a leg with the card renamed away; E19 exercises all three modes and asserts the symlink stays dangling and the hardlink keeps `st_ino` | §5.8, §6, §8, I1, E1, E19, E24 |
| 2 | MED | The per-image re-inference transaction is published verbatim: `BEGIN IMMEDIATE` → `DELETE FROM candidates WHERE box_id IN (SELECT id FROM boxes WHERE sha256=:sha)` → `DELETE FROM boxes WHERE sha256=:sha` → re-insert → `UPDATE images` → `COMMIT`, with `overrides` and `first_seen` untouched. Belt **and** braces: `candidates` gains `PRIMARY KEY(box_id, rank)` and `boxes` a **unique** `(sha256, idx)` index, so a doubled insert is an `IntegrityError` instead of silent corruption. E13 gains a `--reclassify` leg asserting `COUNT(*)` over both tables is unchanged | §5.9, E13 |
| 3 | MED | `--limit` is redefined as a budget on **inference**, not on statuses: without `--reclassify`, `done` hashes are skipped *before* the budget is consulted; with `--reclassify`, every processed image consumes budget and candidates are ordered `last_updated ASC` (NULLs first) so repeated passes sweep the card. E13 asserts two `--reclassify --limit 2` runs touch **4 distinct hashes** | §5.9, E13 |
| 4 | MED | `train` always writes `"temperature": 1.0` with `calibrated_from = None`; a **missing** `temperature` key is a fatal `AssetError` naming `train` / `eval --calibrate`; only `eval --calibrate` writes ≠ 1.0, into a new file (I8). The `artifact.get("temperature", 1.0)` reading is called out as exactly the I7 violation it would be. E7 asserts `temperature == 1.0` and `calibrated_from is None` | §5.5, §7.1, §10.1, E7 |
| 5 | MED | `slug()` is published in full beside `LABEL_RE`: NFKD → ASCII-fold → lowercase → collapse `[^a-z0-9]+` to `_` → strip `_` → truncate 64 → **raise `ConfigError`** if it still fails the regex. It never returns a sanitized guess. `artifact.load()` slugs the entire label space up front and exits 3 on the first failure, so a bad label space fails at startup, not mid-run with files on disk (new invariant I10). E21 uses `Brewer's Blackbird` | §5.8, §7.1, §10.1, §10.2, I10, E21 |
| 6 | MED | One frame, stated once and made an invariant (I9): all persisted geometry is **EXIF-transposed** — `images.width/height = exif_transpose(im).size`, all box coordinates and `area_frac` in that frame, `/thumb` generated from the transposed image, `/full` with an explicit `image-orientation: from-image` on the `<img>`. §5.4's "original-image pixel coordinates" is corrected. E18 gains an orientation-6 fixture asserting `width < height` for a physically landscape raster plus a box valid only in the transposed frame | §5.2, §5.4, §6, I9, E18 |
| 7 | MED | Every scan reason now has exactly one defining rule, in a table. `too_large` is **defined** (`st_size > max_file_bytes`, default 512 MiB, new config key + validation + `--max-file-bytes`), and decode's pixel rejection is renamed `too_large_pixels` so a byte cap and a pixel cap can never be confused. `symlink_loop` is **deleted** (unreachable under `followlinks=False`). New policy: a symlinked file is `skipped(symlink)` unless `--follow-source-symlinks`, which still rejects `realpath` escapes as `symlink_escape`; symlinked directories are never descended. §3 states explicitly that `max_file_bytes` is a whole-file cap and never sees a box (I2). E16 covers all three | §3, §5.1, §5.2, §8, §10.1, §10.2, I2, E16 |
| 8 | MED | `taxonomy/data/ebird_aliases.csv` (`cub_key,ebird_com_name,ebird_sci_name`) added to §4 with a validated schema (unknown `cub_key` or both names empty → exit 3 with the line number). Matching is now exact: sci-name case-folded equality first, else common-name equality after `[^a-z0-9]` stripping; **candidates with no alias row are exempt** from the 0.25 multiplier, because absence from our table is not range evidence. E23 gains a stubbed-transport leg asserting observed / aliased-unobserved / alias-less scores independently | §4, §5.6, §10.1, §10.2, §13.5, E23 |
| 9 | MED | The **catalog is authoritative** for label counts: `/api/labels` is `SELECT label, COUNT(*) … WHERE status IN ('done','planned') GROUP BY label`, and each entry carries `files_on_disk` as a visible cross-check, with the frontend marking any mismatch and pointing at `verify`. E18 asserts both numbers; E17 adds the dry-run leg (`count > 0`, `files_on_disk == 0`) | §6, E17, E18 |
| 10 | MED | `/thumb` is defined: generated from `dest_path` **only**, `exif_transpose` applied, then cached. `planned`, missing `dest_path` and a **dangling symlink** all return `409 {"error": "no readable materialized file for this image", "reason": …}` on both byte routes, and the grid renders a placeholder with the reason. E17 covers `planned`, E18 the dangling link | §6, §10.1, E17, E18 |
| 11 | MED | `export-trainset` gets an explicit label-policy table: species slugs only by default; `landscape`/`junk` exportable via `--include-non-species` as scene classes; `multiple` and `unknown` **never** exported (no single honest box / no identity); a species-labelled image with no boxes is exported with `box` omitted; every run prints a counted summary so nothing looks silently lost. E25 asserts a `junk` override yields no line and is counted | §7.5, §8, E25 |
| 12 | MED | **Adapted to the data-driven form** (the reviewer's second option), which removes the hard-coded length permanently: the frozen lists contain *all* qualifying val-bucket images at recorded thresholds and E4 asserts `>= ceil(0.8 * len(list))`, reading lengths from the JSON. Re-measured this iteration: 22 at > 3.0 (assert ≥ 18), 29 at < 1.3 (assert ≥ 24), 19 at > 4.0 (assert ≥ 16). The remedy is now executable with **no** test-code arithmetic change; the forbidden moves are restated | §11.1, §11.2, E4 |
| 13 | MED | I1 is extended past its path-based argument: because a `--hardlink` destination *is* the card's inode, **no code path ever opens a materialized file for writing or truncates it — only `os.replace` and `os.unlink` are permitted on an existing destination**, and new bytes are always a fresh `.ac-tmp-*` inode. `verify --fix` and GUI re-tag are stated to comply. E1 now runs a hardlink leg (it was copy-only) | §5.8, I1, E1 |
| 14 | MED | §5.9 enumerates exactly when `images.confidence` is NULL (`landscape`, `junk`, `multiple`, box-less `unknown`) and why a scored sub-threshold `unknown` still has one. §6 adds `include_unscored` (default `false`) and makes **every** `/api/images` response carry `unscored_excluded`, so the UI can offer "N images have no confidence score". E18 asserts the count both ways | §5.9, §6, §10.2, E18 |
| 15 | NIT | Narrowed: `verify` reads `runs.source_root` for its nesting check; the GUI uses it for **one** cosmetic purpose — the card path in the header via `GET /api/run` — and never resolves a path or serves bytes against it | §5.9, §6 |
| 16 | NIT | `verify`'s exit codes are mapped onto the documented table: `3` missing/unloadable asset or config, `4` catalog/filesystem inconsistency (the `--fix`-able class, `0` after a successful `--fix`), `1` unexpected IO. E24 asserts the specific codes | §8, §10.1, E24 |
| 17 | NIT | `--force-bird-head` is documented as **diagnostic only**, with one `WARNING` per run stating that non-bird crops may be relabelled and that `runs.config_json` records the flag so affected labels are identifiable | §5.6 |
| 18 | NIT | `run_id = f"{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"` (sortable + collision-safe), and "newest" is defined as `ORDER BY started_at DESC, run_id DESC LIMIT 1` for both `GET /api/run` and `verify` | §5.9, §6, §8 |
| 19 | NIT | `PRIMARY KEY(box_id, rank)` on `candidates`, unique `(sha256, idx)` on `boxes`, plus `idx_boxes_sha`, `idx_images_label`, `idx_sources_sha` | §5.9, §12.2 |
| 20 | NIT | `RESERVED_LABELS = {multiple, landscape, junk, unknown}` is published next to `LABEL_RE`; a colliding artifact label is fatal exit 3 at load, a colliding GUI label is 422 **even under `--allow-new-labels`** (that flag widens the known set, never the syntax). E21 asserts it | §5.8, §6, §7.1, §10.2, I10, E21 |

### 15.1 The reviewer's unverified assumptions, addressed where they were actionable

| Assumption | What changed here |
|---|---|
| `efficientnet_b0` used at 128 px is off its 224 px pretrained config | §7.3 now states that **224 is the shipped default** for every real run and both artifacts, that E7's 128 px is a CPU-budget trade-off **for E7 only**, and that raising E7's input size — not lowering the threshold — is the first remedy if `val_top1 >= 0.55` fails |
| The GUI's 409 path needs a named `busy_timeout` | Named: the GUI's write connection uses **`busy_timeout = 250 ms`** with `BEGIN IMMEDIATE` (the writer keeps 10 s), and E19 provokes the 409 by holding a competing transaction for ~1 s instead of racing a 10 s wait |
| typer's shape for a repeatable `--formats` and the hand-rolled `--link`/`--hardlink` exclusion is unexercised | §8 states both patterns explicitly: `--formats` is a repeatable `list[Format]` over a `StrEnum`, and the mutual exclusion is a callback raising `typer.BadParameter` → exit 2 (typer does not enforce it) |
| E7's and E10's accuracy/wall-clock thresholds, and E4's aggregates against real detector output | Still unproven by construction — they can only be measured by running the suite. §11.2's rule stands, and finding 12's fix makes E4's remedy executable when it is needed |
| `ebird_enrich`'s 0.25 multiplier and `dist=50&back=30` | Still chosen, not fitted (§13.5) — but the **matching rule** they act on is now fully specified (finding 8), so the untestable part is reduced to the constant itself |
