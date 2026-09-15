# Safari Image Classifier — Technical Design

Status: authored 2026-09-15, iteration 1 (no `docs/design-review.json` present).
Grounded in `docs/PLAN.md` (agreed behaviour) and `docs/RECON.md` (what is actually reachable
and verified in-sandbox). Every asset named here has been downloaded, loaded and executed —
no blocked host is assumed available anywhere in this document.

## 1. Overview

`animal-classifier` is a Python CLI + library that walks a read-only SD-card tree of safari
photographs, detects animals with MegaDetector v5a, crops each detection, classifies the crop
with **our own finetuned species model**, resolves one label per image with the dominance rule,
and materializes the image into `~/animal_pics/<label>/` by copy (default), symlink or hardlink.
Every decision is recorded in a SQLite catalog keyed by content sha256, so runs are idempotent
and resumable. A local FastAPI + vanilla-JS GUI on `127.0.0.1:8765` shows each image with its
detection boxes and label, and lets the user re-tag — which moves the file to the correct label
directory and records a human override that later runs respect.

The pipeline is a straight line with one branch:

```
scan → decode → detect (MegaDetector v5a) → crop ─┬→ species head ──┐
                                                  └→ bird head ─────┴→ decide → materialize → catalog
        (no animal boxes) ────────────→ blur metric → landscape | junk
```

Two models are ours and are produced by the training subsystem in this repo:

| Artifact | Label space | Trained on (real, in-sandbox) | Backbone |
|---|---|---|---|
| `models/species.acmodel` | 10 animal classes → canonical species names | COCO val2017 animal crops from ground-truth boxes (2,700 instances, 1,016 images) | timm `efficientnet_b0`, ImageNet-pretrained, finetuned |
| `models/birds.acmodel` | 200 bird species | CUB-200-2011 (11,788 images, per-class dirs, boxes, official split) | timm `efficientnet_b0`, ImageNet-pretrained, finetuned |

`RECON.md` §"Explicit decision on training" applies: a pretrained backbone **and** real labelled
data are both present and verified, so the synthetic-data contingency in the build spec is **not
used as the proof of the training subsystem**. Training is proven by genuinely finetuning a real
ImageNet backbone on real labelled crops and asserting the accuracy reached. A from-scratch
`tinycnn` on generated synthetic data is retained only as a fast deterministic smoke test of the
train → eval → export → infer path (§8.6), never as a substitute for the real run.

## 2. Technology stack (locked once approved)

| Concern | Choice | Why / constraint |
|---|---|---|
| Package + env manager | **`uv` exclusively** — `uv venv`, `uv pip install`, `uv run`, `uv lock` | Never `pip`, `pip3`, `virtualenv`, `python -m venv`. |
| Python | 3.12 (`requires-python = ">=3.10"`) | Sandbox interpreter is CPython 3.12.13. |
| CLI | `typer` **0.25.x** | Resolution pins it: `yolov5` → `roboflow` → `typer<0.26`. Verified by `uv lock` (§2.1). The CLI must not use 0.26+-only APIs. |
| Detection | `yolov5==7.0.14` + MegaDetector v5a checkpoint | The checkpoint is a yolov5-v7 pickle; only this distribution can unpickle it (§5.4). |
| Image I/O | `pillow`, `pi-heif` (HEIC), `rawpy` (optional extra, RAW only) | `pi_heif.register_heif_opener()` verified working. |
| CV metric | `opencv-python-headless` | **Never `opencv-python`** — this image has no `libGL.so.1` (RECON). |
| Modelling | `torch` 2.14.0, `torchvision` 0.29.0, `timm` 1.0.29 | CUDA build from pypi is the only route; inference and training are CPU-only here. |
| Backbone weights | local files in `models/backbones/` | `timm.create_model(arch, pretrained=False)` + explicit `load_state_dict`. **Never `pretrained=True`** (hits blocked huggingface.co). |
| GUI backend | `fastapi` + `uvicorn[standard]` | Resolves to fastapi 0.141.1. |
| GUI frontend | vanilla JS + CSS served as static files | No npm, no bundler, no build step. |
| HTTP client | `httpx` | GUI e2e tests, and the optional eBird provider. |
| Tests | `pytest`, `tests/e2e` only | No unit tests anywhere. |
| Build backend | `hatchling`, `src/` layout | Already established. |

### 2.1 The opencv dependency conflict, and its resolution

`yolov5` 7.0.14 requires `opencv-python` (via itself, `sahi` and `ultralytics`), which imports
`libGL.so.1` and therefore **cannot import in this image**. Declaring `yolov5` naively in
`pyproject.toml` would make `uv run` install the broken wheel and break the detector.

Resolution — a `uv` override that removes `opencv-python` from the resolution entirely, while we
declare the headless build ourselves:

```toml
[tool.uv]
override-dependencies = ["opencv-python; python_version < '3.0'"]
```

Verified by running `uv lock` against the full intended dependency set: 131 packages resolved,
the lock contains **`opencv-python-headless` 5.0.0.93 and no `opencv-python` at all**, with
`typer` 0.25.1 and `setuptools` 80.10.2 (`<81`, required because yolov5 7.0.14 still imports
`pkg_resources`). `setuptools<81` is declared explicitly.

This matters beyond tidiness: `uv run` **syncs the venv to `pyproject.toml` + `uv.lock`**. It was
observed changing an installed package mid-session. Therefore **every runtime dependency must be
declared in `pyproject.toml`** — anything installed ad-hoc with `uv pip install` will be evicted.
Optional extras: `[project.optional-dependencies] raw = ["rawpy"]`, `dev = ["pytest"]`.

## 3. Configuration

Layered, highest wins: **CLI flag → environment variable (`ANIMAL_CLASSIFIER_*`) → TOML file →
built-in default**. The resolved config is dumped into the `runs` catalog row so any result can
be explained later.

TOML is read from `--config`, else `$XDG_CONFIG_HOME/animal-classifier/config.toml`, else
`~/.config/animal-classifier/config.toml`. A `--config` path that does not exist is **fatal**
(the user asked for a specific file); an absent default path is normal and silently skipped.

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
species_model     = "models/species.acmodel"
bird_model        = "models/birds.acmodel"
detector_weights  = "models/md_v5a.0.0.pt"
bird_provider     = "own_bird_head"  # own_bird_head | ebird_enrich | hosted_bird_api
jobs              = 7
formats           = ["jpeg", "png", "tiff", "heic"]
```

`config.py` exposes a frozen `Config` dataclass built by `Config.resolve(...)`. Validation is
total and happens once, before any file is touched (§10.2). **There is no `min_box_area` key, no
`min_box_area_frac`, no `min_animal_area`, and no absolute box-area floor under any other name.**
An unknown key in the TOML file is fatal, which is what stops such a key from ever being
silently honoured.

## 4. Module layout

```
src/animal_classifier/
  cli.py                 # typer app: classify | gui | train | eval | export-trainset | verify
  config.py              # layering, validation, frozen Config
  errors.py              # ConfigError, AssetError, DecodeError, MaterializeError, CatalogError
  scan.py                # read-only recursive walk, extension policy, skip reasons
  images.py              # decode, EXIF orient/time/GPS, sha256, blur metric, crop
  detect/
    base.py              # Detector protocol, Box dataclass
    megadetector.py      # yolov5 alias shim, letterbox, NMS, box scaling
    scripted.py          # deterministic boxes from a JSON sidecar (test detector)
  classify/
    base.py              # Classifier protocol, Prediction dataclass
    artifact.py          # .acmodel save/load, format_version, label space
    own_model.py         # timm backbone + linear head inference
    birds/
      base.py            # BirdProvider protocol
      own_head.py        # DEFAULT — our CUB-200 head
      ebird_enrich.py    # optional online enrichment, degrades gracefully
      hosted_api.py      # documented stub
  taxonomy/
    labels.py            # static table loader, common ↔ scientific, class rollup (Aves)
    data/coco_animals.csv, data/cub200.csv
  training/
    manifest.py          # JSONL manifest read/write/validate
    dataset.py           # ManifestDataset, crop-on-load, stratified split
    transforms.py        # train/eval transforms
    trainer.py           # two-stage finetune, checkpoint/resume
    evaluate.py          # top-1/top-5, per-class recall, confusion matrix, calibration
    export.py            # → .acmodel
    tinycnn.py           # from-scratch CNN for the synthetic smoke test
  decide.py              # dominance rule, thresholds, final label
  materialize.py         # atomic copy/symlink/hardlink, collisions, re-tag moves
  catalog.py             # SQLite schema, migrations, queries, overrides
  gui/
    app.py               # FastAPI app factory
    static/index.html, app.js, app.css
scripts/
  build_coco_manifest.py # COCO val2017 → species manifest (real training data)
  build_cub_manifest.py  # CUB-200-2011 → bird manifest
  make_e2e_fixtures.py   # builds the fixture SD-card tree from data/raw
tests/e2e/               # end-to-end tests ONLY
```

## 5. Pipeline stages

### 5.1 Scan (`scan.py`) — the source tree is read-only

`os.walk(source, followlinks=False)`, sorted for deterministic order. Yields `Candidate(path,
size, mtime)` or `Skipped(path, reason)`. Skip reasons are enumerated and recorded, never
silently dropped: `hidden`, `system_dir` (`.Trash*`, `.thumbnails`, `.Spotlight-V100`,
`__MACOSX`), `unsupported_extension`, `video`, `raw_not_enabled`, `zero_bytes`, `unreadable`,
`symlink_loop`.

Extension policy: JPEG/PNG/TIFF/HEIC (`.jpg .jpeg .png .tif .tiff .heic .heif`) by default. RAW
(`.cr2 .cr3 .nef .arw .dng .raf .orf .rw2`) is skipped with reason `raw_not_enabled` unless
`--raw` is passed. Video (`.mp4 .mov .avi .m4v .mts .mpg .3gp`) is **always** skipped and
recorded as `video`, per spec.

**Read-only enforcement.** Three independent mechanisms, because this is the one irreversible
risk in the tool:

1. Every source read goes through `images.open_source(path)`, which opens with mode `"rb"`. No
   module outside `materialize.py` performs any write, and `materialize.py` refuses any path that
   is not under `output_root`.
2. `Config.resolve` fails fatally if `output_root` is inside `source`, or `source` is inside
   `output_root`, or they are the same directory (resolved, symlinks followed).
3. An e2e test snapshots `(relative path, size, mtime_ns, sha256)` for the whole fixture SD-card
   tree before a run and asserts byte-for-byte equality after `classify`, `gui` re-tag, and
   `export-trainset` (§11, E1). This is the assertion that actually holds the invariant.

### 5.2 Decode (`images.py`)

`PIL.Image.open` → `ImageOps.exif_transpose` → `convert("RGB")`. `pi_heif.register_heif_opener()`
is called once at import. RAW decode uses `rawpy.imread(...).postprocess()` and is only reachable
behind `--raw`. `Image.MAX_IMAGE_PIXELS` is raised to 400 MP; anything larger is a
`DecodeError` → `skipped(reason="too_large")`.

sha256 is computed by streaming the **raw file bytes** in 1 MiB chunks (not the decoded pixels),
so it is stable across Pillow versions and identifies duplicates exactly.

EXIF extracted: `DateTimeOriginal` → ISO-8601, GPS lat/lon → signed decimal degrees. Both are
**optional** in the contract; absent EXIF stores `NULL` and is never fabricated. GPS is only used
by the opt-in `ebird_enrich` provider.

Blur metric: variance of the Laplacian. The image is converted to grayscale and resized so its
long edge is exactly 512 px (so the score is resolution-independent), then
`cv2.Laplacian(gray, cv2.CV_64F).var()`. Lower = blurrier; default `blur_threshold = 100.0`. The
score is always computed and stored for every image, but only *changes* a label in the no-animal
branch (§5.7).

### 5.3 Crop

For each animal box, expand by `crop_margin` (0.08) of the box's own width/height on each side,
clip to the frame, and crop. Boxes narrower or shorter than 2 px after clipping cannot be
resized meaningfully — they are recorded with `species = NULL, species_status = "degenerate"` and
are excluded from classification, but **they still count as animals for the dominance rule**,
because excluding them would be an area floor by the back door. Crops are resized to the model
artifact's declared `input_size` (224) with bilinear resampling, then normalized with the
artifact's declared mean/std.

### 5.4 Detect (`detect/megadetector.py`)

Loading, exactly as proven in `scripts/probe_md_checkpoint.py`:

```python
sys.modules.setdefault("models", importlib.import_module("yolov5.models"))
sys.modules.setdefault("utils",  importlib.import_module("yolov5.utils"))
ckpt = torch.load(weights, map_location="cpu", weights_only=False)
model = (ckpt.get("ema") or ckpt["model"]).float().eval()
```

`weights_only=False` is required (the checkpoint pickles a live `models.yolo.DetectionModel`) and
is acceptable **only** because the file's provenance and exact byte length are pinned in
`RECON.md` (280,766,885 B, sha256 `94e88fe9…`). `MegaDetector.load()` verifies size and sha256
against those constants and fails fatally on mismatch, with the download URL in the message. The
`ema` weights are preferred over `model` when present, which is standard for yolov5 checkpoints.

Inference per image, using yolov5's own helpers (all three verified importable):
`letterbox(im, new_shape=1280, stride=64, auto=False)` → CHW float32 `/255` → `model(x)[0]` →
`non_max_suppression(pred, conf_thres=0.20, iou_thres=0.45, max_det=100)` →
`scale_boxes(letterboxed_shape, boxes, original_shape)`. `stride=64` matches the checkpoint's
reported `model.stride = [8,16,32,64]`.

`torch.set_num_threads(jobs)`, `torch.inference_mode()`. Budget from measured recon numbers:
~0.55 s/frame at 640 px, so **~2 s/image at 1280 px on 8 CPU cores**.

Output `Box(cls: "animal"|"person"|"vehicle", conf, x0, y0, x1, y1)` in original-image pixel
coordinates, plus `area_frac = ((x1-x0)*(y1-y0)) / (W*H)` stored for explainability in the GUI.

**Only `cls == "animal"` boxes participate in labelling.** `person` and `vehicle` boxes are
stored and drawn in the GUI but do not create labels; an image containing only people is
therefore `landscape` (or `junk` if blurry). This is a deliberate consequence of the closed label
taxonomy in `PLAN.md`; the catalog keeps the person/vehicle boxes so the GUI explains why.

`detect/scripted.py` implements the same protocol by reading a `<image>.boxes.json` sidecar from
a fixture directory, giving the e2e suite exact control over box geometry for dominance cases.
It is selected only by `--detector scripted`, which is documented as a testing affordance.

### 5.5 Species classification (`classify/own_model.py`)

Our model, loaded from a `.acmodel` artifact (§7.1). Inference: batch the crops of one image
(batch ≤ 8), forward, softmax with the artifact's calibration temperature, take top-5. Returns
`Prediction(common, scientific, score, top5, model_id)`.

Confidence gate: if `top1.score < min_species_confidence` (0.45) the box's species is
`unknown` — recorded with its top-5 so the GUI can still suggest, but never filed as a species.

### 5.6 Bird refinement (`classify/birds/`)

`taxonomy/labels.py` carries a `class` column; when a box's top-1 species rolls up to `Aves`
(the COCO-trained head's `bird` class, or any Aves entry), the crop is re-run through the
configured `BirdProvider`:

```python
class BirdProvider(Protocol):
    name: str
    def refine(self, crop: Image.Image, coarse: Prediction, gps: GpsPoint | None) -> BirdResult: ...
```

- **`own_bird_head` (DEFAULT).** Our CUB-200 head from `models/birds.acmodel`. Fully offline.
  Its top-1 replaces the coarse `bird` label when its calibrated score ≥
  `min_species_confidence`; otherwise the label stays `bird` if that itself cleared the gate,
  else `unknown`.
- **`ebird_enrich` (opt-in, `--bird-provider ebird_enrich`).** Not a classifier. It canonicalizes
  the predicted name against eBird taxonomy and down-ranks species that do not occur near the
  photo's EXIF GPS point. `api.ebird.org` answers `000` from this sandbox (RECON), so it **must
  degrade gracefully**: 4 s connect / 8 s read timeout, one attempt, responses cached in
  `<output>/.ebird-cache.json`. On any network failure it logs **one** `WARNING` per run, returns
  the unmodified `own_bird_head` result, and records `provider_status = "unreachable"` in the
  catalog. This is the single sanctioned graceful degradation in the design, and it is sanctioned
  precisely because the enrichment is *optional* — no required value is being substituted, and
  the catalog records that enrichment did not happen. Selecting the provider **without**
  `ANIMAL_CLASSIFIER_EBIRD_API_KEY` set is fatal at config time (§10.2), because there the
  missing value *is* required.
- **`hosted_bird_api` (documented stub).** Raises `AssetError` at config time with the exact
  environment variables and endpoint it would need. It never silently no-ops.

Merlin itself remains unusable as an API by design, not by sandbox accident: Merlin Photo ID is
an on-device model with no public photo-ID endpoint (`PLAN.md`, `RECON.md`). `ebird_enrich` is
the genuine Merlin-adjacent integration.

### 5.7 Decide (`decide.py`) — the dominance rule

Input: the list of **animal** boxes with their `area_frac` and per-box predictions, plus the
image's blur score. Output: exactly one label from the closed set
`{<species>, multiple, landscape, junk, unknown}`.

```python
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

Deliberate properties:

- **There is no `min_box_area` and no absolute box-area floor anywhere** — not in `decide.py`,
  not in `Config`, not in the detector, not in the crop step, not in the catalog. `dominance_ratio`
  is the only size comparison in the system, and it is purely *relative*. A distant impala in the
  corner of a lion portrait is handled exactly as intended: the lion's box area is far more than
  1.6× the impala's, so the image is filed as `lion`. Conversely a bird occupying 0.1% of a frame
  with nothing else detected is filed as that bird's species — an area floor would have thrown it
  away, which is the reason the floor is excluded.
- The comparison is written as a **multiplication**, never a division, so a degenerate
  zero-area second box makes the first dominant instead of raising `ZeroDivisionError`.
- Comparison is `>=`, so an exact-ratio tie resolves to dominant; equal areas (ratio 1.0 < 1.6)
  resolve to `multiple`.
- If the dominant box's species is below the confidence gate the label is `unknown`, **not**
  `multiple` — the dominance question was answered; only the identity is uncertain.
- The blur metric can only produce `junk` in the zero-animal branch. A blurry photo with a
  detected animal is still classified; MegaDetector firing is stronger evidence than a
  hand-tuned sharpness threshold.

### 5.8 Materialize (`materialize.py`)

Destination `<output_root>/<label>/<original_filename>`. `label` is validated against
`^[a-z0-9][a-z0-9_-]{0,63}$` before it ever becomes a path component; species names are
lowercased with spaces → `_` by `taxonomy.labels.slug()`.

All three modes are atomic via **temp-then-`os.replace`, with the temp file created in the
destination directory** (guaranteeing the same filesystem, so `os.replace` is a true atomic
rename):

| Mode | Implementation |
|---|---|
| `copy` (default) | `tempfile.mkstemp(dir=dest_dir, prefix=".ac-tmp-")` → stream copy → `shutil.copystat` (preserves mtime) → `os.replace(tmp, dest)` |
| `--link` | `os.symlink(abs_source, tmp)` → `os.replace(tmp, dest)` |
| `--hardlink` | `os.link(source, tmp)` → `os.replace(tmp, dest)` |

On any failure the temp file is removed in a `finally`, so a crash never leaves a partial image
at the real destination. Stale `.ac-tmp-*` files from a killed process are swept at the start of
the next run.

Collisions on `<label>/<name>`: if the existing path's content sha256 equals ours, the work is
already done → no write, counted as `already_present` (this is what makes re-runs cheap). If it
differs, the destination becomes `<stem>-<sha256[:8]><suffix>`. If *that* also exists with
different content — effectively impossible — it is a fatal `MaterializeError` rather than a
guessed third name.

`--hardlink` across filesystems raises `OSError EXDEV`; that is **fatal and actionable**: "SD card
and ~/animal_pics are on different filesystems; hardlinks cannot cross filesystems — use --link
for symlinks or omit the flag to copy."

`--dry-run` runs the whole pipeline, writes the planned destination into the catalog with
`status = "planned"`, and performs no destination writes at all.

**Re-tag (from the GUI)** is a move, per spec: materialize into `<new_label>/` first (same atomic
path), then unlink the old destination, then commit the override row. Ordering is deliberate — a
crash mid-way leaves the image present in two label dirs, which the next `classify` or `verify`
run reconciles, rather than losing the file.

### 5.9 Catalog (`catalog.py`)

SQLite at `<output_root>/.catalog.db`, `PRAGMA journal_mode=WAL`, `foreign_keys=ON`,
`busy_timeout=10000`. Schema version in a `meta` table; on a newer version the tool refuses to
run rather than corrupt data.

```sql
images(sha256 TEXT PRIMARY KEY, bytes INT, width INT, height INT,
       exif_datetime TEXT, gps_lat REAL, gps_lon REAL, blur_score REAL,
       label TEXT, label_source TEXT CHECK(label_source IN ('model','human')),
       confidence REAL, species_common TEXT, species_scientific TEXT,
       status TEXT CHECK(status IN ('planned','materializing','done','skipped','failed')),
       dest_path TEXT, mode TEXT, model_id TEXT, bird_provider TEXT,
       provider_status TEXT, run_id TEXT, first_seen TEXT, last_updated TEXT)
sources(sha256 TEXT, path TEXT PRIMARY KEY, mtime_ns INT, FOREIGN KEY(sha256) REFERENCES images)
boxes(id INTEGER PRIMARY KEY, sha256 TEXT, idx INT, cls TEXT, conf REAL,
      x0 REAL, y0 REAL, x1 REAL, y1 REAL, area_frac REAL,
      species_common TEXT, species_scientific TEXT, species_conf REAL,
      species_status TEXT, is_dominant INT, FOREIGN KEY(sha256) REFERENCES images)
candidates(box_id INT, rank INT, common TEXT, scientific TEXT, score REAL,
           FOREIGN KEY(box_id) REFERENCES boxes)
overrides(id INTEGER PRIMARY KEY, sha256 TEXT, old_label TEXT, new_label TEXT,
          created_at TEXT, note TEXT, FOREIGN KEY(sha256) REFERENCES images)
skipped(path TEXT PRIMARY KEY, reason TEXT, detail TEXT, run_id TEXT, seen_at TEXT)
runs(run_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, argv TEXT,
     config_json TEXT, n_total INT, n_done INT, n_skipped INT, n_failed INT, state TEXT)
```

`sources` is a separate table because one content hash legitimately has many source paths (the
same photo copied twice on the card) — it is classified once and filed once.

Idempotency and resume: `classify` computes sha256 first and skips any hash already at
`status='done'` unless `--reclassify`. Rows left at `materializing` or `planned` by an interrupted
run are re-processed, which is safe because materialization is content-addressed and
collision-aware. `--reclassify` re-runs inference but **keeps `label_source='human'` labels**
unless `--ignore-overrides` is also given; human corrections outrank the model by default, which
is what "later runs respect the correction" means.

## 6. GUI (`gui/`)

`animal-classifier gui` starts `uvicorn` on `127.0.0.1:8765` (`--host` is not offered; the
address is hardcoded to loopback). The app is created by `create_app(config)` so the e2e suite can
also drive it in-process.

| Route | Behaviour |
|---|---|
| `GET /` | Single HTML page; no framework, no build step |
| `GET /static/*` | `app.js`, `app.css` |
| `GET /api/labels` | `[{label, count}]` for the sidebar |
| `GET /api/images` | Query: `label`, `min_conf`, `max_conf`, `date_from`, `date_to`, `q`, `limit` (≤200, default 60), `offset`. Returns rows with boxes, top-5 candidates, blur score, provider status |
| `GET /api/images/{sha256}/thumb` | 320 px JPEG, generated on demand and cached at `<output_root>/.thumbs/<sha256[:2]>/<sha256>.jpg` |
| `GET /api/images/{sha256}/full` | Full image bytes, `Content-Type` from the real format |
| `POST /api/images/{sha256}/label` | `{"label": "...", "note": "..."}` → moves the file (§5.8), writes the override, returns the new `dest_path` |
| `GET /api/run` | Latest `runs` row → live progress while a `classify` run is in flight |

Frontend: a thumbnail grid grouped by label with per-label counts in a sidebar, label/confidence/
date filters, and a detail view drawing boxes on a `<canvas>` overlay scaled from the stored pixel
coordinates and the stored `width`/`height`, with each box captioned `cls`, `area_frac` and its
top-1 species. The detail view shows the top-3 candidates and a label picker; picking a label
`POST`s and updates the grid in place.

Byte-serving safety: paths are never taken from the request. The handler looks up `dest_path` in
the catalog, resolves it, and asserts it is under `output_root` (or under `source_root` in
read-only mode for a `--dry-run` catalog) before opening it read-only. A `sha256` path parameter
that is not 64 hex characters is a 422; an unknown hash is a 404.

New labels: `POST` accepts any label already present in the taxonomy or catalog. An unrecognized
label is rejected 422 unless the GUI was started with `--allow-new-labels`, which is how a user
adds a species the model does not know yet.

## 7. Species model and training subsystem

### 7.1 The `.acmodel` artifact (`classify/artifact.py`)

A single `torch.save` dict — self-describing so inference never has to guess:

```python
{"format_version": 1, "model_id": "species-effb0-20260915-a1b2c3",
 "arch": "efficientnet_b0", "input_size": 224,
 "normalize": {"mean": [...], "std": [...]},
 "labels": [{"key": "zebra", "common": "Zebra", "scientific": "Equus quagga", "class": "Mammalia"}, ...],
 "state_dict": {...},              # backbone + head, one flat dict
 "temperature": 1.37,              # calibration fitted on val
 "train": {"dataset": "coco-animals", "manifest_sha256": "...", "epochs": 8,
           "val_top1": 0.91, "val_top5": 0.99, "backbone_weights": "efficientnet_b0_ra-3dd342df.pth",
           "created_at": "..."}}
```

Loading validates `format_version` (mismatch → fatal `AssetError` naming the expected version and
`animal-classifier train` as the fix), rebuilds `timm.create_model(arch, pretrained=False,
num_classes=len(labels))`, and loads the state dict with `strict=True`. `model_id` is written into
every `images` row, so any historical label can be traced to the exact model that produced it.

### 7.2 Manifests and datasets

Manifest = JSONL, one line per training sample; `path` is absolute or manifest-relative, `box` is
optional and crops on load:

```json
{"path": "data/coco/val2017/000000397133.jpg", "label": "zebra", "box": [12.0, 40.5, 310.2, 288.9], "split": "train"}
```

`manifest.py` validates every line and **fails on the first bad one** with the line number and the
offending field — a training run must never quietly train on a truncated manifest. Missing files
are fatal, not skipped.

Two builders produce real training data from the archives already in `data/raw/`:

- `scripts/build_coco_manifest.py` reads `annotations/instances_val2017.json`, extracts the 10
  animal categories (ids 16–25: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
  giraffe), drops `iscrowd=1` instances, and emits one sample per instance with its `bbox`
  converted to `x0,y0,x1,y1`. This yields the ~2,700 instances counted in RECON, and is exactly
  the "crops from boxes make good training input" idea from `PLAN.md`.
- `scripts/build_cub_manifest.py` extracts CUB-200-2011 and emits one sample per image with the
  species from its directory name and the box from `bounding_boxes.txt`, honouring the official
  `train_test_split.txt`.

`split`: honoured when present in the manifest (CUB), otherwise assigned deterministically by
`sha1(path)` bucketing — 80/20, **stratified per class**, so a rerun produces the identical split
and no class vanishes from val. `dataset.py` crops with the same margin as inference (0.08) so
train and inference see the same framing.

### 7.3 Transforms and training loop

Train: `RandomResizedCrop(input_size, scale=(0.65, 1.0))`, `RandomHorizontalFlip`,
`ColorJitter(0.2, 0.2, 0.2, 0.05)`, normalize. Eval: `Resize(input_size * 1.14)` →
`CenterCrop(input_size)` → normalize. Horizontal flip only — vertical flips do not occur in
wildlife photography and would waste capacity.

Two-stage transfer learning (`trainer.py`), the path `PLAN.md` specifies:

1. **Head only** — backbone frozen (`requires_grad=False`, `eval()` so BatchNorm stats hold),
   `AdamW(lr=3e-3, wd=1e-4)`, cosine schedule, `epochs_head` (default 3).
2. **Finetune** — unfreeze the last `unfreeze_blocks` (default 2) stages, `lr=3e-4` for those and
   `3e-5` elsewhere, `epochs_finetune` (default 5).

Loss `CrossEntropyLoss(label_smoothing=0.1)`. Long-tail handling by
`WeightedRandomSampler(1/sqrt(class_count))`. `torch.set_num_threads(jobs)`; CPU-only here, and
`--device auto` picks CUDA if it is ever present.

Checkpoint after every epoch to `<out>.ckpt` (model, optimizer, scheduler, epoch, RNG states,
best val top-1, manifest sha256). `--resume` restores all of it and refuses to resume across a
different manifest sha256 or arch. Best-val weights are what get exported.

### 7.4 Eval and calibration (`training/evaluate.py`)

`animal-classifier eval --model <artifact> --manifest <m> --split val` writes `metrics.json`
(top-1, top-5, macro/per-class recall, support) and `confusion_matrix.csv`, and prints a summary
table. Confidence honesty matters because `min_species_confidence` gates the `unknown` label, so
eval also fits a single temperature by LBFGS on the val split's NLL and writes it into the
artifact — an uncalibrated softmax would make the 0.45 gate meaningless.

### 7.5 `export-trainset`

`animal-classifier export-trainset --output ~/animal_pics --destination trainset.jsonl` walks the
catalog for `label_source='human'` (and optionally high-confidence model labels via
`--include-model-labels --min-conf`), emits a manifest of the dominant box of each image, and thus
closes the loop: GUI corrections become the next training set. Source images are read read-only.

### 7.6 The synthetic smoke path

`training/tinycnn.py` is a small from-scratch CNN (4 conv blocks, ~180 k params) and
`--dataset synthetic` generates a deterministic seeded set of coloured-shape images. This exists
so the e2e suite has a training test that finishes in seconds without the 1.1 GB CUB archive. It
is explicitly **not** the proof of the training subsystem — E7 (§11) is, by finetuning
`efficientnet_b0` on real COCO/CUB crops.

## 8. CLI surface (`cli.py`)

typer 0.25.x compatible. Global: `--config`, `--verbose/-v`, `--quiet`.

```
classify SOURCE
  -o/--output PATH            (default ~/animal_pics)
  --link | --hardlink         (mutually exclusive; default copy)
  --dry-run  --reclassify  --ignore-overrides
  --raw                       enable RAW decode (requires the `raw` extra)
  --dominance-ratio FLOAT     (default 1.6)
  --min-confidence FLOAT      (default 0.45)
  --blur-threshold FLOAT      (default 100.0)
  --detector [megadetector|scripted]   --detector-weights PATH  --detector-confidence FLOAT
  --species-model PATH  --bird-model PATH
  --bird-provider [own_bird_head|ebird_enrich|hosted_bird_api]
  --jobs INT  --limit INT  --device [auto|cpu|cuda]
gui   -o/--output PATH  --port INT (default 8765)  --allow-new-labels
train --manifest PATH  --out PATH  --arch TEXT  --backbone-weights PATH
      --dataset [manifest|synthetic]  --epochs-head INT  --epochs-finetune INT
      --batch-size INT  --input-size INT  --unfreeze-blocks INT  --resume  --jobs INT  --seed INT
eval  --model PATH  --manifest PATH  --split [train|val]  --report-dir PATH  --calibrate
export-trainset -o/--output PATH  --destination PATH  --include-model-labels  --min-conf FLOAT
verify  -o/--output PATH  --json
```

`verify` checks and reports, exiting non-zero if anything required is missing: detector weights
present with the expected size + sha256 and loadable; species/bird artifacts present, loadable,
`format_version` accepted; backbone weights present; `output_root` writable and not nested with a
source; catalog openable and at a known schema version; catalog/filesystem reconciliation (rows
whose `dest_path` is missing, files with no row, images in a directory that disagrees with their
row's label); bird provider configuration; and an *informational* reachability probe of the
optional online providers (a failure here is reported, never fatal).

## 9. Concurrency and throughput

Decode + sha256 + blur run in a `ThreadPoolExecutor(jobs)` (Pillow and hashlib release the GIL);
detection and classification run on the main thread with `torch.set_num_threads(jobs)`, because
torch already parallelizes internally and competing pools would thrash 8 cores. Images are
processed in a bounded pipeline (queue depth `2 * jobs`) so a 64 GB card never loads into RAM.
Catalog writes are batched per image in one transaction from the main thread only — single-writer,
which is what keeps SQLite happy. At ~2 s/image, 5,000 images ≈ 2.8 h; `--limit` and resume make
that tolerable, and progress is written to the `runs` row so the GUI can show it live.

## 10. Error handling and validation

### 10.1 Per-operation failure table

Rule, applied everywhere: **a missing or unexpected *required* value fails loudly with an
actionable message; it is never replaced by a default.** A per-image failure does not abort the
run — it is recorded as `failed`/`skipped` with its reason and the run's exit code becomes 4.

| Operation | Failure | Class | Caller receives | Log |
|---|---|---|---|---|
| Load config | unknown TOML key, bad type, out-of-range value | fatal | `ConfigError`, exit 3, key + valid range | `ERROR` |
| Load config | `--config` path missing | fatal | exit 3, the path | `ERROR` |
| Load config | `output_root` nested with `source` | fatal | exit 3, both resolved paths | `ERROR` |
| Load config | `ebird_enrich` without API key; `hosted_bird_api` selected | fatal | exit 3, the env var names needed | `ERROR` |
| Detector weights | file absent / wrong size / sha256 mismatch | fatal | `AssetError`, exit 3, expected bytes + sha256 + download URL | `ERROR` |
| Detector weights | unpickle fails (`ModuleNotFoundError: models`) | fatal | exit 3, "reinstall with `uv sync`; yolov5 must be installed" | `ERROR` |
| Model artifact | absent, unreadable, `format_version` too new, `strict` key mismatch | fatal | `AssetError`, exit 3, pointing at `animal-classifier train` | `ERROR` |
| Scan | source missing / not a dir / not readable | fatal | exit 3 | `ERROR` |
| Scan | permission denied on a subdir | recoverable | `skipped(reason="unreadable")` | `WARNING` |
| Decode | truncated/corrupt image, zero bytes, > 400 MP | recoverable | `skipped(reason=…)`, exit 4 | `WARNING` |
| Decode | `.heic` and `pi_heif` missing | fatal | exit 3, the exact `uv` command to fix | `ERROR` |
| Decode | `--raw` without the `raw` extra | fatal | exit 3, `uv pip install -e '.[raw]'` | `ERROR` |
| Detect | inference raises (OOM, malformed tensor) | recoverable | `failed`, exit 4 | `ERROR` + traceback at `-v` |
| Classify | crop degenerate (< 2 px) | recoverable | box kept for dominance, `species_status='degenerate'` | `DEBUG` |
| Bird provider | `ebird_enrich` unreachable / timeout / 429 / bad JSON | **degrade** (optional enrichment) | own-head result unchanged, `provider_status='unreachable'` | one `WARNING` per run |
| Bird provider | `ebird_enrich` 401/403 (bad key) | fatal | exit 3 — a rejected credential is a real misconfiguration | `ERROR` |
| Materialize | destination not writable / out of space (`ENOSPC`) | fatal | `MaterializeError`, exit 1, temp cleaned up | `ERROR` |
| Materialize | `--hardlink` across filesystems (`EXDEV`) | fatal | exit 3, suggests `--link` or copy | `ERROR` |
| Materialize | collision, same content | not an error | `already_present` | `DEBUG` |
| Materialize | collision, different content, hash-suffixed name also taken | fatal | exit 1, both paths | `ERROR` |
| Catalog | schema version newer than this build | fatal | exit 3, refuses to touch the DB | `ERROR` |
| Catalog | `database is locked` past `busy_timeout` | fatal after 3 retries (0.5/1/2 s) | exit 1, "another run or GUI may be writing" | `ERROR` |
| GUI re-tag | unknown label without `--allow-new-labels`; invalid label chars; bad sha256 | rejected | HTTP 422 with the reason | `WARNING` |
| GUI re-tag | source file vanished between listing and move | rejected | HTTP 409, row marked `failed` | `ERROR` |
| Train | manifest bad line / missing image / empty class | fatal | exit 3, line number + field | `ERROR` |
| Train | `--resume` against a different manifest sha256 or arch | fatal | exit 3, both ids | `ERROR` |
| Train | backbone weights absent, or `pretrained=True` attempted | fatal | exit 3, local path expected + RECON note | `ERROR` |

Exit codes: `0` success · `1` unexpected/IO failure · `2` typer usage · `3` missing or invalid
required asset/config · `4` completed with per-image failures or skips.

Logging: `logging` to stderr, `INFO` default, `DEBUG` at `-v`, `WARNING`+ only at `--quiet`.
Never `print` for diagnostics; stdout carries only user-facing results (`verify --json`).

### 10.2 Validation of every external input

| Input | Rule | On failure |
|---|---|---|
| `SOURCE` | required; exists; is a dir; readable | fatal, exit 3 |
| `--output` | `~` expanded; created (parents) if absent; must be writable; must not be nested with `SOURCE` | fatal, exit 3 |
| `--link` / `--hardlink` | mutually exclusive | fatal, exit 2 |
| `dominance_ratio` | float, finite, `>= 1.0` | fatal, exit 3 (`< 1.0` would make the smaller box dominant) |
| `min_species_confidence` | float in `[0.0, 1.0]` | fatal, exit 3 |
| `detector_confidence`, `detector_iou` | float in `(0.0, 1.0]` | fatal, exit 3 |
| `blur_threshold` | float `> 0` | fatal, exit 3 |
| `crop_margin` | float in `[0.0, 0.5]` | fatal, exit 3 |
| `detector_image_size` | int, multiple of 64, `[320, 2048]` | fatal, exit 3 |
| `jobs` | int `>= 1`, warn above `os.cpu_count()` | fatal, exit 3 |
| `--limit` | int `>= 1` when given | fatal, exit 3 |
| `--port` | int in `[1024, 65535]` | fatal, exit 3 |
| weights/model/manifest paths | exist; readable; expected magic/keys | fatal, exit 3 |
| EXIF datetime / GPS | genuinely optional; parsed defensively; unparseable → `NULL` + `DEBUG` | not an error |
| eBird API key | required, non-empty **iff** `ebird_enrich` selected | fatal, exit 3 |
| GUI `label` | `^[a-z0-9][a-z0-9_-]{0,63}$`, and known unless `--allow-new-labels` | HTTP 422 |
| GUI `sha256` | exactly 64 hex chars | HTTP 422 |
| GUI `limit`/`offset` | int, `limit` in `[1, 200]`, `offset >= 0` | HTTP 422 |
| GUI `min_conf`/`max_conf` | float in `[0, 1]`, `min <= max` | HTTP 422 |
| Manifest line | object with `path` (str, resolvable) + `label` (non-empty str); `box` = 4 finite numbers, `x0<x1`, `y0<y1`; `split` ∈ {train,val} | fatal, exit 3, with line number |

### 10.3 Invariants and their owning layer

| # | Invariant | Owner | Why there |
|---|---|---|---|
| I1 | The source tree is never written, moved, renamed or deleted | `config.py` (nesting guard) + `images.open_source` (`"rb"` only) + `materialize.py` (refuses paths outside `output_root`) | Writes exist in exactly one module, so the guard has one place to live; E1 proves it |
| I2 | **No absolute box-area floor exists**; `dominance_ratio` is the only size gate | `decide.py`, with `config.py` rejecting unknown keys | Keeping the rule in one pure function makes its absence auditable at a glance; the rejection of unknown config keys stops a floor being reintroduced by configuration |
| I3 | Every image gets exactly one label from the closed set | `decide.py` returns a `Label` enum/species key; `catalog` has a CHECK on `label_source` | A single pure decision function, so no caller can invent a label |
| I4 | An image's `dest_path` parent directory always equals its catalog `label` | `materialize.py` + `catalog.py` in one write order: row → `materializing` → file → `done` | Crash-consistency needs the intent recorded *before* the filesystem changes; `verify` reconciles |
| I5 | One destination file per (content, label) | `materialize.py` content-addressed collision check | Only that module knows both the hash and the destination |
| I6 | A human override outranks the model on later runs | `catalog.py` (`label_source`) read by `classify` | The durable record is the DB, so the DB owns precedence |
| I7 | No required value is ever silently defaulted | `config.py` validation + `errors.py` (no bare `except`, no `.get(k, fallback)` for required keys) | Fail-loud must be structural, not a convention |

## 11. Testability — end-to-end tests only

`tests/e2e/`, run with `uv run pytest tests/e2e`. **No unit tests anywhere**, no `tests/unit`,
no in-process assertions on private helpers. Every test drives a real entry point — the installed
`animal-classifier` console script via `subprocess`, or the real ASGI app over HTTP — and asserts
observable outcomes: the output tree, the catalog contents, exit codes, stderr, HTTP responses.

Fixtures are built by `scripts/make_e2e_fixtures.py` from the archives already in `data/raw/`
(session-scoped, cached under `tests/e2e/_fixtures/`, gitignored). If `data/raw` is missing the
fixture builder **fails with the exact download commands** rather than skipping — a silently
skipped suite is worse than a red one. Real-COCO images give real ground truth (RECON: 123
clear-dominance vs 194 ambiguous multi-animal images).

| # | Test | Drives | Asserts |
|---|---|---|---|
| E1 | Source tree immutability | `classify`, then GUI re-tag, then `export-trainset` | Pre/post snapshot of `(path, size, mtime_ns, sha256)` over the whole fixture card is byte-identical |
| E2 | Happy path, copy mode | `classify` on a mixed fixture card | Exact output tree; one dir per label; catalog `images`/`boxes`/`sources` rows; exit 0 |
| E3 | `--link` and `--hardlink` | `classify` twice into separate roots | `Path.is_symlink()` and resolved target; `st_nlink == 2` and equal `st_ino`; both destinations readable |
| E4 | Dominance on **real** COCO images with **real** MegaDetector weights | `classify --detector megadetector` | Images COCO says have a clear area winner get that species; images with ratio < 1.5 get `multiple`; no image gets a label outside the closed set |
| E5 | Dominance boundary cases | `classify --detector scripted` | ratio exactly 1.6 → dominant; 1.59 → `multiple`; equal areas → `multiple`; zero-area second box → dominant, no exception |
| E6 | **No area floor** | `classify --detector scripted` with a single animal box covering 0.05% of the frame, and a 0.1%-vs-0.5% pair | The lone tiny animal is filed as its species (never dropped, never `landscape`); the tiny pair still resolves by ratio; `classify --help` and `verify --json` expose no `min_box_area`-like option |
| E7 | Real training | `build_coco_manifest.py` → `train --arch efficientnet_b0 --backbone-weights models/backbones/efficientnet_b0_ra-3dd342df.pth` (subset: 6 classes, 128 px, few epochs) → `eval` → `classify --species-model <new artifact>` | Artifact written with `format_version`, labels and `train` metadata; val top-1 beats the 1/6 chance floor by a margin asserted numerically; the exported model then labels images through the real CLI |
| E8 | Bird path | `build_cub_manifest.py` → `train` a 5-species CUB head → `classify` a CUB image | Label is a CUB species; catalog `bird_provider = own_bird_head`; box species matches |
| E9 | Training resume | `train` interrupted after epoch 1, then `--resume`; and `--resume` with a different manifest | Resumes at the right epoch and finishes; the mismatched resume exits 3 with both ids |
| E10 | Synthetic smoke | `train --dataset synthetic --arch tinycnn` → `eval` | Full train→eval→export→infer path in seconds; accuracy above chance |
| E11 | `landscape` / `junk` | `classify` on a sharp animal-free photo and a heavily blurred one | `landscape` vs `junk`; `blur_score` stored for both |
| E12 | `unknown` | `classify --min-confidence 0.999` | Label `unknown`; top-5 candidates still recorded; file still materialized |
| E13 | Idempotency / resume | `classify` twice; kill mid-run and rerun | Second run writes nothing new, `already_present` counted, no duplicate rows; killed run resumes and completes |
| E14 | Duplicate content | Same photo twice under different names | Classified once, one destination file, two `sources` rows |
| E15 | Collision, different content | Two different images sharing a filename in one label | Second becomes `<stem>-<sha8><suffix>`; both intact |
| E16 | Formats and skips | Card with JPEG/PNG/TIFF/HEIC, an `.mp4`, a `.cr2`, a 0-byte file, a truncated JPEG | The four formats processed; `.mp4` → `skipped(video)`; `.cr2` → `raw_not_enabled` (and processed with `--raw` when the extra is present, else fatal + actionable); 0-byte and truncated → `skipped`; exit 4 |
| E17 | `--dry-run` | `classify --dry-run` | Destination tree has no image files; catalog rows are `planned`; exit 0 |
| E18 | GUI browse | real uvicorn subprocess + `httpx` | `/` serves HTML; `/api/labels` counts match the tree; `/api/images?label=…` filters; thumb and full bytes decode as images; box coordinates are inside the stored dimensions |
| E19 | GUI re-tag moves the file | `POST /api/images/{sha}/label` | File is gone from the old label dir, present in the new one, content unchanged; `overrides` row written; `label_source='human'` |
| E20 | Override survives reclassify | E19, then `classify --reclassify`, then `--reclassify --ignore-overrides` | Human label kept in the first case; model label restored in the second |
| E21 | GUI validation | bad label, bad sha256, out-of-range `limit`, unknown hash | 422/422/422/404; nothing on disk changed |
| E22 | Fail-loud config | missing weights; unknown TOML key; `dominance_ratio=0.5`; `hosted_bird_api`; `ebird_enrich` without a key; output nested inside source | Exit 3 each, with the offending name in stderr, and **no** destination directory created |
| E23 | `ebird_enrich` degrades | `classify --bird-provider ebird_enrich` with a dummy key (host is `000` here) | Run succeeds; labels equal the `own_bird_head` run; exactly one warning; `provider_status='unreachable'` |
| E24 | `verify` | `verify --json` on a good tree, then after deleting a materialized file | Exit 0 and all-present report; then non-zero naming the missing `dest_path` |
| E25 | `export-trainset` round trip | E19 → `export-trainset` → `train --manifest` | Manifest lines validate, include the human label and the dominant box, and train consumes them |

What is *hard* to e2e-test, and the deliberate choice made: browser-driven GUI testing needs a
Playwright browser download from a host not probed in RECON, so the GUI is tested at the HTTP +
filesystem boundary (E18–E21) instead. That is where the risky behaviour lives — the re-tag
**move** and the validation — so the coverage loss is cosmetic rendering only. Full-dataset
training accuracy is likewise not asserted; E7 asserts a real but CPU-sized finetune beats chance
by a stated margin, and full runs are the user's `animal-classifier train` on their own machine.

## 12. Sequenced build order

1. `errors.py`, `config.py`, `catalog.py`, `taxonomy/` + static label tables.
2. `scan.py`, `images.py`, `materialize.py` → `classify` end-to-end with `detect/scripted.py` and
   a pass-through classifier. Lands E1, E2, E3, E5, E6, E11, E13–E17, E22.
3. `detect/megadetector.py` (weights verification, alias shim, letterbox/NMS). Lands E4.
4. `training/` + `scripts/build_*_manifest.py` + `train`/`eval`. Lands E7–E10.
5. `classify/own_model.py`, `classify/birds/*`, wire `decide` thresholds. Lands E8, E12, E23.
6. `gui/`. Lands E18–E21.
7. `export-trainset`, `verify`, README. Lands E24, E25.

## 13. Open assumptions

1. `~/animal_pics` and the SD card are on the same filesystem often enough that `--hardlink` is
   useful; when they are not, the `EXDEV` message tells the user what to do instead.
2. The COCO-trained 10-class head is a genuine, honestly-scoped safari species model (zebra,
   elephant, giraffe, bear, bird, plus horse/sheep/cow/cat/dog) — it is the best real labelled
   data reachable in-sandbox, and `train` + `export-trainset` are the documented path to a wider
   label set on the user's machine, where LILA/Snapshot Serengeti and HuggingFace are reachable.
   Species outside that set will come back as `unknown` rather than as a confident wrong guess,
   which is the intended failure mode.
3. Scientific names ship as a static in-repo table (no taxonomy API is reachable). The 10 mammal/
   bird COCO classes get hand-authored binomials; CUB-200 ships common names with `scientific`
   nullable where no reliable offline source exists — `NULL` here is honest absence of an optional
   field, not a substituted default.
