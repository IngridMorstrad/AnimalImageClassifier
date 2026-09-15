# Safari Image Classifier — Technical Design

Status: **iteration 2**, revised 2026-09-15 after `docs/design-review.json` (verdict
`CHANGES_REQUESTED`, 2 HIGH / 15 MEDIUM / 8 NIT). Every finding is resolved; §14 maps each one to
where it was addressed. Grounded in `docs/PLAN.md` (agreed behaviour) and `docs/RECON.md` (what is
actually reachable). Every asset named here has been downloaded, loaded and executed in-sandbox —
no blocked host is assumed available anywhere in this document.

Facts newly measured for this iteration (all re-measured, not inherited):

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
    labels.py            # static table loader, common ↔ scientific, rank, class rollup (Aves)
    data/coco_animals.csv, data/cub200.csv
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
dropped: `hidden`, `system_dir` (`.Trash*`, `.thumbnails`, `.Spotlight-V100`, `__MACOSX`),
`unsupported_extension`, `format_disabled`, `video`, `raw_not_enabled`, `zero_bytes`, `unreadable`,
`symlink_loop`, `too_large`.

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
raised to 400 MP; anything larger is a `DecodeError` → `skipped(reason="too_large")`.

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

Output `Box(cls: "animal"|"person"|"vehicle", conf, x0, y0, x1, y1)` in original-image pixel
coordinates, plus `area_frac = ((x1-x0)*(y1-y0)) / (W*H)` stored for explainability in the GUI.

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
(batch ≤ 8), forward, softmax with the artifact's calibration temperature, take top-5.

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

- **`own_bird_head` (DEFAULT).** Our CUB-200 head from `models/birds.acmodel`. Fully offline. Returns
  `status="refined"` when its calibrated top-1 ≥ `min_species_confidence`; otherwise
  `status="kept_coarse"` (the label then stays the coarse `bird` if that cleared the gate, else
  `unknown`).
- **`ebird_enrich` (opt-in, `--bird-provider ebird_enrich`).** Not a classifier; a re-ranker.
  - **No EXIF GPS** → canonicalize names against the static eBird-style table only, ranking
    **unchanged**, `status="no_gps"`, `provider_status="no_gps"`. No network call is attempted.
  - **With GPS** → one cached call to
    `https://api.ebird.org/v2/data/obs/geo/recent?lat=<lat>&lng=<lon>&dist=50&back=30` (4 s connect
    / 8 s read timeout, one attempt, no retry), responses cached in `<output_root>/.ebird-cache.json`
    keyed by `(round(lat,2), round(lon,2))`. Every candidate **absent** from the response has its
    score multiplied by **0.25**; the top-5 is then re-sorted and `min_species_confidence` is
    re-applied — so a demoted top-1 can legitimately become `unknown`. `status="refined"` if the
    order or the gated outcome changed, else `"kept_coarse"`.
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
    return taxonomy.slug(box.species_common)      # e.g. "Plains Zebra" -> "plains_zebra"

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
species (E26).

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
`.ac-tmp-*`. Species names are lowercased with spaces → `_` by `taxonomy.labels.slug()` and
validated against the same regex before they ever become a path component.

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

`--dry-run` runs the whole pipeline, writes the planned destination into the catalog with
`status = "planned"`, and performs no destination writes at all.

**Re-tag (from the GUI)** is a move, per spec: materialize into `<new_label>/` first (same atomic
path), then unlink the old destination, then commit the override row. Ordering is deliberate — a
crash mid-way leaves the image present in two label dirs rather than losing it, and that duplicate is
reconciled by **`verify --fix`** (§8), which is the only code path that resolves it. `classify` skips
`status='done'` hashes and therefore does *not* reconcile; the iteration-1 claim that it did was
wrong and is withdrawn.

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
           FOREIGN KEY(box_id) REFERENCES boxes)
overrides(id INTEGER PRIMARY KEY, sha256 TEXT, old_label TEXT, new_label TEXT,
          created_at TEXT, note TEXT, FOREIGN KEY(sha256) REFERENCES images)
skipped(path TEXT PRIMARY KEY, reason TEXT, detail TEXT, run_id TEXT, seen_at TEXT)
runs(run_id TEXT PRIMARY KEY, source_root TEXT NOT NULL, started_at TEXT, finished_at TEXT,
     argv TEXT, config_json TEXT, n_total INT, n_done INT, n_skipped INT, n_failed INT, state TEXT)
```

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
tree for later commands: `gui` and `verify` take no `SOURCE` argument and read it from the newest
`runs` row (§6, §8).

**Idempotency, resume and `--limit`.** `classify` computes sha256 first and skips any hash already at
`status='done'` unless `--reclassify`. Rows left at `materializing` or `planned` by an interrupted run
are re-processed, which is safe because materialization is content-addressed and collision-aware.
**`--limit N` caps images newly submitted to inference**; hashes already at `status='done'` are
skipped *without consuming budget*, so repeated `--limit` runs advance through the card instead of
re-examining its first N files forever (E13).

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
all. A re-tag `POST` uses a short-lived write connection; on lock contention it changes **nothing on
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
| `GET /api/labels` | `[{label, count}]` for the sidebar — enumerated from `output_root` subdirectories matching the label regex, tool-internal entries ignored (§5.8) |
| `GET /api/images` | Query: `label`, `min_conf`, `max_conf`, `date_from`, `date_to`, `q`, `limit` (≤200, default 60), `offset`. Rows include boxes, top-5 candidates, `species_rank`, blur score + `blur_ref_edge`, provider status |
| `GET /api/images/{sha256}/thumb` | 320 px JPEG, generated on demand, cached at `<output_root>/.thumbs/<sha256[:2]>/<sha256>.jpg` |
| `GET /api/images/{sha256}/full` | Full image bytes from `dest_path` under `output_root`; `Content-Type` from the real format |
| `POST /api/images/{sha256}/label` | `{"label": "...", "note": "..."}` → moves the file (§5.8), writes the override, returns the new `dest_path` |
| `GET /api/run` | Newest `runs` row (including `source_root`) → live progress while a `classify` run is in flight |

**Byte-serving safety.** Paths are never taken from the request. The handler looks up `dest_path` in
the catalog, resolves it, and asserts it is under `output_root` before opening it read-only. The GUI
**never serves bytes from the source tree** — the iteration-1 "read-only source mode" is removed,
because nothing supplied a source root to it. A row with `status='planned'` (a `--dry-run` catalog)
therefore has no materialized file and `/full` returns
`409 {"error": "image was planned by --dry-run and has no materialized file"}`. A `sha256` path
parameter that is not 64 hex characters is a 422; an unknown hash is a 404.

Frontend: a thumbnail grid grouped by label with per-label counts in a sidebar, label/confidence/date
filters, and a detail view drawing boxes on a `<canvas>` overlay scaled from the stored pixel
coordinates and the stored `width`/`height`. Each box caption shows `cls`, `area_frac`, its top-1
species **and its rank** (`species` / `class` / …), so a coarse `bird` is visibly coarse. The detail
view shows the top-3 candidates and a label picker; picking a label `POST`s and updates the grid in
place.

New labels: `POST` accepts any label already present in the taxonomy or catalog. An unrecognized
label is rejected 422 unless the GUI was started with `--allow-new-labels`, which is how a user adds
a species the model does not know yet.

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
 "temperature": 1.37,              # calibration, fitted on val
 "train": {"dataset": "coco-animals", "manifest_sha256": "...", "epochs": 8,
           "val_top1": 0.91, "val_top5": 0.99,
           "backbone_weights": "efficientnet_b0_ra-3dd342df.pth",
           "calibrated_from": None, "created_at": "..."}}
```

(The `val_top1` above is an illustrative sample value, not a target; the asserted threshold is E7's.)

Loading validates `format_version` (mismatch → fatal `AssetError` naming the expected version and
`animal-classifier train` as the fix), rebuilds `timm.create_model(arch, pretrained=False,
num_classes=len(labels))`, and loads the state dict with `strict=True`. Every label entry carries
`rank`, which flows into `Prediction.rank_level`, `boxes.species_rank` and the GUI caption.

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
`--include-model-labels --min-conf`), emits a manifest of the dominant box of each image, and thus
closes the loop: GUI corrections become the next training set. Source images are read read-only.

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
verify  -o/--output PATH  --json  --fix
```

`classify --help` carries the person/vehicle → `landscape` note from §5.4.

**`verify`** checks and reports, exiting non-zero if anything required is missing or inconsistent:
detector weights present with the expected size + sha256 and loadable; species/bird artifacts present,
loadable, `format_version` accepted; backbone weights present; catalog openable at a known schema
version; `output_root` writable; **`output_root` not nested with the source root read from the newest
`runs` row** — with no `runs` row this specific check is reported as `skipped(no_runs)`, not passed and
not failed; catalog/filesystem reconciliation (rows whose `dest_path` is missing, files with no row,
images in a directory that disagrees with their row's label, ignoring tool-internal entries per §5.8);
bird provider configuration; and an *informational* reachability probe of the optional online providers
(a failure there is reported, never fatal).

**`verify --fix`** is the only reconciler, and it is deliberately narrow. For an image whose row label
disagrees with a copy found under a different label directory, **and** where an `overrides` row
justifies the current label (the interrupted-re-tag case from §5.8), it deletes the **stale copy under
the old label directory** and logs it at `INFO`. It never touches anything under the source root, never
deletes a file that the catalog does not explain, and never invents a label. Without `--fix`, the same
condition is reported and the exit code is non-zero.

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
`--limit` (which consumes budget only on newly processed images) and resume make that tolerable, and
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
| Decode | truncated/corrupt image, zero bytes, > 400 MP | recoverable | `skipped(reason=…)`, exit 4 | `WARNING` |
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
| GUI `/full` | row `status='planned'` (dry-run catalog) | rejected | HTTP 409, "planned by --dry-run, no materialized file" | `DEBUG` |
| GUI re-tag | unknown label without `--allow-new-labels`; invalid label chars; bad sha256 | rejected | HTTP 422 with the reason | `WARNING` |
| GUI re-tag | destination file vanished between listing and move | rejected | HTTP 409, row marked `failed` | `ERROR` |
| Train | manifest bad line / missing image / empty class | fatal | exit 3, line number + field | `ERROR` |
| Train | `--resume` against a different manifest sha256 or arch | fatal | exit 3, both ids | `ERROR` |
| Train | backbone weights absent, or `pretrained=True` attempted | fatal | exit 3, local path expected + RECON note | `ERROR` |
| Eval | `--calibrate` without `--out`, or `--out` equal to an existing artifact path | fatal | exit 3, artifacts are immutable once recorded | `ERROR` |
| Verify | inconsistency found, `--fix` not given | reported | exit non-zero, the offending paths | `ERROR` |

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
| `jobs` | int `>= 1`, warn above `os.cpu_count()` | fatal, exit 3 |
| `--limit` | int `>= 1` when given | fatal, exit 3 |
| `--port` | int in `[1024, 65535]` | fatal, exit 3 |
| weights/model/manifest paths | exist; readable; expected magic/keys | fatal, exit 3 |
| EXIF datetime / GPS | genuinely optional; parsed defensively; unparseable → `NULL` + `DEBUG` | not an error |
| GPS values when present | `lat ∈ [-90, 90]`, `lon ∈ [-180, 180]`; out of range → `NULL` + `WARNING` | not an error |
| eBird API key | required, non-empty **iff** `ebird_enrich` selected | fatal, exit 3 |
| GUI `label` | `^[a-z0-9][a-z0-9_-]{0,63}$`, and known unless `--allow-new-labels` | HTTP 422 |
| GUI `sha256` | exactly 64 hex chars | HTTP 422 |
| GUI `limit`/`offset` | int, `limit` in `[1, 200]`, `offset >= 0` | HTTP 422 |
| GUI `min_conf`/`max_conf` | float in `[0, 1]`, `min <= max` | HTTP 422 |
| Manifest line | object with `path` (str, resolvable) + `label` (non-empty str); `box` = 4 finite numbers, `x0<x1`, `y0<y1`; `split` ∈ {train,val} | fatal, exit 3, with line number |

### 10.3 Invariants and their owning layer

| # | Invariant | Owner | Why there |
|---|---|---|---|
| I1 | The source tree is never written, moved, renamed or deleted — including by `verify --fix` | `config.py` (nesting guard) + `images.open_source` (`"rb"` only) + `materialize.py` (refuses paths outside `output_root`) | Writes exist in exactly one module, so the guard has one place to live; E1 proves it |
| I2 | **No absolute box-area floor exists**; `dominance_ratio` is the only size gate | `decide.py`, with `config.py` rejecting unknown keys | Keeping the rule in one pure function makes its absence auditable at a glance; rejecting unknown config keys stops a floor being reintroduced by configuration |
| I3 | Every image gets exactly one label from the closed set | `decide.py` (`species_or_unknown` is total) | A single pure decision function, so no caller can invent a label |
| I4 | An image's `dest_path` parent directory always equals its catalog `label` | `materialize.py` + `catalog.py` in one write order: row → `materializing` → file → `done` | Crash-consistency needs the intent recorded *before* the filesystem changes; `verify --fix` reconciles the one interleaving that can still diverge |
| I5 | One destination file per (content, label) | `materialize.py` content-addressed collision check | Only that module knows both the hash and the destination |
| I6 | A human override outranks the model on later runs, and is never destroyed by a flag | `catalog.py` (`overrides` rows are append-only; `label_source`) read by `classify` | The durable record is the DB, so the DB owns precedence |
| I7 | No required value is ever silently defaulted (`--device cuda`, missing weights, missing API key, missing artifact) | `config.py` validation + `errors.py` (no bare `except`, no `.get(k, fallback)` for required keys) | Fail-loud must be structural, not a convention |
| I8 | An artifact's behaviour never changes after its `model_id` has been recorded | `training/evaluate.py` (calibration writes a new file) + `artifact.py` (no in-place save) | Traceability is a property of the artifact writer, not of its readers |

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

| File | Contents | Measured availability |
|---|---|---|
| `tests/e2e/data/coco_dominance.json` | `high`: 20 images with GT area ratio **> 3.0**; `low`: 20 images with GT area ratio **< 1.3**; each entry `{image_id, file_name, gt_ratio, top_class, second_class}` | 22 and 29 val-bucket images exist, so 20 + 20 is satisfiable with headroom |
| `tests/e2e/data/coco_species.json` | 20 images with **exactly one** non-crowd animal instance, `area_frac >= 0.20`, GT class in E7's seven | 24 exist |

Sizes are 20 rather than the reviewer-suggested 40 for one measured reason: after the `iscrowd`
filter, only **104** COCO images have a GT ratio > 3.0 and only **22 of those fall in the val
bucket**, so a 40-image high-ratio list cannot be built without breaking the leakage rule. 20 + 20
keeps E4 honest *and* fast (~40 detections ≈ 80 s).

### 11.2 The tests

| # | Test | Drives | Asserts |
|---|---|---|---|
| E1 | Source tree immutability | `classify`, then a GUI re-tag, `export-trainset`, `verify --fix` | Pre/post snapshot of `(path, size, mtime_ns, sha256)` over the whole fixture card is byte-identical |
| E2 | Happy path, copy mode | `classify` on a mixed fixture card | Exact output tree; one dir per label; `images`/`boxes`/`sources`/`runs` rows (incl. `runs.source_root`); exit 0 |
| E3 | `--link` and `--hardlink` | `classify` twice into separate roots | `Path.is_symlink()` + resolved target; `st_nlink == 2` and equal `st_ino`; both destinations readable |
| E4 | Dominance on **real** COCO images with **real** MegaDetector weights | `classify --detector megadetector` over `coco_dominance.json` | Every label in the closed set; **≥ 16/20** of `high` are not `multiple`; **≥ 16/20** of `low` are `multiple`; both counts printed per-image so a failure is diagnosable. No species identity assertion here (that is E7), because MegaDetector's box set legitimately differs from the annotator's |
| E5 | Dominance boundary cases | `classify --detector scripted` | ratio exactly 1.6 → dominant; 1.59 → `multiple`; equal areas → `multiple`; zero-area second box → dominant, no exception |
| E6 | **No area floor** | `classify --detector scripted` with a single animal box covering 0.05% of the frame, and a 0.1%-vs-0.5% pair | The lone tiny animal is filed as its species (never dropped, never `landscape`); the tiny pair still resolves by ratio; `classify --help` and `verify --json` expose no `min_box_area`-like option |
| E7 | **Real training + species identity** | `build_coco_manifest.py --classes zebra,elephant,giraffe,bear,cow,sheep,bird` (filtered to `split == "train"`: 1,625 crops) → `train --arch efficientnet_b0 --backbone-weights models/backbones/efficientnet_b0_ra-3dd342df.pth --input-size 128 --epochs-head 2 --epochs-finetune 2 --batch-size 32` → `eval --split val` (349 crops) → `classify --species-model <artifact>` over `coco_species.json` | Artifact has `format_version`, 7 labels with `rank`, and `train` metadata; **`val_top1 >= 0.55`** (chance 0.143, majority-class baseline 0.229); artifact `train.val_top1` equals `metrics.json` top-1 within `1e-6`; **≥ 14/20** of the frozen single-animal val images receive their GT species label through the real CLI; wall clock **< 20 min** on 8 CPU cores. Per-class recall is reported but not asserted for `bear` (11 val instances) |
| E8 | Bird path | E7's artifact (its label space contains `bird`) + `build_cub_manifest.py` → `train` a 5-species CUB head → `classify` a CUB test-split image; then a second run with `--force-bird-head` | Label is one of the 5 CUB species; `bird_provider='own_bird_head'`; `provider_status='refined'`; `images.model_id == "<species_id>+<bird_id>"`; box `species_rank='species'`. The `--force-bird-head` run shows the bird head invoked without relying on the coarse head's ranking |
| E9 | Training resume | `train` interrupted after epoch 1, then `--resume`; and `--resume` with a different manifest | Resumes at the right epoch and finishes; the mismatched resume exits 3 with both ids |
| E10 | Synthetic smoke | `train --dataset synthetic --arch tinycnn` (4 classes, 500 train / 100 val) → `eval` → `classify` | Full train→eval→export→infer path; **`val_top1 >= 0.90`**; wall clock **< 90 s** |
| E11 | `landscape` / `junk` | `classify` on a sharp animal-free photo and a heavily blurred one | `landscape` vs `junk`; `blur_score` **and `blur_ref_edge`** stored for both; a small (≤ 512 px) sharp animal-free photo is `landscape`, proving no upscaling bias |
| E12 | `unknown` (low confidence) | `classify --min-confidence 0.999` | Label `unknown`; top-5 candidates still recorded; file still materialized |
| E13 | Idempotency, resume, `--limit` | `classify` twice; kill mid-run and rerun; two `classify --limit 2` runs over a 5-image fixture | Second run writes nothing new, `already_present` counted, no duplicate rows; killed run resumes and completes; after the two `--limit 2` runs there are **4** `status='done'` rows (budget is spent on new work only) |
| E14 | Duplicate content | Same photo twice under different names | Classified once, one destination file, two `sources` rows |
| E15 | Collision, different content | Two different images sharing a filename in one label | Second becomes `<stem>-<sha8><suffix>`; both intact |
| E16 | Formats and skips | Card with JPEG/PNG/TIFF/HEIC, an `.mp4`, a `.cr2`, a 0-byte file, a truncated JPEG; plus a `--formats jpeg` run | Four formats processed by default; `.mp4` → `skipped(video)`; `.cr2` → `raw_not_enabled`; 0-byte and truncated → `skipped`; exit 4. With `--formats jpeg` the PNG/TIFF/HEIC files are `skipped(format_disabled)` and the JPEGs still process. `--formats raw` exits 3 listing the valid set |
| E17 | `--dry-run` | `classify --dry-run`, then `GET /api/images/{sha}/full` | Destination tree has no image files; rows are `planned`; exit 0; the GUI returns **409** for a planned row's `/full` |
| E18 | GUI browse | real uvicorn subprocess + `httpx` | `/` serves HTML; `/api/labels` counts match the tree and exclude `.thumbs`/`.catalog.db`/`.ebird-cache.json`; `/api/images?label=…` filters; thumb and full bytes decode as images; box coordinates lie inside the stored dimensions; captions carry `species_rank` |
| E19 | GUI re-tag moves the file (and 409s under contention) | `POST /api/images/{sha}/label`; then the same POST while a writer holds the catalog lock | File gone from the old label dir, present in the new one, content unchanged; `overrides` row written; `label_source='human'`. Under contention: HTTP **409**, and **nothing on disk or in the DB changed** |
| E20 | Override lifecycle | E19 → `classify --reclassify` → `--reclassify --ignore-overrides` → `classify --reclassify` again | Human label kept; then model label restored *and* the `overrides` row still present; then the human label and destination **return** — proving `--ignore-overrides` suppresses without destroying |
| E21 | GUI validation | bad label, bad sha256, out-of-range `limit`, unknown hash | 422/422/422/404; nothing on disk changed |
| E22 | Fail-loud config | missing weights; unknown TOML key; `dominance_ratio=0.5`; `hosted_bird_api`; `ebird_enrich` without a key; output nested inside source; `--formats raw`; `--device cuda` on this CUDA-less host; `eval --calibrate` without `--out` | Exit 3 each, with the offending name in stderr, and **no** destination directory created |
| E23 | `ebird_enrich` degrades | `classify --bird-provider ebird_enrich` with a dummy key (host answers `000` here) on a GPS-tagged fixture, and on a fixture with **no** GPS | Both runs succeed with labels equal to the `own_bird_head` run; exactly one warning in the GPS case with `provider_status='unreachable'`; the no-GPS case makes **no** network call and records `provider_status='no_gps'` |
| E24 | `verify` and `verify --fix` | `verify --json` on a good tree; after deleting a materialized file; then after simulating an interrupted re-tag (same content under two label dirs + an `overrides` row) | Exit 0 with an all-present report; then non-zero naming the missing `dest_path`; then non-zero for the duplicate, `--fix` exits 0, exactly **one** copy remains (the one the override justifies), and the source tree is untouched |
| E25 | `export-trainset` round trip | E19 → `export-trainset` → `train --manifest` | Manifest lines validate, include the human label and the dominant box, and `train` consumes them |
| E26 | Degenerate dominant box | `classify --detector scripted` with a 1 px animal box as the only detection | Label `unknown`; `boxes.species_status='degenerate'`; the file is materialized under `unknown/`; exit 0 (no crash, no invented species) |

What is *hard* to e2e-test, and the deliberate choice made: browser-driven GUI testing needs a
Playwright browser download from a host not probed in RECON, so the GUI is tested at the HTTP +
filesystem boundary (E17–E21) instead. That is where the risky behaviour lives — the re-tag **move**,
the 409 paths and the validation — so the coverage loss is cosmetic rendering only. Full-dataset
training accuracy is likewise not asserted: E7 asserts a real CPU-sized finetune clears a stated
threshold, and full runs are the user's `animal-classifier train` on their own machine.

If E4's aggregate thresholds ever fail on real MegaDetector output, the sanctioned remedy is to
**re-freeze the list at a stricter GT ratio** (19 val-bucket images are available at ratio > 4.0) and
record that change here and in `PROGRESS.md` — never to quietly lower the ratio in test code.

## 12. Sequenced build order

1. **Dependency contract first**: write `pyproject.toml` exactly as §2.1, add `.python-version`
   (`3.12`), `uv lock`, commit `uv.lock`, then re-run `scripts/probe_md_checkpoint.py`,
   `probe_md_inference.py`, `probe_backbones.py` against the synced venv. Nothing else starts until
   those three pass on the locked environment.
2. `errors.py`, `config.py`, `catalog.py`, `taxonomy/` + static label tables.
3. `scan.py`, `images.py`, `materialize.py` → `classify` end-to-end with `detect/scripted.py` and a
   pass-through classifier. Lands E1, E2, E3, E5, E6, E11, E13–E17, E22, E26.
4. `detect/megadetector.py` (weights verification, alias shim, letterbox/NMS) +
   `scripts/build_coco_dominance.py`. Lands E4.
5. `training/` + `scripts/build_*_manifest.py` + `train`/`eval`. Lands E7, E9, E10.
6. `classify/own_model.py`, `classify/birds/*`, wire `decide` thresholds. Lands E8, E12, E23.
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
4. `ebird_enrich`'s 0.25 down-rank multiplier is a deliberate, documented constant, not a fitted one:
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
| 16 | MED | `verify --fix` added as the **only** reconciler, scoped to deleting a stale copy under an old label directory when an `overrides` row justifies the current label; never touches the source; report-only without `--fix` (non-zero exit). The "classify reconciles" claim is withdrawn. E24 extended | §5.8, §8, E24 |
| 17 | MED | `--limit N` caps images **newly submitted to inference**; `done` hashes are skipped without consuming budget. E13 asserts two `--limit 2` runs over a 5-image card yield 4 done rows | §5.9, §9, E13 |
| 18 | NIT | `rank` added to every artifact label entry, `Prediction.rank_level`, `boxes.species_rank` and `images.species_rank`; shown in the GUI detail caption | §5.5, §5.9, §6, §7.1 |
| 19 | NIT | Stated once: label directories match `^[a-z0-9][a-z0-9_-]{0,63}$`; every other entry under `output_root` is tool-internal and ignored by `/api/labels` and by `verify`'s orphan scan. E18 asserts it | §5.8, §6, §8, E18 |
| 20 | NIT | `sources` upsert specified as `ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, mtime_ns=excluded.mtime_ns`, with the note that the previous `images` row and destination file are retained because content is the identity | §5.9 |
| 21 | NIT | Blur metric **downscales only** when the long edge exceeds 512 px; `blur_ref_edge` stored beside `blur_score` and asserted in E11 | §5.2, §5.9, E11 |
| 22 | NIT | Link mode compares `os.readlink(dest)` with the intended absolute source before any hashing; a differing or dangling link is replaced atomically and counted as `relinked`; only regular files are hashed | §5.8, §10.1 |
| 23 | NIT | The unreachable "missing HEIF library" fatal row is deleted; `pillow-heif` is core | §10.1 |
| 24 | NIT | "~2,700" corrected to **2,666** (2,700 − 34 `iscrowd`) over 1,016 images, with per-class counts, and the explicit note that `bear` (71 total, 11 val) limits per-class recall claims — so no threshold is asserted on it | §1, §7.2, E7 |
| 25 | NIT | People/vehicle-only images filing as `landscape` (or `junk` if blurry) is documented in `README.md` **and** `classify --help`, noting their boxes are still catalogued and drawn | §5.4, §8, §12.8 |




