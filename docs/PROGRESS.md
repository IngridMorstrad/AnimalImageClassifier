# Progress log

Append-only. Every entry is UTC-timestamped so it can be read back in order.
Format: `## YYYY-MM-DD HH:MM UTC — <who> — <what>`

## 2026-09-15 14:33 UTC — orchestrator — repo cloned
`IngridMorstrad/AnimalImageClassifier` cloned, empty (no commits), branch `main`.

## 2026-09-15 14:45 UTC — orchestrator — first plan drafted
`docs/PLAN.md` written: pipeline, label taxonomy, dominance rule, GUI, provider abstraction.
Established that Merlin has no public photo-ID API (on-device only); eBird API 2.0 covers
taxonomy/observations, not image ID; iNaturalist vision endpoint is access-restricted.

## 2026-09-15 14:52 UTC — orchestrator — network reachability measured
Sandbox network is partially restricted (COMMON_DEPENDENCIES). Measured with curl:

| Reachable | Blocked |
|---|---|
| pypi.org, github.com, raw.githubusercontent.com, GitHub release assets | huggingface.co, download.pytorch.org, storage.googleapis.com, lila.science, api.ebird.org, api.inaturalist.org |

Decisive finding: **MegaDetector v5a weights are downloadable** —
`https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt` returned
HTTP 206 for a ranged request. Real detection can therefore be exercised in-sandbox.
Blocked: SpeciesNet weights, BioCLIP weights, torchvision/timm ImageNet backbones, and all
standard wildlife training sets.

## 2026-09-15 14:54 UTC — orchestrator — plan revised per review
- Removed `min_box_area`; the dominance ratio alone handles a distant background animal, and
  an absolute floor would risk discarding a small-but-only subject.
- Added an own/finetuned species model (transfer learning) plus a training subsystem.
- Testing narrowed to end-to-end only, no unit tests.

## 2026-09-15 14:56 UTC — orchestrator — implementation workflow launched
Handing off to the workflow. Subsequent entries are written by workflow steps.
## 2026-09-15 15:08 UTC — setup step — project skeleton created
Branch `feat/safari-classifier` created off `main` (nothing is ever committed to `main`).

Created:
- `pyproject.toml` — project `animal-classifier`, hatchling backend, `src/` layout,
  `[project.scripts] animal-classifier = "animal_classifier.cli:app"`, pytest `testpaths = tests/e2e`.
- `src/animal_classifier/__init__.py` — package marker, `__version__ = "0.1.0"`.
- `src/animal_classifier/cli.py` — typer app with placeholder `classify`, `gui`, `train`,
  `eval`, `export-trainset`, `verify` commands. Each prints "not implemented yet" to stderr
  and exits 1, so a later step replacing a body cannot be mistaken for working code.
- `tests/e2e/conftest.py` — placeholder for the e2e-only suite.
- `models/.gitkeep` — artifact dir; weights are gitignored, never committed.
- `README.md` — setup (`uv venv`) and command reference.
- `.gitignore` — `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `models/*.pt`,
  `models/*.pth`, `models/*.onnx`, `*.sqlite`, `*.db`, generated test output dirs, build artifacts.

Environment: `uv venv` → CPython 3.12.13 at `.venv`; `uv pip install -e .` installed typer 0.27.2.
`uv` is the only Python tool used.

Verified: `uv run animal-classifier --help` lists all six commands; `uv run animal-classifier verify`
exits 1 with the placeholder message.

Preserved existing `docs/PLAN.md` and `docs/PROGRESS.md` untouched (this entry appended).
No classifier logic implemented in this step.

## 2026-09-15 15:24 UTC — recon step — real weights, real backbones and real datasets acquired

Full detail with every measured HTTP status code in `docs/RECON.md`. Headline: the sandbox is much
more capable than the 14:52 sweep suggested. **No part of the vision subsystem needs to be faked.**

Downloaded and verified (all gitignored):
- `models/md_v5a.0.0.pt` — MegaDetector v5a, HTTP **200**, 280,766,885 bytes in 16.1 s,
  sha256 `94e88fe97c8050f2e3d0cc4cb4f64729d639d74312dcbe2f74f8eecd3b01b276`. Size matches the
  GitHub release API exactly. Mirror (`agentmorris/MegaDetector`) also **206**.
- `models/backbones/efficientnet_b0_ra-3dd342df.pth` (**200**, 21,376,743 B) and
  `convnext_nano_d1h-7eb4bdea.pth` (**200**, 62,396,822 B) — ImageNet-pretrained timm weights,
  fetched from GitHub release assets. Computed sha256 of each matches the digest timm embeds in the
  filename.
- `data/raw/CUB_200_2011.tgz` (**200**, 1,150,585,339 B, 12,005 entries, 200 bird species).
- `data/raw/val2017.zip` (**200**, 815,585,330 B) + `annotations_trainval2017.zip` (**200**,
  252,907,541 B).

Confirmed by execution, not assumption:
- MegaDetector loads: `top-level keys = ['best_fitness','date','ema','epoch','model','optimizer','updates','wandb_id']`,
  `model class = models.yolo.DetectionModel`, 140,054,656 params, `names = ['animal','person','vehicle']`.
  Three real errors had to be fixed first — `ModuleNotFoundError: models` (alias `sys.modules['models']`
  → `yolov5.models`), `ImportError: libGL.so.1` (use `opencv-python-headless`), and
  `ModuleNotFoundError: pkg_resources` (pin `setuptools<81`).
- MegaDetector runs: forward pass OK in 0.55 s on CPU at 640 px, output `(1, 25500, 8)`.
- Both backbones load into timm with `missing=0 unexpected=0` and emit real `(1, 1000)` logits.
- COCO val2017 parsed: 1016 images contain ≥1 animal, 471 multi-animal, 70 multi-species, and by
  real box area **123 have a clear dominance winner (ratio>3)** vs **194 ambiguous (ratio<1.5)**.
  The dominance rule and the `multiple` label now have real ground truth to be tested against.

Blocked, confirmed by probe (`000`): download.pytorch.org, huggingface.co, storage.googleapis.com,
lila.science, Kaggle, and **every** live taxonomy/species API — eBird, Macaulay, iNaturalist, GBIF,
ITIS, ChecklistBank, Wikidata, Wikimedia Commons, Wikipedia. Therefore species naming ships as a
static in-repo table (offline-first); the remote bird provider stays optional behind the plan's
provider abstraction. SpeciesNet/PytorchWildlife/megadetector code is installable from pypi but
their non-MegaDetector weights are Kaggle/HF-hosted, so unusable here.

**Decision recorded (supersedes the synthetic-only contingency):** the recon brief's fallback — prove
training on a generated synthetic dataset with a from-scratch CNN and defer real training — is NOT
needed and will not be used. A pretrained backbone and a real labelled dataset are both present and
verified, so the training subsystem will be proven by genuinely fine-tuning a real backbone on a
CPU-sized CUB-200 subset and asserting achieved accuracy. `animal-classifier train` remains a
first-class documented command for full-dataset runs on the user's own machine.

Added `scripts/probe_md_checkpoint.py`, `scripts/probe_md_inference.py`, `scripts/probe_backbones.py`,
`scripts/probe_urls.sh` and the URL lists under `scripts/recon/` so every probe is reproducible.
`.gitignore` extended with `models/backbones/` and `data/`. No classifier logic implemented in this step.


## 2026-09-15 15:35 UTC — design step — docs/DESIGN.md authored (iteration 1)

No `docs/design-review.json` present, so this is the first design pass. Grounded in `docs/PLAN.md`
and `docs/RECON.md`; every asset named in the design is one RECON verified by download + load + run.

Design covers: module layout, config layering/validation, the 8-stage pipeline, the `.acmodel`
artifact format, the training subsystem, the bird provider trio, the GUI's routes, a per-operation
error table, a per-input validation table, 7 invariants with owning layers, and 25 numbered e2e
tests mapped to the build spec.

Load-bearing decisions:
- **Dependency conflict found and solved, verified by running `uv lock`.** `yolov5` 7.0.14 requires
  `opencv-python` (via itself, `sahi`, `ultralytics`), which cannot import in this image
  (`libGL.so.1`). Fix: `[tool.uv] override-dependencies = ["opencv-python; python_version < '3.0'"]`
  removes it from resolution while we declare `opencv-python-headless` directly. Probe resolved 131
  packages with **`opencv-python-headless` 5.0.0.93 and zero `opencv-python`**, `typer` 0.25.1,
  `setuptools` 80.10.2 (`<81`, needed for yolov5's `pkg_resources`).
- **`uv run` syncs the venv to `pyproject.toml` + `uv.lock`** — observed swapping typer 0.25.1 →
  0.27.2 mid-session. So every runtime dep must be declared in `pyproject.toml`; ad-hoc
  `uv pip install` packages get evicted. Recorded as a hard constraint.
- **`typer` is capped at 0.25.x** by `yolov5` → `roboflow` (`typer<0.26`). CLI must not use 0.26+ APIs.
- **Two real training sets, no synthetic substitute.** Per RECON's explicit decision: species head =
  COCO val2017 animal crops from ground-truth boxes (10 classes, ~2,700 instances); bird head =
  CUB-200-2011 (200 species, boxes, official split). Both finetune the verified local
  `efficientnet_b0` ImageNet weights. `tinycnn` + synthetic data is kept ONLY as a seconds-long
  smoke test (E10), explicitly not as the proof of the subsystem — E7 is, on real data.
- **Dominance rule is the only size gate.** No `min_box_area`, no floor under any name, stated in
  §3, §5.7 and invariant I2; enforced structurally by `config.py` rejecting unknown TOML keys, and
  behaviourally by E6 (a lone animal at 0.05% of frame must still be filed as its species). The
  comparison is written as multiplication (`a0 >= ratio * a1`), so a zero-area second box makes the
  first dominant instead of dividing by zero. Degenerate sub-2px crops still count as animals for
  dominance — excluding them would be an area floor by the back door.
- **Source tree immutability gets three independent mechanisms**: a config-time nesting guard, a
  single `open_source()` read-only accessor with writes confined to `materialize.py`, and e2e test
  E1 snapshotting `(path, size, mtime_ns, sha256)` across the whole fixture card.
- **Atomicity**: temp file created *in the destination directory* (same filesystem guaranteed) then
  `os.replace`, for all three of copy/symlink/hardlink; `finally` cleanup; stale temp sweep on the
  next run. `--hardlink` across filesystems (`EXDEV`) is fatal and actionable, never a silent copy.
- **Crash consistency**: catalog row → `materializing` → file write → `done`, so an interrupted run
  is reconciled by `verify` rather than losing a file. Re-tag materializes into the new label dir
  before unlinking the old one.
- **The one sanctioned graceful degradation** is `ebird_enrich` being unreachable (`api.ebird.org` is
  `000` here): the run keeps the own-head label, warns once, and records
  `provider_status='unreachable'`. Justified because the enrichment is optional. A *rejected* key
  (401/403) and a selected-but-unconfigured provider are both fatal at config time.

Verified while designing, not assumed: `yolov5.utils.general.non_max_suppression` / `scale_boxes`
and `yolov5.utils.augmentations.letterbox` all import and their signatures match the design's usage;
`pi_heif.register_heif_opener()` works; `rawpy`, `fastapi` and `pytest` are NOT yet installed (RAW
becomes an optional extra, the rest get declared); COCO animal category ids confirmed as 16–25 and
`instances_val2017.json` annotation keys confirmed; 8 CPU cores / 30 GB RAM available.

No implementation code written in this step.

## 2026-09-15 15:47 UTC — design review (cold, adversarial) → **CHANGES_REQUESTED**

Reviewed `docs/DESIGN.md` iteration 1 against `PLAN.md` and `RECON.md` without the authoring
context. Findings: **2 HIGH, 15 MEDIUM, 8 NIT** → `docs/design-review.json` verdict
`CHANGES_REQUESTED`; full narrative in `docs/design-review.md`.

None of the hard blocking conditions fire. Checked and clear: no `min_box_area` or any absolute
area floor under any name (`dominance_ratio` 1.6 is the only size gate, and §5.3 even keeps
degenerate boxes in the count to avoid a back-door floor); e2e-only testing (`testpaths =
["tests/e2e"]`, E1–E25, no `tests/unit`); no dependence on a RECON-blocked host; no writes to the
source tree; no specified silent defaulting; and every required element present (MegaDetector v5a,
own finetuned species model + real training subsystem + `train`, three bird providers, the closed
label set, atomic writes, copy/`--link`/`--hardlink`, sha256 idempotent resume, SQLite catalog,
FastAPI+vanilla-JS GUI on 127.0.0.1:8765 with re-tag that moves files, the six typer commands, the
format policy).

The two HIGHs:
1. **Dependency contract is wrong and unreproducible.** `pyproject.toml` declares only
   `typer>=0.12` and `uv.lock` has 9 packages, so §2.1's "131 packages, typer 0.25.1,
   opencv-python-headless 5.0.0.93" is not verifiable here. A real resolution
   (`uv pip compile --override … --python-version 3.12`) gives **115** packages, **typer 0.27.2**
   (resolver picks `roboflow` 1.3.8, which has no `typer<0.26` cap — the cap is only in 1.4.2), and
   `opencv-python-headless` **4.10.0.84**. Because `uv run` syncs the venv to the lock, the first
   `uv sync` would replace the torch 2.14.0 / torchvision 0.29.0 environment that every RECON
   verification was measured on. Fix: exact pins + committed lock + `uv sync --frozen`.
2. **Degenerate dominant box has no defined label.** `species_or_unknown()` is called but never
   defined, and real data hits it: 7 non-crowd COCO animal boxes are <2 px in one dimension.

Verified rather than trusted (all re-measured this step): `md_v5a.0.0.pt` sha256
`94e88fe9…` = the RECON constant; `efficientnet_b0_ra-3dd342df.pth` hash matches its filename;
`letterbox`/`non_max_suppression`/`scale_boxes` import with the signatures §5.4 uses; the `uv`
override really eliminates `opencv-python`; fastapi 0.141.1 / torch 2.14.0 / torchvision 0.29.0 /
timm 1.0.29 as designed; COCO ids 16–25 correct; **2,666** usable animal instances (not ~2,700 —
34 are `iscrowd=1`) over 1,016 images; at the design's own 1.6 gate the 471 multi-animal images
split **235 dominant / 236 multiple**; CUB ships `bounding_boxes.txt` + `train_test_split.txt`.

Newly discovered blocker for E16: **`pi-heif` cannot encode HEIF** (`im.save(format='HEIF')` →
`KeyError: 'HEIF'`), and `data/raw` holds only JPEGs, so the HEIC fixture is unbuildable as
specified. Full `pillow-heif` 1.1.1 (libheif 1.20.2) encodes and re-reads correctly — switch back
to it, as `PLAN.md` originally said.

Other MEDIUMs, in one line each: undefined `BirdResult`/`GpsPoint`/`top5` shapes; bird refinement
unreachable in E8 because the trigger is the coarse head's top-1; E4 has no pass criterion and
compares MegaDetector boxes to GT-derived expectations; no hold-out guarantee against COCO
train/test leakage; E7/E10 thresholds unstated; `formats` key has no semantics; `source_root` has no
provenance for `gui`/`verify`; a GUI re-tag can abort a running `classify` via the fatal
`database is locked` rule; `--device cuda` with no GPU unspecified; `ebird_enrich` with no GPS and
"down-rank" undefined; `eval --calibrate` mutates artifacts in place, breaking `model_id`
traceability; `--ignore-overrides` state is ambiguous; §5.8 "reconciles" contradicts §8's
report-only `verify`; `--limit` vs resume ambiguous.

No implementation code written in this step. Next: design iteration 2 addressing the 17 HIGH/MEDIUM
findings, then re-review.
