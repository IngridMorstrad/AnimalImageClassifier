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

