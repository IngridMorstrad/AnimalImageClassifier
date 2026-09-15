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

## 2026-09-15 16:14 UTC — design step — docs/DESIGN.md iteration 2 (all 25 review findings resolved)

Rewrote `docs/DESIGN.md` (743 → 1,101 lines) against `docs/design-review.json`. Every finding is
resolved and §14 maps all 25 (2 HIGH / 15 MEDIUM / 8 NIT) to the section that fixes it. I re-measured
the load-bearing claims rather than trusting either the old design or the review — three of the
review's own suggested fixes were wrong or unbuildable and are adapted with the measurement that
proves it:

1. **Finding 1's pin set is unsatisfiable.** Measured: `roboflow==1.3.8` pins
   `opencv-python-headless==4.10.0.84` *exactly*, and `roboflow` 1.4.2 pins `typer<0.26` — so
   "`roboflow==1.3.8` + `opencv-python-headless==5.0.0.93` + `typer==0.27.2`" cannot resolve
   (`uv` says unsatisfiable). Fix taken instead: pin the RECON-verified versions and override
   `roboflow` **and** `sahi` out of the resolution alongside `opencv-python`. Verified by probe that
   `import yolov5`, `letterbox`, `non_max_suppression`, `scale_boxes`, the `models`/`utils` alias
   shim, the MegaDetector load and a 640×640 forward (`(1, 25500, 8)`,
   `names=['animal','person','vehicle']`) all work with `roboflow`+`sahi` blocked. The pinned set
   resolves to **109 packages** with no `opencv-python`/`roboflow`/`sahi` and the exact verified
   `torch 2.14.0 / torchvision 0.29.0 / timm 1.0.29 / numpy 2.5.3 / opencv-python-headless 5.0.0.93`.
   `uv.lock` committed + `uv sync --frozen` mandatory; moving a modelling pin forces a re-run of the
   three RECON probes.
2. **Findings 5 + 6 conflict on real data.** Requiring e2e assertion images to be val-split (finding 6,
   anti-leakage) leaves only **22** COCO images with GT dominance ratio > 3.0 — a 40-image frozen list
   (finding 5) is unbuildable. E4 therefore freezes **20 + 20** val-bucket images with ≥ 16/20
   aggregate thresholds. Also refined `split_for()` to key on the file **basename**, so the split is
   machine-independent and every crop of one photo stays on one side of it.
3. **Finding 8 confirmed, version corrected.** On py3.12 + pillow 12.3.0 the resolvable
   `pillow-heif` is **1.7.0** (libheif 1.23.3), not 1.1.1; verified `save(format="HEIF")` +
   re-open round-trip, so the E16 HEIC fixture is buildable. `pillow==12.3.0` is real (latest);
   `pi-heif` stays rejected for its missing encoder.

Other numbers re-measured from `instances_val2017.json` and written into the design: **2,666** usable
animal instances (2,700 − 34 `iscrowd`) over 1,016 images; per-class counts with `bear` at 71 (11 val)
flagged as the limit on per-class claims; **7** sub-2px boxes, smallest `area_frac` 0.00001243;
471 multi-animal images → **235 dominant / 236 multiple** at the 1.6 gate; E7's 7-class subset =
**1,625 train / 349 val** crops (chance 0.143, majority baseline 0.229 → asserted `val_top1 >= 0.55`);
24 val-split single-animal images available for E7's species-identity list (20 used, ≥ 14 must hit).

Substantive design additions: `species_or_unknown()` defined and total (degenerate winner → `unknown`,
new test E26); `Candidate`/`Prediction`/`BirdResult`/`GpsPoint` dataclasses plus the bird merge rule
and `model_id = "<species>+<bird>"`; bird trigger widened to top-3 Aves plus a `--force-bird-head`
diagnostic; `formats` given real semantics (`format_disabled` skip reason, repeatable `--formats`,
fatal on `raw`); `runs.source_root NOT NULL` as the provenance for `gui`/`verify` and the GUI's
source-serving mode deleted (409 for `--dry-run` rows); read-only GUI connections + 409 on catalog
contention so a re-tag can never abort a 2.8 h run; `--device cuda` without a CUDA device fatal;
`ebird_enrich` no-GPS path and the 0.25 down-rank multiplier specified; calibration writes a **new**
artifact (`+calN`, new invariant I8); `--ignore-overrides` suppresses without deleting; `verify --fix`
added as the only reconciler; `--limit` defined as new-work budget; `species_rank` threaded through
artifact → catalog → GUI; label-dir regex with tool-internal entries ignored; `sources` upsert rule;
blur metric downscale-only with `blur_ref_edge`; link-mode `readlink` comparison before hashing.

Still zero implementation code. Test count is now **E1–E26, end-to-end only**. Next: design re-review,
then implementation starting with the §2.1 dependency contract and the three probe re-runs.

## 2026-09-15 16:26 UTC — design review iteration 2 (cold): CHANGES_REQUESTED (1 HIGH / 13 MEDIUM / 6 NIT)

Reviewed `docs/DESIGN.md` iteration 2 without the context that produced it. Wrote
`docs/design-review.md` (narrative) and `docs/design-review.json` (gate). Verdict
**CHANGES_REQUESTED** — 20 findings, none of them a hard blocker.

**No blocking condition applies.** Re-checked each one: no `min_box_area` or absolute area floor
under any name (the only four mentions are explicit negations; `dominance_ratio` 1.6 is the single
size gate, applied as a multiplication); no unit tests or any layer outside `tests/e2e`; no
dependence on a RECON-blocked host (`api.ebird.org` is opt-in and off by default, its
unreachability recorded as `provider_status='unreachable'`); no write/move/rename/delete in the
source tree; no silent default for a required value; and every required capability present
(MegaDetector v5a, own finetuned species model + training subsystem + `train`, three bird
providers, the five-label set, atomic writes, copy/`--link`/`--hardlink`, sha256 idempotency, the
SQLite catalog, the FastAPI + vanilla-JS GUI on 127.0.0.1:8765 with a file-moving re-tag, the six
typer commands, and the JPEG/PNG/TIFF/HEIC + `--raw` + skip-video format policy).

**Everything measurable in DESIGN.md was re-measured, and all of it reproduced exactly.** Four
independent probes, not inherited from RECON:

1. MegaDetector `md_v5a.0.0.pt` — 280,766,885 B and sha256 `94e88fe9…b01b276` recomputed; loads
   through the `models`/`utils` alias shim; `names=['animal','person','vehicle']`,
   `stride=[8,16,32,64]`, forward → `(1, 25500, 8)`.
2. §2.1's load-bearing claim — with `roboflow` and `sahi` blocked at import via a `sys.meta_path`
   hook, `import yolov5`, `letterbox`, `non_max_suppression`, `scale_boxes`, the checkpoint load and
   a 640×640 forward all succeed and neither module is imported.
3. `uv pip compile` on the exact §2.1 `pyproject.toml` — resolution contains
   `opencv-python-headless==5.0.0.93`, `typer==0.27.2`, `torch==2.14.0`, `timm==1.0.29`,
   `pillow==12.3.0`, `pillow-heif==1.7.0`, `rawpy==0.27.1`, and **no** `opencv-python`, `roboflow`
   or `sahi`. The override trick works as designed.
4. `pillow-heif==1.7.0` + `pillow==12.3.0` on py3.12 — `libheif 1.23.3`, `save(format="HEIF")` and
   re-open round-trip OK, so E16's HEIC fixture is genuinely buildable (iteration-1 blocker cleared).
5. COCO recount from `instances_val2017.json` with the design's own `split_for`: 2,700 − 34 crowd =
   **2,666** over **1,016** images, per-class counts identical; **7** sub-2px boxes, min `area_frac`
   `1.2426814988e-05`; **235 dominant / 236 multiple** at 1.6; **813/203** image split; ratio > 3.0 →
   104/**22 val**, ratio < 1.3 → 149/**29 val**, ratio > 4.0 → 79/**19 val**; E7 crops
   **1,625/349** with majority baseline **0.229** and `bear` at 11 val instances; 24 candidates for
   `coco_species.json`. Every number matches the design.

**The findings that must be fixed before coding.** HIGH: re-tag never says what bytes it
materializes from — all three materialize modes read the SD-card source, but the GUI is used with
the card unplugged, so the specified path is unimplementable in the normal case (fix: re-tag is a
pure `os.replace` inside `output_root`, never opening the source). MEDIUM, in short: `--reclassify`
leaves duplicate `boxes`/`candidates` rows; `--limit` budget contradicts `--reclassify`;
`temperature` undefined for an uncalibrated artifact; `slug()` handles only spaces and has no
defined failure; the EXIF-orientation frame for `width`/`height`/boxes/thumbs is unstated;
`too_large`/`symlink_loop` have no defining rule and symlinked source files no policy;
`ebird_enrich`'s static table is absent from the layout and its candidate-matching rule undefined;
`/api/labels` (filesystem) and `/api/images` (catalog) are two sources of truth; `/thumb` has no
defined input; `export-trainset` is undefined for `multiple`/`landscape`/`junk`/`unknown`; E4's
sanctioned remedy needs 20 images where only **19** exist at ratio > 4.0 (measured); hardlink inode
aliasing is outside I1's path-based proof; and NULL-confidence rows silently vanish from confidence
filters.

Next: design iteration 3 addressing the HIGH and all 13 MEDIUMs, then re-review. Still zero
implementation code.

### Addendum (16:30 UTC) — review pipeline diagrams

The previous commit's message lost its two ASCII diagrams to shell expansion (backticks in a
heredoc-free `-m` string). Recording them here instead of rewriting a pushed commit:

Relationships:

```
PLAN.md ──┐
RECON.md ─┼─> DESIGN.md ──> design-review.md ──> design-review.json (gate: verdict)
          │                        ^
probes ───┘                        │
(MegaDetector load/forward with roboflow+sahi blocked, uv pip compile of §2.1,
 pillow-heif HEIF round-trip, COCO recount via the design's own split_for)
```

Sequence:

```
reviewer -> DESIGN.md          : read in full (1101 lines)
reviewer -> sandbox            : 4 probes + COCO recount
sandbox  -> reviewer           : every number matches DESIGN.md
reviewer -> design-review.md   : 20 findings + verified/unverified assumptions
reviewer -> design-review.json : verdict CHANGES_REQUESTED (1 HIGH / 13 MEDIUM / 6 NIT)
reviewer -> PROGRESS.md        : this entry
```

Probe artifacts live outside the repo (`/projects/sandbox/.review_block_probe.py`,
`.review_coco_check.py`, `.depcheck/pyproject.toml`, `.heifprobe/`) so they do not enter the package;
each is a few lines and reproducible from the descriptions above.

---

## 2026-09-15 16:41 UTC — DESIGN.md iteration 3: all 20 iteration-2 review findings resolved

`docs/design-review.json` (iteration 2) returned `CHANGES_REQUESTED` with 1 HIGH / 13 MEDIUM / 6 NIT
and **no blocking condition tripped**. Every one of the 20 findings is now resolved in
`docs/DESIGN.md` (1101 → 1542 lines), with a new **§15** mapping each finding to where it was
addressed and a **§15.1** answering the reviewer's actionable unverified assumptions.

**Re-measured before writing (not inherited from the review):**

| Quantity | Result |
|---|---|
| Val-bucket multi-animal COCO images, GT ratio > 3.0 / > 4.0 / < 1.3 | **22 / 19 / 29** |
| `ceil(0.8 × len)` for those | **18 / 16 / 24** |
| `coco_species.json` candidates | **24** |
| Dominance GT at the 1.6 gate | **235 dominant / 236 multiple** of 471 |

**The HIGH finding (re-tag) — rewritten, and stricter than the suggestion.** Re-tag was defined by
reference to the three materialize modes, *all* of which read the SD card, so it was unimplementable
in the normal case (card unplugged). It is now its own operation, `materialize.retag()`: resolve the
new name, then **one `os.replace`** inside the output tree — which moves a regular file, keeps a
hardlink's inode, and moves a symlink *as* a symlink without dereferencing it. The source is never
opened. I deliberately **inverted the reviewer's ordering** (they put the override row after the
rename): intent is written **before** the rename because I4 demands it, which turns the crash window
from "two label dirs" into a *pending move* that `verify --fix` completes forward from the `overrides`
row. No duplicate, no lost image, no intent inferred from the filesystem.

**Two other adaptations, both stated in §15 with the reason:**
- **Finding 12** (E4's remedy was arithmetically impossible: 19 images available for a hard-coded
  20-entry list) → the frozen lists now contain **all** qualifying val-bucket images and E4 asserts
  `>= ceil(0.8 * len(list))` read from the JSON. Re-freezing at ratio > 4.0 now needs **zero** test-code
  arithmetic changes.
- **Finding 7** (`too_large` had no rule) → defined as a scan-time `max_file_bytes` (512 MiB) cap and
  decode's pixel rejection renamed `too_large_pixels`, so the two can never be confused. §3 and I2
  state explicitly that this byte cap is a whole-file gate that **never sees a box** — it is not an
  area floor by another name.

**Hard constraints re-verified against the final text:** no `min_box_area` / area floor anywhere
(`dominance_ratio` is still the only size gate, I2); e2e tests only (`tests/e2e`, no unit tests — §11.2
now also names the only two sanctioned boundary seams: the shipped `--detector scripted` and E23's
`httpx.MockTransport` for the blocked eBird host); source tree read-only and now provably so under
`--hardlink` (I1 gained the no-write rule; E1 gained a hardlink leg and a card-renamed-away leg); no
silent default for a required value (`temperature` is always written and fatal-if-missing, `slug()`
raises, reserved labels rejected).

**Also tightened:** re-inference `DELETE`-then-insert order plus `PRIMARY KEY(box_id, rank)` and a
unique `boxes(sha256, idx)` index (doubling is now impossible, not merely avoided); `--limit` redefined
as an *inference* budget that `--reclassify` consumes in `last_updated ASC` order; one coordinate frame
(EXIF-transposed) promoted to invariant I9 with an orientation-6 fixture; catalog-authoritative
`/api/labels` with `files_on_disk` as a visible cross-check; `/thumb` and `/full` 409 with a reason
(including dangling symlinks); `include_unscored` + `unscored_excluded` so NULL-confidence rows can
never be silently filtered away; `export-trainset` label policy table; `verify` exit codes mapped to
3/4; GUI `busy_timeout = 250 ms`; `run_id` format and "newest run" ordering; 224 px restated as the
shipped default with E7's 128 px called a test-budget trade-off.

Component relationships for this step:

```
docs/design-review.json ──┐
docs/design-review.md  ───┼─→ (this step) ──→ docs/DESIGN.md §1–§13 revised, §15 + §15.1 added
docs/PLAN.md           ───┤                        │
docs/RECON.md          ───┘                        └─→ docs/PROGRESS.md (this entry)
                            ^
data/raw/annotations/instances_val2017.json
(re-measured frozen-list availability with the design's own split_for)
```

Sequence:

```
step     -> design-review.json/.md : read all 20 findings + assumption lists
step     -> COCO annotations       : recount 22/19/29 + 24, confirm 235/236
step     -> DESIGN.md              : resolve findings 1-20 in place
step     -> DESIGN.md §15/§15.1    : per-finding resolution map + assumption answers
step     -> PROGRESS.md            : this entry
```

Next: the design is ready for another cold review pass; nothing in it now depends on a
RECON-blocked host, and every threshold that could move has a documented, executable remedy.

## 2026-09-15 21:11 UTC — design review iteration 3 (cold): CHANGES_REQUESTED (0 HIGH / 11 MEDIUM / 8 NIT)

Reviewed `docs/DESIGN.md` iteration 3 (1,542 lines) fresh, against `PLAN.md` and `RECON.md`, with no
context from the step that wrote it. Wrote `docs/design-review.md` (narrative) then
`docs/design-review.json` (gate). **Verdict `CHANGES_REQUESTED`: 11 MEDIUM, 8 NIT, no HIGH, and no
blocking condition tripped.**

**All six blocking conditions are clear, and I checked each by grep rather than by reading the
document's own claims:** every one of the 7 `min_box_area`/area-floor mentions is an explicit negation
(`dominance_ratio` remains the only size gate; `max_file_bytes` is walled off in `scan.py` where it
structurally cannot see a box); `tests/e2e` only, with the two doubles both sitting on external
boundaries; the three blocked-host mentions are all "never do this" or "on the user's machine"; the
source tree is read-only with I1 now covering the `--hardlink` aliased-inode case; I7 is structural;
and every mandated component is present (MegaDetector v5a, own finetuned model + `train`, the
three-implementation bird provider, the closed label set, atomic writes, copy/`--link`/`--hardlink`,
sha256 idempotency, SQLite catalog, GUI on 127.0.0.1:8765 with a re-tag that MOVES, the six-command
typer CLI, the format policy).

**I re-measured every quantitative claim instead of trusting the fact table** — probe committed as
`scripts/recon/verify_design_facts.py`, which reuses the design's own published `split_for()`. **All 15
numbers matched exactly**: 2,666 non-crowd instances / 34 crowd, 1,016 images, per-class counts
(`bird 427 … bear 71`), 813/203 split, 471 multi-animal, 235/236 at the 1.6 gate, val-bucket 22 at
ratio > 3.0 and 19 at > 4.0 and 29 at < 1.3 (so E4's `ceil(0.8×n)` = 18/16/24), 24 `coco_species`
candidates, 1,625/349 seven-class crops, 0.229 majority baseline, 11 `bear` val instances. The
iteration-3 fact table is trustworthy.

**Three claims turned out to be wrong, and two of them were found only by reading the real data:**
- **CUB names carry no apostrophes.** §5.8 justifies `slug()` with "`Brewer's Blackbird`,
  `Le Conte's Sparrow`". The archive's actual `classes.txt` is `001.Black_footed_Albatross`,
  `022.Chuck_will_Widow` — already underscored, numerically prefixed. Since §7.2 takes the label from
  the directory name, the user would get `~/animal_pics/022_chuck_will_widow/`, against PLAN.md's
  `~/animal_pics/lion` (finding 2).
- **"List lengths are data-driven, not hard-coded"** is true of `coco_dominance.json` only;
  `coco_species.json` is still frozen at 20 of 24 with E7 asserting `>= 14/20` — iteration 2's defect
  surviving in the second list (finding 8).
- **"train and inference see the same framing"** holds only while `crop_margin` keeps its default,
  which the config explicitly permits changing, and the artifact does not record it (finding 7).

**One defect I reproduced rather than argued.** The published schema's `skipped(path TEXT PRIMARY KEY)`
has no upsert (unlike `sources`, which has one spelled out). Created the exact schema on sqlite 3.40.0
and re-inserted a skipped path: `IntegrityError -> UNIQUE constraint failed: skipped.path`. Every second
run over a card containing a video walks into it, which is E13 and E16 (finding 1). Also verified
positively: the schema creates cleanly with `rank`/`idx` as column names, and `slug()` behaves exactly
as published (`brewer_s_blackbird`, `nandu`, raises on `'熊'` and whitespace, truncates at 64).

**The remaining MEDIUMs cluster in what iteration 3 added:** `q` is named once and never defined
(finding 3); `date_from`/`date_to` reintroduce the NULL-silent-drop bug that `include_unscored` was
invented to fix one paragraph earlier (finding 4); exit 4 "or skips" contradicts §10.1's "not an error"
so E2 and E16 cannot both pass (finding 5); re-tagging a `--dry-run` row flips a healthy `planned` row
to `failed` and hides it from the sidebar (finding 6); the re-processing policy omits `failed` and
`skipped` (finding 10); and nothing says how the shipped `.acmodel` pair comes to exist or what its
relative path resolves against (finding 11). E7's `val_top1 >= 0.55` and its `>= 14/20` identity leg are
also unreconciled, with a sanctioned remedy for only one of them (finding 9).

Component relationships for this step:

```
docs/DESIGN.md  ─┐
docs/PLAN.md    ─┼─→ (this step: cold review) ─→ docs/design-review.md ─→ docs/design-review.json
docs/RECON.md   ─┘              │                                              (gate: verdict)
                                ├─→ scripts/recon/verify_design_facts.py (new, read-only)
                                └─→ docs/PROGRESS.md (this entry)
   ^                    ^                      ^
data/raw/annotations   data/raw/CUB_200_2011.tgz   models/ + models/backbones/
(recount all 15 facts) (real classes.txt names)    (assets exist, sizes match RECON)
```

Sequence:

```
step -> DESIGN/PLAN/RECON            : read all 1542 + 151 + 320 lines
step -> grep DESIGN.md               : audit the 6 blocking conditions directly
step -> COCO annotations             : re-measure all 15 numbers via the design's split_for
step -> CUB_200_2011.tgz classes.txt : read the REAL label strings -> finding 2
step -> sqlite3 + published schema    : reproduce the skipped-table IntegrityError -> finding 1
step -> published slug()             : run it on real awkward names (verified correct)
step -> design-review.md             : narrative, findings, verified/wrong assumptions
step -> design-review.json           : gate JSON (verdict CHANGES_REQUESTED)
step -> PROGRESS.md                  : this entry
```

Next: the loop returns to the design step for iteration 4. No HIGH and no blocker, so the 11 MEDIUMs
are all small, local edits — six of them are one clarifying rule each, and the two schema/label ones
(findings 1 and 2) are the only changes with implementation consequences.
