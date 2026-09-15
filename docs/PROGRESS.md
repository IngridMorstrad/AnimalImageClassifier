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


## 2026-09-15 21:35 UTC — orchestrator — first workflow aborted, build workflow relaunched
Run `wf_7b416174f3949cd5` **aborted**. Cause: the `design-loop` was configured `maxIterations: 3`,
`onMaxIterations: "abort"`, gated on the design reviewer returning `verdict == "APPROVED"`. The
reviewer never approved (iteration 3 verdict CHANGES_REQUESTED: 0 HIGH / 11 MEDIUM / 8 NIT), so the
loop exhausted its budget and aborted before `plan`, `build-loop` and `consolidate` ever ran. Three
hours produced a 1542-line DESIGN.md and no pipeline code. The gate was set too strictly by the
orchestrator — a reviewer that blocks on MEDIUM findings will always find another MEDIUM.

Corrective actions:
- **DESIGN.md is frozen** as the spec of record (0 HIGH findings). No design step in the new run.
- The 11 outstanding MEDIUM findings are converted into code work items, including the two the
  reviewer reproduced against real data: the `skipped` table PRIMARY KEY raising `IntegrityError`
  on any second run, and raw dataset class names leaking into paths (`~/animal_pics/022_chuck_will_widow/`).
- New run `safari-classifier-build`: `plan` → `build-loop` (maxIterations **24**,
  onMaxIterations **continue**, so exhaustion cannot kill it) → `consolidate`.
- The build reviewer has an explicit anti-stall rule: only correctness / spec-violation / data-loss /
  safety defects may block; style and polish go to docs/FOLLOWUPS.md.
- Implementation is sequenced so a working end-to-end slice (scan → detect → decide → materialize →
  catalog → `classify` CLI green on fixtures) lands FIRST, before model training and GUI.

Recon superseded an earlier orchestrator caveat: real training IS possible in-sandbox. timm
`efficientnet_b0` / `convnext_nano` load from GitHub release assets, and `s3.amazonaws.com` reaches
the fast.ai mirror for CUB-200-2011 and COCO. The synthetic-dataset fallback is retired.


## 2026-09-15 21:42 UTC — plan step — implementation checklist written, DESIGN.md left frozen
Converted the frozen `docs/DESIGN.md` (iteration 3, 1,542 lines) into an ordered implementation
queue. **No design work was done**: DESIGN.md, PLAN.md, RECON.md and design-review.md were read only,
and not one byte of any of them was modified. No new design document was produced. No pipeline source
code was written in this step — the build loop writes the code.

Deliverables:
- `docs/IMPL-PLAN.md` — **26** `- [ ]` chunks, each naming its exact absolute file paths, what it must
  do and the e2e test that proves it, sequenced so a **working end-to-end slice lands first**:
  dependency contract → errors/config → catalog → taxonomy/slug → scan → images → detector protocol +
  scripted detector → decide + materialize → `classify` wired green on committed fixtures (chunk 9) →
  idempotency/limit/reclassify (10) → immutability + link modes + collisions (11). Only then formats/
  skips/fail-loud (12–13), real MegaDetector + E4 against COCO ground truth (14), training and the
  species head (15–17), birds (18–19), GUI (20–22), export-trainset/verify/shipped artifacts/docs
  (23–26).
- `docs/impl-status.json` — `{"complete": false, "total_items": 26, "done_items": 0, "current_chunk":
  "1. Dependency contract and the three probes on the locked env"}` for the build loop's reviewer.

All 11 outstanding MEDIUM findings from `docs/design-review.md`, all 8 NITs, and the 2 defects
reproduced against real data are folded in as explicit items, with a traceability table at the foot of
IMPL-PLAN.md mapping each to its owning chunk. The two reproduced defects get regression gates rather
than prose: the `skipped` PRIMARY KEY `IntegrityError` is fixed by an `ON CONFLICT(path) DO UPDATE`
upsert in chunk 3 and proved by chunk 10 running the pipeline **twice** over the same fixture tree
(second run must succeed as a clean no-op, with exactly one `skipped` row carrying the second run's
`run_id`); the raw-class-name leak gets a normalization layer in chunk 4
(`022.Chuck_will_Widow` → `chuck_will_widow`, display names from the authoritative
`taxonomy/data/cub200.csv`) and chunk 18 asserts every created label directory matches
`^[a-z][a-z0-9_]*$` with no leading digits, so `~/animal_pics/022_chuck_will_widow/` can never appear.

Non-negotiables carried into the plan verbatim: `dominance_ratio` 1.6 is the only size gate and there
is **no `min_box_area`** under any name (chunk 2's fatal unknown-TOML-key check is what stops one being
reintroduced by config, chunk 9's E6 asserts `classify --help` exposes no such option); the closed
label set; read-only source with atomic temp+`os.replace` writes and the I1 no-write rule for
`--hardlink`'s aliased inode; copy default with `--link`/`--hardlink`; sha256-keyed idempotent
resumable runs against `<root>/.catalog.db`; the FastAPI + vanilla-JS GUI on 127.0.0.1:8765 with a
re-tag that MOVES the file and records the override; `ebird_enrich` optional/off/unreachable and
`hosted_bird_api` failing loudly; **E2E tests only** under `tests/e2e/`; and no silent default for any
required value.

Verified with real output in this step (read-only inspection, nothing built):
- `git -C … log --oneline -5` → HEAD `3335e88` on `feat/safari-classifier`, clean tree.
- Assets on disk: `models/md_v5a.0.0.pt` 280,766,885 B; `models/backbones/`
  `efficientnet_b0_ra-3dd342df.pth` + `convnext_nano_d1h-7eb4bdea.pth`; `data/raw/CUB_200_2011.tgz`,
  `val2017.zip`, `annotations_trainval2017.zip`, and `data/raw/annotations/instances_val2017.json`
  already extracted. Everything chunks 14–18 need is local — no blocked host is on the path.
- `.venv/lib/python3.12` exists, so the 3.12 pin chunk 1 writes matches the venv the probes ran in
  (the host `python3` is 3.9.25, which is exactly why `.python-version` is load-bearing).
- `rg -c '^- \[ \] ' docs/IMPL-PLAN.md` → **26**, matching `total_items`.

Blocked / risks recorded, not hidden:
- Three thresholds cannot be known until the suite runs — E4's `>= ceil(0.8 * len(list))` against real
  MegaDetector output, E7's `val_top1 >= 0.55` plus its identity leg, and E10's `>= 0.90` in < 90 s.
  The plan carries DESIGN.md §11.2's sanctioned remedies inline (re-freeze E4 at ratio > 4.0 and let
  the list length follow the data; raise E7's `--input-size` toward 224 and/or `--epochs-finetune`) and
  forbids the alternatives (never lower a threshold in test code, never pad a frozen list with
  train-bucket images, never widen `min_species_confidence` for a test).
- `pyproject.toml` on disk still says `requires-python = ">=3.10"` with `typer` as its only dependency,
  and there is no `.python-version` — i.e. the §2.1 dependency contract is **not yet in place**. That is
  chunk 1, and nothing else may start until `uv sync --frozen` plus all three RECON probes pass on the
  locked environment.
- `api.ebird.org` answers `000` here, so chunk 19's live legs can only assert the degradation path; the
  matching rule is covered by the single sanctioned `httpx.MockTransport` seam.

Component relationships for this step:

```
docs/DESIGN.md (frozen)  ─┐
docs/design-review.md    ─┼─→ (this step: plan only) ─→ docs/IMPL-PLAN.md ──→ build-loop (coder)
docs/RECON.md            ─┤                         └─→ docs/impl-status.json (reviewer's gate)
src/animal_classifier/cli.py (placeholders) ─┘       └─→ docs/PROGRESS.md (this entry)
```

Sequence:

```
step -> DESIGN.md §1-§15        : read all 1542 lines (stack, pipeline, schema, GUI, E1-E26, build order)
step -> design-review.md        : lift the 11 MEDIUMs + 8 NITs into owned checklist items
step -> RECON.md                : confirm every asset the plan leans on is local
step -> disk (models/, data/raw): verify sizes/presence directly, not from the document
step -> cli.py, conftest.py     : confirm the placeholder surface the plan must replace
step -> pyproject.toml, uv.lock : confirm the dependency contract is NOT yet applied -> chunk 1
step -> docs/IMPL-PLAN.md       : 26 ordered chunks + traceability table
step -> docs/impl-status.json   : complete=false, total_items=26
step -> docs/PROGRESS.md        : this entry
```

Next: the build loop starts at chunk 1 (dependency contract + the three probes), then chunks 2–8, and
the first green e2e run is chunk 9 (`classify` on the fixture card: E2, E5, E6, E26).

---

## 2026-09-15 21:51 UTC — build iteration 1: chunks 1 and 2 (dependency contract; errors.py + config.py)

`docs/build-review.json` did not exist, so this was the first build iteration: no blocking findings to
fix, start at the first unchecked chunk of `docs/IMPL-PLAN.md`.

### Chunk 1 — dependency contract and the three probes (DONE)

`pyproject.toml` rewritten exactly to DESIGN.md §2.1: `requires-python = ">=3.12,<3.14"`, the 13
pinned runtime dependencies, `[project.optional-dependencies] raw = ["rawpy==0.27.1"]` /
`dev = ["pytest==9.1.1"]`, and the `[tool.uv] override-dependencies` that drop `opencv-python`,
`roboflow` and `sahi` via a marker that can never be true. Added `.python-version` containing `3.12`.
`[tool.pytest.ini_options]` keeps `testpaths = ["tests/e2e"]` and gains `addopts = "-ra"` plus the
`slow` marker. `uv.lock` regenerated and committed.

Real output — `uv lock` then `uv sync --frozen --extra dev` (tail):

```
 + pytest==9.1.1
 - requests-toolbelt==1.0.0
 - roboflow==1.4.2
 - sahi==0.12.6
 - shapely==2.1.2
 + starlette==1.6.0
 + uvicorn==0.39.0
```

Real output — installed versions in the synced venv (`uv run --frozen python -c "importlib.metadata"`):

```
torch==2.14.0            typer==0.27.2        setuptools==80.10.2
torchvision==0.29.0      pillow==12.3.0       pytest==9.1.1
timm==1.0.29             pillow-heif==1.7.0   absent (good): opencv-python
numpy==2.5.3             fastapi==0.141.1     absent (good): roboflow
yolov5==7.0.14           uvicorn==0.39.0      absent (good): sahi
opencv-python-headless==5.0.0.93              httpx==0.28.1
```

Real output — the three RECON probes, re-run against that venv:

```
### probe_md_checkpoint
torch 2.14.0+cu130
checkpoint /projects/sandbox/AnimalImageClassifier/models/md_v5a.0.0.pt (280766885 bytes)
LOAD OK: top-level type = <class 'dict'>
  model class = models.yolo.DetectionModel
  model parameters = 140,054,656
  model.names = ['animal', 'person', 'vehicle']
  model.stride = tensor([ 8., 16., 32., 64.])

### probe_md_inference
forward OK in 0.92s on CPU
  raw prediction tensor shape = (1, 25500, 8)
  class names = ['animal', 'person', 'vehicle']

### probe_backbones
timm 1.0.29 / torch 2.14.0+cu130
efficientnet_b0: LOAD OK params=5,288,548 missing=0 unexpected=0 logits=(1, 1000)
convnext_nano: LOAD OK params=15,593,560 missing=0 unexpected=0 logits=(1, 1000)
```

All three reproduce RECON's original measurements and **no pin moved**, so the verified environment is
intact. A dated, append-only "re-verified" note recording this (including the harmless 113-vs-109
package-count difference, which is the lockfile counting the root project and the `raw`/`dev` extras)
was added to the end of `docs/RECON.md`.

### Chunk 2 — `errors.py` + `config.py` (DONE)

`src/animal_classifier/errors.py`: `AnimalClassifierError` base carrying the exit code as a class
attribute, with `ConfigError` (3), `AssetError` (3), `DecodeError` (4), `MaterializeError` (1) and
`CatalogError` (3), plus the named constants `EXIT_OK/UNEXPECTED/USAGE/CONFIG/PARTIAL` = 0/1/2/3/4.
A single call site may override the code where §10.1 gives one failure a different class (the
cross-filesystem `--hardlink` `EXDEV` is a config error, not an IO error).

`src/animal_classifier/config.py`: frozen (`slots=True`) `Config` dataclass + `Config.resolve()`
implementing CLI → env (`ANIMAL_CLASSIFIER_*`) → TOML → default, the §3.1 `formats` family policy,
`--device` resolution, the nesting guard in both directions plus identical, the fatal unknown-TOML-key
check, a fatal missing `--config`, and the §10.2 range checks. `catalog_path` is always
`<output_root>/.catalog.db`; `to_json_dict()` produces the `runs.config_json` payload and reduces the
eBird key to a presence flag. Relative `species_model` / `bird_model` / `detector_weights` resolve
against the **current working directory** and every message prints the resolved absolute path
(finding 11a); a missing `species_model` prints the exact copy-pasteable `animal-classifier train …`
invocation (finding 11b). `Config.resolve` creates **no** directories, so a fatal config error can
never leave a destination tree behind — the property E22 asserts.

Invariant I2 is structural here: `KNOWN_TOML_KEYS` is exactly the §3 key set, so `min_box_area` in a
TOML file is a fatal unknown key, and its error message says so explicitly.

Verified with real output (`uv run --frozen python` driving `Config.resolve` over 25 cases):

```
defaults: copy 1.6 0.45 (jpeg, png, tiff, heic) cpu 7 8765 1280 536870912
catalog: /tmp/tmp_mdksfl8/pics/.catalog.db
eligible ext: ['.heic', '.heif', '.jpeg', '.jpg', '.png', '.tif', '.tiff']
toml layer: 2.5 3 link        env over toml: 3.0        cli over env: 1.9
exit3 missing --config: --config file does not exist: /tmp/tmp_mdksfl8/nope.toml
exit3 unknown TOML key (min_box_area): unknown key(s) in …/bad.toml: min_box_area. Valid keys are:
      bird_model, …, output_root, species_model. Note there is deliberately no box-area floor
      setting; dominance_ratio is the only size gate.
exit3 dominance_ratio=0.5: must be >= 1.0, got 0.5 (a ratio below 1.0 would make the smaller box
      dominant)
exit3 --formats raw: formats (--formats) contains 'raw', which is not one of {jpeg, png, tiff, heic}.
      RAW is not a format family: enable it with --raw …
exit3 device cuda: --device cuda was requested but no CUDA device is available on this host. Omit the
      flag to auto-select, or pass --device cpu.
exit3 output inside source / source inside output / identical  (all three directions)
exit3 hosted_bird_api: … would require ANIMAL_CLASSIFIER_HOSTED_BIRD_API_URL and
      ANIMAL_CLASSIFIER_HOSTED_BIRD_API_KEY …
exit3 ebird no key: requires a non-empty ANIMAL_CLASSIFIER_EBIRD_API_KEY …   OK with key: ebird_enrich
exit3 image_size 700 (not /64) / crop_margin 0.9 / detector_iou 0 / SOURCE missing / SOURCE nonexistent
exit3 missing species_model: … not found at /projects/…/models/species.acmodel. Produce it with:
        animal-classifier train --manifest data/manifests/coco_species.jsonl --out /projects/…
exit3 missing detector weights: … Download MegaDetector v5a from https://github.com/…/md_v5a.0.0.pt
output_root created? False
```

Also verified: `uv run --frozen python -c "import animal_classifier.config, animal_classifier.errors"`
→ `imports clean`; `uv run --frozen pytest tests/e2e -q` → `no tests ran in 0.00s` (0 failures, as the
chunk-2 gate requires — the first tests land in chunk 9); and
`rg -n 'min_box_area|min_box_area_frac|min_animal_area|box_area_floor' src/ tests/ pyproject.toml`
matches nothing but the docstring in `config.py` that states the absence as an invariant.

### One deliberate, reasoned deviation from DESIGN.md (not a silent one)

§8 says `--formats` should be a `list[Format]` typer option, "so an invalid family is a typer usage
error" — which would exit **2**. But §11's E22 row and IMPL-PLAN chunk 13 both require `--formats raw`
to exit **3** with the offending value in stderr. Two independent test expectations outrank the
implementation hint, so `--formats` is taken as `list[str]` by typer and validated in `config.py`,
raising `ConfigError` (exit 3) with the valid family set and a pointer to `--raw`. Recorded here so
the reviewer sees a decision, not an accident.

### Also worth flagging

`detector_max_det` is a real config key (default 100, int in [1, 10000]) even though §3's TOML block
does not list it: IMPL-PLAN chunk 14 requires the detector to read `config.detector_max_det` rather
than hard-code §5.4's inline `max_det=100`. It is therefore in `DEFAULTS` and hence in
`KNOWN_TOML_KEYS`.

### Blocked

Nothing. Next up is chunk 3 (`catalog.py`: the seven tables, four indexes, WAL, the schema-version
guard, the `skipped`/`sources` upserts that fix DEFECT 1, the re-inference transaction, and the full
re-processing policy).

## 2026-09-15 21:54 UTC — test runner — ran the e2e suite: 0 collected, 0 failures
Command: `uv run pytest tests/e2e -v` (plus `--collect-only` to confirm). Real output captured
verbatim in `docs/test-report.md`.

Counts: **passed 0 / failed 0 / errored 0 / skipped 0 — collected 0.** The suite is empty:
`tests/e2e/` holds only `conftest.py`, which defines no fixtures yet. Nothing passed, nothing
broke — there is simply nothing to run. This matches `impl-status.json` (2 of 26 chunks done,
current chunk 3) and the note that the first e2e tests land in chunk 9.

Two things for whoever reads this next:
- pytest exits **5** on an empty suite, so a gate that keys off a non-zero exit code will flag
  this run as a failure even though no test failed. Resolves itself once chunk 9 lands.
- E2E-only spec is respected: zero `.py` files under `tests/` outside `tests/e2e/`.

No source or test file was edited in this step. `uv sync` was not needed. Blocked: nothing.

---

## 2026-09-15 21:57 UTC — gate review, iteration 1: **CHANGES_REQUESTED (1 blocking finding)**

Reviewed the real diff `6a3cc67..HEAD` (commits `a8103ab`, `c59a656`) plus the captured e2e output
in `docs/test-report.md`. Did not re-run the suite. Full narrative in `docs/build-review.md`;
machine verdict in `docs/build-review.json`.

**The one blocking finding is sequencing, not a defect.** `docs/impl-status.json` has
`complete: false` with `done_items: 2` of `26`, and approval requires `complete == true`. Nothing in
`errors.py`, `config.py` or `pyproject.toml` needs to be undone. Continue at chunk 3.

**Chunks 1-2 are clean against every item on the blocking checklist.** Specifically verified:

- **No area floor anywhere.** `rg` over the repo matches `min_box_area` and friends only in prose
  asserting their absence. No box area is compared to any constant. `dominance_ratio` is
  range-checked `minimum=1.0` with the rationale string spelling out why — invariant I2 exactly.
  The unknown-TOML-key error message itself says a floor setting deliberately does not exist, so
  reintroducing one by config fails loudly.
- **No silent defaulting of a required value.** Missing SOURCE, `ebird_enrich` without its key
  (empty string included), `hosted_bird_api`, `--device cuda` on this CUDA-less host, a broken torch
  import, and each missing model asset all raise with the resolved absolute path and a
  copy-pasteable remedy.
- **e2e-only respected.** Zero `.py` files under `tests/` outside `tests/e2e/`; `testpaths` pinned.
- **TOML key set matches DESIGN.md §3 exactly**, plus the one documented, justified addition
  `detector_max_det`.

**On the empty suite:** the report claims nothing passed and says so plainly, so this is not a
"green without evidence" violation and I did not treat it as one. The honest consequence is that
every correctness claim above rests on source reading, not observed behaviour, until E22 and E6 land
in chunk 9.

**Still entirely unproven** and the reason the loop must continue: the chunk-3 `skipped` PRIMARY KEY
`IntegrityError` / second-run-no-op fix; the first e2e tests (chunk 9); atomic writes and the
`--link`/`--hardlink`/`--dry-run`/`--reclassify` contract (chunk 11); low-confidence-files-as-`unknown`
and human-readable species slugs (chunks 15-19).

**5 non-blocking observations** recorded in the new `docs/FOLLOWUPS.md` — none withholds approval.
The two worth acting on soon: `output_root` uses `os.path.abspath` while `source_root` uses
`Path.resolve()`, so a symlinked `output_root` could evade `_guard_nesting` (fix before
`materialize.py` lands in chunk 11); and `pytest` sits in the `dev` extra that neither documented
test command requests, which will break under `--frozen` on a clean machine.

---

## 2026-09-15 22:11 UTC — chunk 3: `catalog.py` (schema, upserts, statuses) + DEFECT 1 fixed

**Blocking finding B1** from build review iteration 1 was "keep going: 2 of 26 chunks done". This
entry advances that: chunk 3 is complete, chunks 1-3 of 26 are ticked, and the e2e suite is now
non-empty and green.

### What landed

`src/animal_classifier/catalog.py` (1033 lines) — the seven tables and four indexes of DESIGN.md
§5.9 verbatim plus the `meta` schema-version table, `journal_mode=WAL`, `foreign_keys=ON`,
`busy_timeout=10000`, `run_id = f"{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"`, and "newest run"
defined once as `ORDER BY started_at DESC, run_id DESC LIMIT 1`.

- **DEFECT 1 is structurally fixed.** `skipped` is written only through the module-level
  `SKIPPED_UPSERT_SQL` constant (`ON CONFLICT(path) DO UPDATE SET reason/detail/run_id/seen_at`),
  and `record_skip` is the only write path into the table. `sources` uses the same shape on `path`
  via `SOURCES_UPSERT_SQL`. The constants are named so the gate `rg "INSERT INTO skipped" src/`
  lands on a statement whose name says upsert.
- **`replace_inference()`** publishes §5.9's re-inference transaction as one function in the fixed
  order (delete candidates → delete boxes → insert boxes → insert candidates → `UPDATE images`),
  inside `BEGIN IMMEDIATE`. It never touches `overrides` (append-only, I6), never rewrites
  `first_seen`, and never sets `label_source`, so a human label survives `--reclassify`.
- **`plan_disposition()`** is the full re-processing policy in one place: no row → process; `done` →
  skipped unless `--reclassify` and skipped *before* the budget is consulted; `planned` /
  `materializing` → re-processed; `failed` → **retried, consuming `--limit` budget**; `skipped` →
  re-evaluated.
- **Lock contention is a per-image outcome.** `do_write()` retries the whole transaction on
  0.5/1/2/4/8 s and then raises the new `CatalogLockedError` (exit 4, added to `errors.py`), never a
  fatal exit 1. `write_tx()` is the single-attempt form the GUI's re-tag uses to turn contention
  into a fast 409. A read-only `Catalog` refuses to write at all.
- **Fail-loud, no silent defaults:** `update_image`/`ensure_image` reject unknown column names
  against `UPDATABLE_IMAGE_COLUMNS`; an update that matches no row raises rather than reporting
  success; `start_run` refuses an empty `source_root`; duplicate box `idx` or candidate `rank` raise
  before touching the DB; a schema version newer than this build refuses to open (exit 3).

Two review follow-ups fixed ahead of their deadlines: `output_root` is now `Path.resolve()`d so a
symlinked output root cannot evade `_guard_nesting`, and `pytest` moved to a PEP 735
`[dependency-groups] dev` so `uv run --frozen pytest` resolves on a clean machine.

`tests/e2e/test_cli_surface.py` lands **E6's CLI-surface leg** early (chunk 9 still owns E6's full
form): every command's `--help` is asserted to offer no `min-box-area` / `min_area` / `area-floor` /
`min-animal-area` / `min-box-frac` option, which is the executable guard on invariant I2.

### Verified with real output (commands and results)

```
$ uv run --frozen pytest tests/e2e -v
7 passed in 1.04s          (exit 0 — the suite was 0 collected / exit 5 before this commit)

$ rg -n "INSERT INTO skipped" src/
catalog.py:194 -> SKIPPED_UPSERT_SQL (the upsert); the other 3 hits are prose. No bare INSERT.

$ uv run --frozen python /projects/sandbox/_verify_chunk3.py     # driver outside the repo
pass1 disposition: process ; pass2 (same card, no --reclassify): skip_done
row counts after 2 passes: images 2, sources 2, boxes 2, candidates 3, skipped 2, runs 2
  -> the second run over the same card is a clean no-op: no IntegrityError, no duplicated rows
pass3 with --reclassify: boxes 2, candidates 3 (unchanged — replace, not append)
override survives re-inference: 1 ; boxes after an empty re-inference: 0
failed-hash disposition: process (retry) ; done: skip_done ; done + --reclassify: process
journal_mode: wal  foreign_keys: 1  busy_timeout: 10000
tables: boxes candidates images meta overrides runs skipped sources
indexes: idx_boxes_sha idx_boxes_sha_idx idx_images_label idx_sources_sha
orphan source rejected by FK: FOREIGN KEY constraint failed
unknown column (min_box_area) -> CatalogError exit=3
update of a missing row -> CatalogError exit=3
duplicate box idx -> CatalogError exit=3
empty source_root -> CatalogError exit=3
schema_version 99 -> CatalogError exit=3 ("refusing to touch it")
write on a read-only catalog -> CatalogError ; read on it works
missing DB opened read-only -> CatalogError naming the path
contended write -> CatalogLockedError exit=4 (not a dead run)
rolled-back transaction leaves 0 rows
ALL CHUNK 3 CHECKS EXECUTED

$ uv run --frozen python /projects/sandbox/_verify_symlink_guard.py
ConfigError: output_root /…/card/DCIM/out is inside SOURCE /…/card   # symlink evasion now caught
```

The catalog driver is a throwaway script kept **outside** the repo on purpose: e2e tests only, and
the catalog's real e2e proof is E2 (chunk 9) and E13's second-run and retry legs (chunk 10).

### Blocked / not yet proven

Nothing is blocked. Still unproven by an e2e test: everything the catalog exists to serve — a real
`classify` run needs `scan.py`, `images.py`, `detect/`, `decide.py` and `materialize.py` (chunks
5-8), so DEFECT 1's regression test arrives with E13 in chunk 10 as planned. 23 of 26 chunks remain,
so `impl-status.json` `complete` stays `false`.

---

## 2026-09-15 22:14 UTC — test step — real e2e run: 7 passed, 0 failed, exit 0

Command actually run (nothing else; no source or test file edited, `uv sync` not needed):

```
cd /projects/sandbox/AnimalImageClassifier && uv run pytest tests/e2e -v
```

Real counts from the captured output: **7 passed, 0 failed, 0 errored, 0 skipped**, 7 collected,
1.00 s, **exit code 0**. Verbatim stdout is in `docs/test-report.md` §2, untruncated. This is the
first non-zero-exit-0 suite: the previous run (21:54 UTC) collected 0 tests and exited 5.

All 7 tests come from `tests/e2e/test_cli_surface.py` and assert on `--help` output only —
`test_top_level_help_lists_every_command` (all six commands present, exit 0) and
`test_no_command_offers_an_area_floor` parametrized over the six commands, each checking that
`<command> --help` exits 0 and contains none of 10 spellings of an absolute area floor. So
**invariant I2 now has a green executable guard**: if anyone reintroduces `min_box_area` under any
name, the suite fails immediately rather than waiting for review.

Scope honesty: this run proves nothing about detection, the dominance rule, materialization, the
catalog, the GUI, training, or the bird providers. It is the CLI-surface leg of E6; E6's full form
(a real `classify` over the fixture card) lands in chunk 9. `impl-status.json` reads
`done_items: 3` / `total_items: 26`, `current_chunk: 4 (taxonomy/)`, so 7 CLI tests is the expected
state here, not a coverage regression.

Spec compliance: **no violation**. `tests/` holds exactly two `.py` files, both under `tests/e2e/`
(`conftest.py`, `test_cli_surface.py`); no unit test exists outside `tests/e2e/`, and
`pyproject.toml` still pins `testpaths = ["tests/e2e"]`. Nothing deleted.

### Blocked / not yet proven

**Nothing blocked.** No failure to hand to the `code` step. Unproven by any e2e test so far:
everything past the CLI surface — chunks 4-26, including E1-E5 and E7-E26. 23 of 26 chunks remain,
so `impl-status.json` `complete` stays `false`.
