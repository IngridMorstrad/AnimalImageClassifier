# Design review — `docs/DESIGN.md` (iteration 1)

Reviewed 2026-09-15 15:36 UTC, cold: `DESIGN.md` read against `PLAN.md` (agreed behaviour) and
`RECON.md` (measured reachability), with every load-bearing claim re-probed in this sandbox rather
than taken on the document's word. Commands and their real output are quoted below.

**Verdict: CHANGES_REQUESTED** — 2 HIGH, 15 MEDIUM, 8 NIT.

None of the hard blocking conditions fire. Specifically, and checked explicitly:

| Blocking condition | Result |
|---|---|
| `min_box_area` / minimum box area / any absolute area floor | **Absent.** §3, §5.7 and I2 forbid it by name and by concept; §5.3 even keeps degenerate boxes in the dominance count "because excluding them would be an area floor by the back door". `dominance_ratio = 1.6` is the only size gate. |
| Unit tests or any layer other than `tests/e2e` | **Absent.** §2, §11 and `[tool.pytest.ini_options] testpaths = ["tests/e2e"]` agree; E1–E25 all drive the console script or the real ASGI app. |
| Dependence on a RECON-unreachable asset without a buildable fallback | **None.** Every artifact used is local and verified (MegaDetector, timm backbones, CUB, COCO). `ebird_enrich` is opt-in, degrades, and E23 tests the degradation; `hosted_bird_api` is a declared stub; Playwright is deliberately avoided. |
| Writes/moves/renames/deletes in the source tree | **None.** Three mechanisms (§5.1) plus E1's byte-for-byte snapshot. |
| Silent default for a missing required value | **None specified.** The one degradation (`ebird_enrich`) is optional enrichment and is recorded as `provider_status='unreachable'`. Finding 12 is a *gap* where such a substitution could be written, not a specified one. |
| Any required element omitted | **None.** MegaDetector v5a ✅ · own finetuned species model + training subsystem + `train` ✅ · three-implementation bird provider ✅ · `species\|multiple\|landscape\|junk\|unknown` ✅ · atomic writes ✅ · copy/`--link`/`--hardlink` ✅ · sha256-keyed idempotent resumable ✅ · SQLite catalog ✅ · FastAPI + vanilla-JS GUI on `127.0.0.1:8765` with re-tag that moves files and records overrides ✅ · typer CLI `classify\|gui\|train\|eval\|export-trainset\|verify` ✅ · format policy ✅. |

The design is strong: the pipeline is coherent, the fail-loud table is real, and the dominance rule
is stated as a pure function with the right degenerate-case reasoning. What blocks it is a
dependency contract that does not match reality, one undefined label path that real COCO data will
hit, and a set of test criteria that are not yet decidable by an implementer.

---

## Findings

### HIGH

**1. HIGH — The dependency contract in §2/§2.1 is partly wrong, and the repo cannot reproduce it.**
`§2` states typer **0.25.x** ("Resolution pins it: `yolov5` → `roboflow` → `typer<0.26`. Verified by
`uv lock`"), and §2.1 claims a verified lock of "131 packages … `opencv-python-headless` 5.0.0.93 …
`typer` 0.25.1". The repo contradicts this and so does a fresh resolution:

- `pyproject.toml` declares exactly one dependency, `typer>=0.12`; `uv.lock` contains **9**
  packages (`grep -c '^\[\[package\]\]' uv.lock` → 9) and pins `typer` **0.27.2**. Nothing in the
  repo resolves yolov5, torch, fastapi or opencv at all, so the §2.1 verification is not
  reproducible here.
- A real resolution of the intended set (`uv pip compile req.in --override over.txt
  --python-version 3.12`) yields **115** packages with **`typer==0.27.2`** — because with typer
  unpinned the resolver selects **`roboflow==1.3.8`**, which has *no* typer cap. The `typer<0.26`
  cap exists only in `roboflow` 1.4.2 (the version that happens to be installed in `.venv`, where
  typer is nonetheless 0.27.2). The pin is therefore resolution-order dependent, and the design's
  self-imposed rule "the CLI must not use 0.26+-only APIs" rests on a false premise.
- The same resolution gives `opencv-python-headless==4.10.0.84`, not 5.0.0.93 (4.10.0.84 is
  `roboflow`'s floor; the 5.0.0.93 in `.venv` came from an ad-hoc install).

This is HIGH rather than cosmetic because §2.1 also states that "`uv run` **syncs the venv to
`pyproject.toml` + `uv.lock`**". The first `uv sync` in build step 1 will therefore *replace* the
environment in which RECON verified MegaDetector loading, MegaDetector inference and both backbone
loads — a silent regression of every verified fact, on the first command the implementer runs.
The override itself is sound and I confirmed it: the compiled output contains
`opencv-python-headless` and **no** `opencv-python` line at all.

*Fix.* Replace the §2 table's "verified" numbers with pins that reproduce the verified environment,
and make the lock the artifact of record:

```toml
dependencies = [
  "typer==0.27.2", "torch==2.14.0", "torchvision==0.29.0", "timm==1.0.29",
  "yolov5==7.0.14", "opencv-python-headless==5.0.0.93", "pillow==12.3.0",
  "pillow-heif>=1.1.1",            # see finding 8 — replaces pi-heif
  "fastapi==0.141.1", "uvicorn[standard]==0.39.0", "httpx==0.28.1", "setuptools<81",
]
[tool.uv]
override-dependencies = ["opencv-python; python_version < '3.0'"]
```

and add to §2: "`uv lock` is committed; `uv sync --frozen` must reproduce torch 2.14.0 /
torchvision 0.29.0 / timm 1.0.29 exactly, because RECON's detector and backbone verification was
performed on those versions. If a resolution moves them, re-run `scripts/probe_md_checkpoint.py`,
`scripts/probe_md_inference.py` and `scripts/probe_backbones.py` and record the result in RECON
before proceeding." Delete the `typer<0.26` claim; if 0.25-compatibility is genuinely wanted, pin
`typer>=0.25,<0.28` **and** `roboflow==1.3.8` so the resolution is deterministic either way.

**2. HIGH — The label for a degenerate dominant box is undefined, and real data hits it.**
§5.3 says a box under 2 px after clipping is recorded `species = NULL, species_status =
"degenerate"`, is excluded from classification, but "still count[s] as animals for the dominance
rule". §5.7's pseudocode then calls `species_or_unknown(animals[0])` — a function that is never
defined anywhere in the document — on a box that has no species and no score. Both plausible
readings (`unknown`, or crash on `None < 0.45`) are reachable, and this is not hypothetical: in
`data/raw/annotations/instances_val2017.json`, **7 non-crowd animal boxes are under 2 px in one
dimension**, and the smallest animal box is `area_frac = 0.000012`, so the fixture corpus contains
the case.

*Fix.* Define the helper explicitly in §5.7 and give it a test:

```python
def species_or_unknown(box) -> str:
    # a degenerate box has no classifiable pixels: identity unknown, presence certain
    if box.species_status == "degenerate" or box.species_conf is None:
        return "unknown"
    if box.species_conf < cfg.min_species_confidence:
        return "unknown"
    return slug(box.species_common)
```

Add to §11: **E26** — `classify --detector scripted` with a 1 px animal box as the only detection
asserts label `unknown`, `species_status='degenerate'` in `boxes`, the file materialized under
`unknown/`, and exit code 0 (a degenerate box is not a failure).

### MEDIUM

**3. MEDIUM — Types referenced but never defined: `BirdResult`, `GpsPoint`, `Prediction.top5`.**
§5.6 declares `def refine(...) -> BirdResult` without saying what a `BirdResult` contains or how it
merges back into the box row, and §5.5's `Prediction(common, scientific, score, top5, model_id)`
never states `top5`'s element shape (the `candidates` table wants `rank, common, scientific,
score`). Two implementers will write two different merge rules — in particular whether the bird
head's score overwrites `boxes.species_conf` and whether the coarse `bird` candidates stay in
`candidates`.

*Fix.* Add to §5.6:

```python
@dataclass(frozen=True)
class GpsPoint: lat: float; lon: float
@dataclass(frozen=True)
class Candidate: rank: int; common: str; scientific: str | None; score: float
@dataclass(frozen=True)
class BirdResult:
    provider: str                    # -> images.bird_provider
    status: str                      # 'refined' | 'kept_coarse' | 'unreachable' | 'no_gps'
    prediction: Prediction | None    # None => keep the coarse prediction unchanged
```
Merge rule: on `status='refined'`, `boxes.species_*` and `candidates` are **replaced** by the bird
result (top-5 from the bird head) and `images.model_id` records both ids as
`"<species_model_id>+<bird_model_id>"`; on any other status the coarse row is kept verbatim and
only `provider_status` is written.

**4. MEDIUM — The bird path is unreachable as specified in E8, because refinement triggers off the
coarse head's top-1.** §5.6 fires the bird provider only "when a box's top-1 species rolls up to
`Aves`". E8 trains a 5-species CUB head and classifies a CUB image, but says nothing about which
*species* artifact is loaded; the only real species artifact the suite builds is E7's **6-class**
COCO subset, and §11's E7 example ("6 classes") need not contain `bird` at all. If the coarse head
has no `bird` class, or ranks a warbler as `cow`, `own_bird_head` is never invoked and E8 fails for
a reason unrelated to the bird head.

*Fix.* Pin both ends. In §11 E8: "the species artifact is the 10-class COCO head (which contains
`bird`); the E7 subset must include `bird` when it is reused." In §5.6 widen and make the trigger
explicit: "refinement fires when **any of the coarse top-3** rolls up to `Aves`, or when the coarse
top-1 is below `min_species_confidence` and any top-3 entry is `Aves`" — and add
`--force-bird-head` (documented as a testing/diagnostic affordance, like `--detector scripted`) so
E8 can exercise the bird head independently of coarse-head accuracy.

**5. MEDIUM — E4 has no pass criterion and compares MegaDetector boxes against COCO
ground-truth-derived expectations.** "Images COCO says have a clear area winner get that species;
images with ratio < 1.5 get `multiple`" is not a decidable assertion: MegaDetector detects a
different box set than the COCO annotator (missed small animals, merged herds, extra animals COCO
did not annotate, and `person`/`vehicle` boxes that §5.4 excludes from labelling), so per-image
equality will fail on some fraction and the test will be flaky from day one. Recomputed from the
real annotations at the design's own threshold: of the 471 multi-animal images, **235 have
`ratio >= 1.6` and 236 have `ratio < 1.6`** — the split is essentially even, so "clear winner" is
not a small safe subset either.

*Fix.* Make E4 an aggregate test over a frozen list, and assert the *decision*, not the species:

> E4: `scripts/make_e2e_fixtures.py` writes `tests/e2e/data/coco_dominance.json` containing 40
> image ids with GT `ratio > 3.0` and 40 with GT `ratio < 1.3` (deterministic: sorted by id, first
> 40 of each). `classify --detector megadetector` over those 80 images asserts (a) every label is
> in the closed set, (b) **≥ 32/40** of the `ratio > 3.0` images get a single-species-or-`unknown`
> label (not `multiple`), (c) **≥ 32/40** of the `ratio < 1.3` images get `multiple`, and (d) the
> aggregate counts are printed so a regression is diagnosable. Species identity is asserted in E7,
> not here.

**6. MEDIUM — Nothing prevents train/test leakage between E7 and the accuracy it asserts.** §1 and
§7.2 train the species model on COCO **val2017** — the only labelled multi-animal source in-sandbox
— and E4/E7 then evaluate on COCO val2017 images. The 80/20 `sha1(path)` split is deterministic
(good), but no section says the e2e evaluation images must come from the *val* bucket, so the
obvious implementation reports accuracy on images it trained on and E7's "beats chance by a margin"
becomes meaningless.

*Fix.* Add to §7.2: "the split function `split_for(path) = 'val' if int(sha1(path),16) % 5 == 0
else 'train'` is public and is the single source of truth." Add to E7 and E4: "every image used for
assertion must satisfy `split_for(path) == 'val'`; the test asserts this before running, and the
training manifest is filtered to `split == 'train'` lines only."

**7. MEDIUM — E7 and E10 have no numeric pass thresholds.** "val top-1 beats the 1/6 chance floor
by a margin asserted numerically" does not state the margin, and E10 says only "accuracy above
chance". The implementer must invent both, and whichever number they pick is unreviewable.

*Fix.* State them: **E7** — 6 classes (`zebra, elephant, giraffe, bear, cow, sheep`), train-split
crops only, `--input-size 128 --epochs-head 2 --epochs-finetune 2 --batch-size 32`, assert
`val_top1 >= 0.55` (chance 0.167), `val_top5 == 1.0` trivially skipped for k>n, artifact
`train.val_top1` equal to `metrics.json` top-1 within `1e-6`, and wall clock under 20 min on 8 CPU
cores. **E10** — 4 synthetic shape classes, 500 train / 100 val, assert `val_top1 >= 0.90` and
total runtime under 90 s.

**8. MEDIUM — E16's HEIC fixture cannot be produced with the chosen library.** §2 selects
`pi-heif`, and §5.2 relies on `pi_heif.register_heif_opener()`. Decode is fine, but E16 requires a
HEIC *input file* and `data/raw` contains only JPEG corpora (COCO, CUB), so the fixture builder must
**encode** one. It cannot with `pi-heif`:

```
pi_heif 1.4.0
HEIF ENCODE FAIL: KeyError 'HEIF'        # im.save(buf, format="HEIF")
```

whereas the full package works (fresh venv, installed from pypi):

```
pillow_heif 1.1.1 libheif 1.20.2
HEIF ENCODE OK bytes= 386 ; reopen -> HEIF (64, 64)
```

*Fix.* Use `pillow-heif` (a decode superset of `pi-heif`, plus the x265 encoder) as the runtime
dependency — `PLAN.md` said `pillow-heif`, and the switch to `pi-heif` in §2 is what breaks this —
and state in §11: "`make_e2e_fixtures.py` writes the HEIC fixture by re-encoding a COCO JPEG with
`pillow_heif`; if HEIF encoding is unavailable the builder **fails** with that message rather than
skipping E16." (Do not substitute AVIF: it encodes here, but it is not the format the policy names.)

**9. MEDIUM — The `formats` config key has no defined meaning and no validation.** §3 ships
`formats = ["jpeg", "png", "tiff", "heic"]`, while §5.1 states the extension policy as a hardcoded
allowlist and §10.2 validates neither. Does removing `"heic"` cause `.heic` files to be
`skipped(unsupported_extension)`? Does adding `"raw"` substitute for `--raw`? Is there a CLI flag?
All three are guessable, none is stated.

*Fix.* Define it in §3 and validate it in §10.2: "`formats` selects which of the four supported
format families are eligible; each entry must be one of `jpeg|png|tiff|heic` (unknown value →
fatal, exit 3, listing the valid set); an extension belonging to a family not listed is skipped with
reason `format_disabled` (a new enumerated reason in §5.1). RAW is governed solely by `--raw`, never
by `formats`. `--formats` is exposed on `classify` as a repeatable flag." Add to E16 a
`--formats jpeg` run asserting `format_disabled` for the PNG/TIFF/HEIC fixtures.

**10. MEDIUM — `source_root` has no provenance for `gui` and `verify`.** §6's byte-serving safety
rule permits paths "under `source_root` in read-only mode for a `--dry-run` catalog", and §8's
`verify` checks that "`output_root` … [is] not nested with a source" — but `gui` and `verify` take
only `-o/--output`, and no schema column holds a source root (`runs.config_json` is described as a
config dump, not as an interface). The implementer cannot write either check as specified.

*Fix.* Make it explicit in §5.9 and §8: add `runs.source_root TEXT NOT NULL` (written at run
start), and specify "`gui` and `verify` read `source_root` from the most recent `runs` row; if no
`runs` row exists, `verify` reports the nesting check as `skipped(no_runs)` and the GUI serves
`/full` **only** from `output_root`, returning **409** with `"image was planned by --dry-run and has
no materialized file"` for `status='planned'` rows." That also removes the need for a read-only
source-serving mode.

**11. MEDIUM — Catalog concurrency contradicts the "live progress" design.** §9 says catalog writes
come "from the main thread only — single-writer, which is what keeps SQLite happy", and §10.1 makes
`database is locked` past `busy_timeout` **fatal, exit 1** after 3 retries. But §6 exposes
`GET /api/run` precisely so the GUI can be open *during* a run, and `POST /api/images/{sha}/label`
writes overrides — so a user re-tagging during a 2.8 h run can abort it, contradicting §10.1's own
rule that "a per-image failure does not abort the run".

*Fix.* Split reader from writer in §6/§9/§10.1: "the GUI opens the catalog with
`sqlite3.connect('file:…?mode=ro', uri=True)` for all `GET` routes. `POST /api/images/{sha}/label`
opens a short write connection with `busy_timeout=10000`; if it still cannot acquire the write lock
it returns **409** `{"error": "a classify run is writing the catalog; retry"}` and changes nothing on
disk. In `classify`, lock contention is retried with backoff (0.5/1/2/4/8 s) and, if it still fails,
recorded as a per-image `failed` row (exit 4) — never a fatal exit 1." Add to E19 the 409 path.

**12. MEDIUM — `--device cuda` on a machine without CUDA is unspecified — exactly the shape of
substitution the design forbids.** §7.3 says "`--device auto` picks CUDA if it is ever present" and
§8 offers `--device [auto|cpu|cuda]` on `classify` and `train`, but no row in §10.1 or §10.2 covers
an explicit `--device cuda` with no usable device. Silently running on CPU would violate I7.

*Fix.* Add to §10.2: "`--device`: `auto` → `cuda` if `torch.cuda.is_available()` else `cpu`, and the
choice is logged at `INFO` and stored in `runs.config_json`; `cuda` when
`torch.cuda.is_available()` is false → **fatal, exit 3**, `'--device cuda requested but torch
reports no available CUDA device; omit the flag or pass --device cpu'`; `cpu` always valid."

**13. MEDIUM — `ebird_enrich` is underspecified in two ways that change output.** §5.6 says it
"down-ranks species that do not occur near the photo's EXIF GPS point", but (a) §5.2 makes GPS
optional and nothing says what happens when it is absent — skip the geo step, or fail? — and (b)
"down-rank" has no numeric definition, so the re-ranked top-1 (and therefore the filed label) is
implementation-defined.

*Fix.* Specify in §5.6: "with no GPS the provider canonicalizes names only, records
`provider_status='no_gps'`, and never changes ranking. With GPS: query
`/v2/data/obs/geo/recent?lat&lng&dist=50&back=30` once, cache it, and multiply the score of any
candidate whose species code is absent from the response by **0.25**; re-sort, then re-apply
`min_species_confidence` (so a demoted top-1 can become `unknown`). Names are canonicalized against
the eBird taxonomy for every candidate regardless of geo." Add to E23 an assertion that a `no_gps`
fixture yields `provider_status='no_gps'` and byte-identical labels to the `own_bird_head` run.

**14. MEDIUM — `eval --calibrate` mutates the artifact in place, breaking the traceability
guarantee.** §7.4 "writes it [the temperature] into the artifact", while §7.1 promises `model_id` is
written into every `images` row "so any historical label can be traced to the exact model that
produced it". After calibration, two behaviourally different models share one `model_id`, and every
pre-calibration `images` row now points at a model that no longer exists.

*Fix.* Make calibration produce a new artifact: "`eval --calibrate --out <new.acmodel>` writes a
copy with `temperature` set and `model_id = f'{old_model_id}+cal{n}'`, recording
`train.calibrated_from = old_model_id`. Writing calibration into an existing artifact path is
refused (fatal, exit 3) — artifacts are immutable once `model_id` has been used."

**15. MEDIUM — `--ignore-overrides` leaves an ambiguous state that can flip labels on a later
run.** §5.9 says `--reclassify` keeps `label_source='human'` labels "unless `--ignore-overrides` is
also given"; E20 asserts the model label is restored. It is not stated whether the `overrides` rows
survive. If they do, the *next* plain `--reclassify` re-applies the human label and moves the file
back; if they are deleted, the user's corrections are destroyed by a flag whose name only says
"ignore".

*Fix.* State in §5.9: "`--ignore-overrides` never deletes `overrides` rows; it suppresses them for
the current run and sets `images.label_source='model'`. A subsequent run *without* the flag
re-applies the newest override for that sha256 and moves the file back — this is intended and
idempotent. Discarding a correction permanently requires the GUI (re-tag to the model's label),
never a CLI flag." Extend E20 with a third step: `classify --reclassify` again → the human label and
the human destination return.

**16. MEDIUM — §5.8 and §8 contradict each other about who reconciles a crashed re-tag.** §5.8
accepts a duplicate across two label dirs because "the next `classify` or `verify` run reconciles"
it; §8 describes `verify` as a command that "checks and reports" and offers no `--fix`, and
`classify` skips `status='done'` hashes entirely. As written, nothing ever reconciles, and the
duplicate is permanent.

*Fix.* Pick one and say it. Recommended: add `verify --fix` — "for each image whose row `label`
disagrees with a file found under another label directory, and where an `overrides` row justifies the
current `label`, delete the stale copy under the old label (never anything under `source`) and log
it; without `--fix`, report only, exit non-zero." Then add to E24 a step that simulates the
interrupted re-tag (two copies present) and asserts `verify` exit non-zero, `verify --fix` exit 0,
one copy remaining under the new label, and the source tree untouched.

**17. MEDIUM — `--limit` semantics against a resumed run are ambiguous.** §8 lists `--limit INT`
and §9 says "`--limit` and resume make that tolerable", but it is not stated whether the limit caps
*candidates scanned* or *images newly processed*. Under the first reading, a user running
`--limit 100` twice does 100 images then 0 new ones (all already `done`), which is the opposite of
what §9 promises.

*Fix.* State in §8: "`--limit N` caps the number of images **newly submitted to inference** in this
run; hashes already at `status='done'` are skipped without consuming budget, so repeated
`classify --limit 100` invocations walk the card 100 new images at a time." Assert it in E13:
two `--limit 2` runs over a 5-image fixture produce 4 `done` rows.

### NIT

**18. NIT — `taxon_rank` from PLAN's classifier contract was dropped.** `PLAN.md` §5 specifies
`{common_name, scientific_name, taxon_rank, confidence}`; the `.acmodel` label entries in §7.1 carry
`class` but no rank, so a coarse label (`bird`, `cow`) is indistinguishable in the catalog and GUI
from a real species identification (`Equus quagga`). *Fix:* add `"rank": "species"|"class"|"order"`
to each label entry and a `species_rank` column on `boxes`; show it in the GUI detail caption.

**19. NIT — Internal files under `output_root` are not excluded from label enumeration.**
`.catalog.db`, `.thumbs/`, `.ebird-cache.json` and stale `.ac-tmp-*` all live under `output_root`, so
`GET /api/labels` and `verify`'s orphan scan must skip any entry starting with `.`. *Fix:* state
that rule once in §5.8 ("label directories match `^[a-z0-9][a-z0-9_-]{0,63}$`; every other entry
under `output_root` is tool-internal and ignored by the GUI and by `verify`").

**20. NIT — `sources` needs an explicit upsert rule.** `sources(path TEXT PRIMARY KEY)` with a
`sha256` FK: if the user edits or replaces a file in place, the same path arrives with a new hash.
*Fix:* "`INSERT … ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, mtime_ns=excluded.mtime_ns`;
the previous `images` row is retained (its destination file stays) since content, not path, is the
identity."

**21. NIT — The blur metric upscales small images.** §5.2 resizes so the long edge "is exactly
512 px"; for a 300 px thumbnail this interpolates upward and depresses Laplacian variance, biasing
small sharp images toward `junk`. *Fix:* "resize only when the long edge exceeds 512 px; images
smaller than that are measured as-is, and `blur_score` records `blur_ref_edge` alongside it."

**22. NIT — Collision hashing through a symlink can fail.** §5.8's `already_present` check hashes
the existing destination's content; in `--link` mode that dereferences a symlink whose target may be
gone (card unplugged, source deleted). *Fix:* "in `link` mode, compare `os.readlink(dest)` to the
intended absolute source before hashing; a dangling symlink at the destination is replaced (the same
atomic temp+`os.replace` path) and counted as `relinked`."

**23. NIT — §2 makes `pi-heif` a core dependency while §10.1 keeps a fatal "`pi_heif` missing"
row.** With HEIC in the default format set the branch is unreachable. *Fix:* delete the row, or
(preferred, and consistent with finding 8) keep `pillow-heif` core and drop the branch.

**24. NIT — "~2,700 instances" is 2,666 after the design's own `iscrowd` filter.** Verified from
`instances_val2017.json`: 2,700 animal annotations total, 34 with `iscrowd=1`, **2,666** usable, over
1,016 images (per class: bird 427, sheep 354, cow 372, horse 272, zebra 266, elephant 252,
giraffe 232, dog 218, cat 202, bear 71). *Fix:* quote 2,666 in §1 and §7.2 and note that `bear` (71,
≈14 in val) is the class that limits per-class recall claims.

**25. NIT — The person-only → `landscape` consequence deserves user-facing documentation.** §5.4's
reasoning is sound and follows the closed taxonomy, but a card full of family photos filed under
`landscape/` will surprise the user. *Fix:* one line in the README and in `classify`'s help text:
"images containing only people or vehicles are filed as `landscape` (or `junk` if blurry); their
person/vehicle boxes are kept in the catalog and shown in the GUI."

---

## Verified assumptions

Each of these was a claim in `DESIGN.md` that I re-measured rather than trusted.

| Claim | How verified | Result |
|---|---|---|
| MegaDetector weights present with the pinned identity (§5.4) | `sha256sum models/md_v5a.0.0.pt` | `94e88fe97c8050f2e3d0cc4cb4f64729d639d74312dcbe2f74f8eecd3b01b276`, 280,766,885 B — matches the RECON constant the design fails fatally against ✅ |
| Backbone weights present and self-consistent (§7.3, E7) | `sha256sum models/backbones/efficientnet_b0_ra-3dd342df.pth` | `3dd342df…` matches the digest embedded in the filename ✅ |
| yolov5's helpers are importable with the signatures §5.4 uses | imported `letterbox`, `non_max_suppression`, `scale_boxes` in `.venv` | `letterbox(im, new_shape, color, auto, scaleFill, scaleup, stride)` (accepts `stride=64`, `auto=False`), `non_max_suppression(prediction, conf_thres, iou_thres, …, max_det, nm)`, `scale_boxes(img1_shape, boxes, img0_shape, ratio_pad)` ✅. Note the real module paths: `yolov5.utils.augmentations` for `letterbox`, `yolov5.utils.general` for the other two |
| The `models`/`utils` alias shim is required and sufficient | same import with `sys.modules.setdefault('models', …)` | works; and the load emitted the expected `pkg_resources is deprecated … pin to Setuptools<81` warning, confirming §2's `setuptools<81` note ✅ |
| The `uv` override really removes `opencv-python` (§2.1) | `uv pip compile … --override "opencv-python; python_version < '3.0'" --python-version 3.12` | resolved set contains `opencv-python-headless` and **no** `opencv-python` line ✅ (versions differ — finding 1) |
| fastapi/torch/torchvision/timm versions in §2 | same resolution | `fastapi==0.141.1`, `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29` ✅ |
| COCO animal category ids 16–25 (§7.2) | parsed `instances_val2017.json` | `[bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe]` ✅ |
| COCO gives real dominance ground truth (§1, E4) | recomputed box areas | 1,016 animal images, 471 multi-animal, 235 with `ratio >= 1.6` vs 236 below ✅ (the corpus exists; the *assertion* still needs finding 5) |
| CUB ships boxes and the official split (§7.2) | `tar -tzf data/raw/CUB_200_2011.tgz` | `bounding_boxes.txt`, `train_test_split.txt`, `classes.txt`, `images.txt`, `images/<class>/…` all present ✅ |
| CUB has no scientific names, so §13.3's nullable `scientific` is honest | `classes.txt` naming (`001.Black_footed_Albatross`) | common names only ✅ |
| HEIC decode path (§5.2) | `pi_heif 1.4.0` + `register_heif_opener()` | decode support present ✅ (encode is not — finding 8) |
| No area floor anywhere in the design | `rg 'min_box_area\|min_area\|area_floor\|minimum box'` over `DESIGN.md` | only the three passages that **forbid** it ✅ |
| e2e-only testing | §2, §11, `pyproject.toml` `testpaths` | `tests/e2e` only, no `tests/unit` ✅ |

## Unverified or wrong assumptions

| Claim | Status | Evidence |
|---|---|---|
| "`typer` **0.25.x** … Resolution pins it: `yolov5` → `roboflow` → `typer<0.26`. Verified by `uv lock` (§2.1)" | **WRONG as stated** | A py3.12 resolution picks `roboflow==1.3.8` (no typer cap) → `typer==0.27.2`; the cap exists only in `roboflow` 1.4.2. The installed `.venv` also has typer 0.27.2 *with* yolov5 present. Finding 1 |
| "131 packages resolved … `opencv-python-headless` 5.0.0.93 … `typer` 0.25.1" (§2.1) | **NOT REPRODUCIBLE / partly wrong** | Repo `uv.lock` has 9 packages and only `typer`; a real resolution gives 115 packages and `opencv-python-headless==4.10.0.84`. Finding 1 |
| "`pi_heif.register_heif_opener()` verified working" (§2) as sufficient for the format policy | **INCOMPLETE** | True for decode; `im.save(..., format='HEIF')` raises `KeyError: 'HEIF'` under `pi_heif` 1.4.0, so E16's fixture cannot be built. `pillow-heif` 1.1.1 encodes and re-reads correctly. Finding 8 |
| "This yields the ~2,700 instances counted in RECON" (§7.2) | **OFF BY THE `iscrowd` FILTER** | 2,666 after dropping the 34 `iscrowd=1` annotations the same sentence drops. Finding 24 |
| "the next `classify` or `verify` run reconciles" a crashed re-tag (§5.8) | **CONTRADICTED** by §8, where `verify` only reports and has no `--fix` | Finding 16 |
| "`ema` weights preferred … standard for yolov5 checkpoints" (§5.4) | **UNVERIFIED HERE, LOW RISK** | RECON confirms the checkpoint has both `ema` and `model` keys and that `model.names == ['animal','person','vehicle']`; I did not compare `ema` vs `model` outputs. Left as a NIT-free note because RECON's inference probe used the same construction and produced a valid `(1, 25500, 8)` tensor |
| GUI serving from "`source_root` … for a `--dry-run` catalog" (§6) | **UNSPECIFIED INPUT** | no schema column and no CLI flag supplies it. Finding 10 |
| `formats` TOML key behaviour (§3) | **UNSPECIFIED** | never referenced again in §5.1 or §10.2. Finding 9 |
| E7's "margin asserted numerically", E10's "above chance" (§11) | **UNSPECIFIED** | no numbers anywhere. Finding 7 |
| `train.val_top1 = 0.91` in the §7.1 artifact example | **ILLUSTRATIVE ONLY** — read as a sample, not a target; E7's real threshold is finding 7 | — |

## What is good and should not be churned

The dominance rule as a pure function with a multiplication instead of a division, `>=` at the
boundary, and the explicit "degenerate boxes still count, because excluding them would be an area
floor by the back door" reasoning — that is exactly right, and E5/E6 pin it. The three-mechanism
read-only guard with E1's byte-level snapshot is the correct shape for the one irreversible risk.
The `already_present` content-hash check giving cheap re-runs, the write order in I4
(`row → materializing → file → done`), the "duplicate rather than lose a file" choice in re-tag, and
the fail-loud table in §10.1 are all sound. Keep them as they are.
