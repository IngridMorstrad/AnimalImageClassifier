# Implementation plan — safari image classifier

Spec of record: `docs/DESIGN.md` (iteration 3, **frozen** — do not edit it).
Environment facts: `docs/RECON.md` (ground truth, do not re-litigate).
Outstanding review findings folded in below: the 11 MEDIUMs and 8 NITs of `docs/design-review.md`.

This file is the ordered work queue. It is not a design document and it does not re-decide anything
DESIGN.md already decided. Chunks are sized so that **one chunk = one coder iteration**, and every
chunk leaves the repo buildable with the suite green.

## Standing rules for every chunk (non-negotiable)

- **Absolute paths only.** The repo is `/projects/sandbox/AnimalImageClassifier`. Git is always
  `git -C /projects/sandbox/AnimalImageClassifier ...`. Branch is `feat/safari-classifier`; never
  commit to `main`; never open a PR.
- **E2E TESTS ONLY**, all under `/projects/sandbox/AnimalImageClassifier/tests/e2e/`. No unit tests,
  no `tests/unit`, no test that imports a private helper. Every test drives the installed
  `animal-classifier` console script via `subprocess`, or the real ASGI app over HTTP.
- **Never claim a test passes without pasting the real captured output** of
  `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest tests/e2e/<file> -q`.
- **`uv` only** (`uv venv`, `uv pip`, `uv run --frozen`, `uv sync --frozen`). Never `pip`/`venv`.
  Search with `rg`.
- **No `min_box_area`**, no `min_box_area_frac`, no `min_animal_area`, no absolute box-area floor
  under any name, in code, config, docs, CLI help or tests. `dominance_ratio` (default **1.6**) is
  the only size gate, and the comparison is a multiplication (`a0 >= ratio * a1`), never a division.
- **The source tree is read-only.** Only `materialize.py` writes, only under `output_root`, only via
  temp-file + `os.replace`. On an existing destination the only permitted calls are `os.replace` and
  `os.unlink` (I1 — a `--hardlink` destination *is* the card's inode).
- **Never silently default a required value.** Fail loudly (exit 3) or skip-with-warning exactly as
  `docs/DESIGN.md` §10.1 prescribes. No `dict.get(key, fallback)` for a required key, no bare
  `except`.
- **Every chunk ends with**: `date -u '+%Y-%m-%d %H:%M UTC'`, an *appended* entry to
  `/projects/sandbox/AnimalImageClassifier/docs/PROGRESS.md` (what was done, the real verification
  output, what is blocked), then `git -C /projects/sandbox/AnimalImageClassifier add -A`, commit, and
  `git -C /projects/sandbox/AnimalImageClassifier push origin feat/safari-classifier`.
- After each chunk, update `/projects/sandbox/AnimalImageClassifier/docs/impl-status.json`
  (`done_items`, `current_chunk`; `complete: true` only when every item below is checked).
- Anything that is genuinely a polish/style idea and not required by DESIGN.md goes to
  `/projects/sandbox/AnimalImageClassifier/docs/FOLLOWUPS.md` — it never blocks a chunk.

## Phase A — the working end-to-end slice (chunks 1–11)

Goal of Phase A: `animal-classifier classify` walks a real fixture card, detects, decides by
dominance, materializes by copy/link/hardlink, and records everything in SQLite — green on
committed fixtures, with the scripted detector and a pass-through classifier. No model training,
no GUI yet.

- [ ] 1. **Dependency contract and the three probes on the locked env.** Rewrite
      `pyproject.toml` exactly as DESIGN.md §2.1 (`requires-python = ">=3.12,<3.14"`, the pinned
      dependency list, `[project.optional-dependencies] raw`/`dev`, and the `[tool.uv]
      override-dependencies` that drop `opencv-python`, `roboflow` and `sahi`); add
      `.python-version` containing `3.12`; keep `[tool.pytest.ini_options] testpaths = ["tests/e2e"]`
      and add `-ra` plus a `slow` marker. Regenerate `uv.lock`. Then re-run the three RECON probes
      against the synced venv — nothing else starts until they pass.
      Files: `/projects/sandbox/AnimalImageClassifier/pyproject.toml`,
      `/projects/sandbox/AnimalImageClassifier/.python-version`,
      `/projects/sandbox/AnimalImageClassifier/uv.lock`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv lock && uv sync --frozen --extra dev &&
      uv run --frozen python scripts/probe_md_checkpoint.py && uv run --frozen python
      scripts/probe_md_inference.py && uv run --frozen python scripts/probe_backbones.py` — the
      checkpoint loads with `names=['animal','person','vehicle']`, a forward pass returns
      `(1, 25500, 8)`, and both backbones load with 0 missing / 0 unexpected keys. Paste all three
      outputs into PROGRESS.md. If any pin moved, record the new version in
      `/projects/sandbox/AnimalImageClassifier/docs/RECON.md` under a dated "re-verified" note
      (append only — do not rewrite RECON's measurements).

- [ ] 2. **`errors.py` + `config.py`: total validation, layered resolution, fail-loud.**
      `errors.py` defines `ConfigError`, `AssetError`, `DecodeError`, `MaterializeError`,
      `CatalogError` with the exit codes of DESIGN.md §10.1 (`0/1/2/3/4`). `config.py` exposes a
      frozen `Config` dataclass and `Config.resolve(...)` implementing CLI → env
      (`ANIMAL_CLASSIFIER_*`) → TOML → default layering, every key and range in §3 and §10.2, the
      `formats` family policy of §3.1 (`raw` in `formats` is fatal), `--device` resolution (explicit
      `cuda` on this CUDA-less host is fatal exit 3), the source/output nesting guard (both
      directions plus identical), a **fatal unknown-TOML-key** check, and a fatal missing `--config`.
      Relative `species_model`/`bird_model`/`detector_weights` paths resolve against the **current
      working directory** and every error message prints the resolved absolute path
      (review finding 11a). There is deliberately no `min_box_area` key, and the unknown-key check is
      what stops one being reintroduced by configuration.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/errors.py`,
      `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/config.py`
      Verify: chunk 3's `catalog` work does not depend on this, so verification is chunk 12's E22;
      for now `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen python -c "import
      animal_classifier.config"` imports clean and `uv run --frozen pytest tests/e2e -q` still
      collects 0 failures.

- [ ] 3. **`catalog.py`: schema, indexes, upserts, statuses — including reproduced DEFECT 1.**
      Create the seven tables and four indexes of §5.9 verbatim (`PRIMARY KEY(box_id, rank)` on
      `candidates`, unique `(sha256, idx)` on `boxes`, `runs.source_root NOT NULL`), `PRAGMA
      journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout=10000`, a `meta` schema version that
      refuses to run against a newer version, `run_id = f"{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"`,
      and "newest run" defined as `ORDER BY started_at DESC, run_id DESC LIMIT 1`.
      **DEFECT 1 (reproduced against real data — `IntegrityError` on any second run):** every
      `skipped` insert must be
      `INSERT INTO skipped(path, reason, detail, run_id, seen_at) VALUES(?,?,?,?,?) ON CONFLICT(path)
      DO UPDATE SET reason=excluded.reason, detail=excluded.detail, run_id=excluded.run_id,
      seen_at=excluded.seen_at;` — there must be no bare `INSERT INTO skipped` anywhere. The
      `sources` upsert of §5.9 is the same shape on `path`. Publish the §5.9 re-inference
      transaction as one function (`BEGIN IMMEDIATE` → delete candidates → delete boxes → insert →
      `UPDATE images` → `COMMIT`) that never touches `overrides` (append-only) or `first_seen`.
      Implement the **full re-processing policy** (review finding 10): `done` → skipped unless
      `--reclassify`; `planned`/`materializing` → re-processed; `failed` → **retried**, and the retry
      consumes `--limit` budget; paths in `skipped` → always re-walked and re-evaluated against the
      current format policy.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/catalog.py`
      Verify: proven end-to-end by chunk 9 (E2) and chunk 10 (E13's second-run and retry legs); this
      chunk's own gate is `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e -q` still green and `rg -n "INSERT INTO skipped" src/` showing only the upsert form.

- [ ] 4. **`taxonomy/`: `slug()`, `LABEL_RE`, `RESERVED_LABELS` and the static name tables — DEFECT 2's
      normalization layer.** `labels.py` publishes `LABEL_RE = ^[a-z0-9][a-z0-9_-]{0,63}$`,
      `RESERVED_LABELS = {multiple, landscape, junk, unknown}` and `slug()` exactly as §5.8 (NFKD →
      ASCII fold → lowercase → collapse `[^a-z0-9]+` → strip → truncate 64 → **raise `ConfigError`**;
      never a sanitized guess), plus the static table loader, common ↔ scientific lookup, `rank`, and
      the class rollup used by the bird trigger (`Aves`).
      **DEFECT 2 (reproduced — raw dataset class names leak into output paths):** dataset directory
      names are never labels. `022.Chuck_will_Widow` → strip `^\d{3}\.` → `key = "chuck_will_widow"`
      (this is also the `cub_key` spelling used by `ebird_aliases.csv`), and the **display** `common`
      and `scientific` come from `taxonomy/data/cub200.csv`, which is authoritative
      (`scientific` empty where no reliable offline source exists — honest NULL, never invented).
      Hand-author `coco_animals.csv` for the 10 COCO classes with explicit `rank` (`bird` is
      `rank=class`, not a species) and `cub200.csv` for the 200 CUB classes (generate the key column
      from the archive's `classes.txt`, then fill display names).
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/taxonomy/__init__.py`,
      `.../taxonomy/labels.py`, `.../taxonomy/data/coco_animals.csv`,
      `.../taxonomy/data/cub200.csv`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest tests/e2e -q`
      green; the real assertion lands in chunk 18's E8 label-directory check
      (`^[a-z][a-z0-9_]*$`, no leading digits, never `022_chuck_will_widow`).

- [ ] 5. **`scan.py`: read-only walk, one rule per skip reason, benign vs abnormal exit class.**
      `os.walk(source, followlinks=False)`, sorted for determinism, yielding `Candidate(path, size,
      mtime)` / `Skipped(path, reason)` for every reason in §5.1 with exactly the rule given there
      (`hidden`, `system_dir`, `unsupported_extension`, `format_disabled`, `video`,
      `raw_not_enabled`, `zero_bytes`, `unreadable`, `too_large`, `symlink`, `symlink_escape`). No
      `symlink_loop`. `max_file_bytes` is a whole-file `st_size` cap that structurally never sees a
      box.
      **Review finding 5 (exit-code contradiction):** classify skips into two sets and record which
      one fired — exit **0** when only *benign* skips occurred (`hidden`, `system_dir`,
      `unsupported_extension`, `format_disabled`, `video`, `raw_not_enabled`, `too_large`,
      `symlink`); exit **4** when any *abnormal* outcome occurred (`zero_bytes`, `unreadable`,
      `symlink_escape`, `too_large_pixels`, `decode_error`, or any `failed` image).
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/scan.py`
      Verify: covered by chunk 12's E16 and chunk 9's E2 (E2's card carries benign skips only and
      must exit 0); interim gate is a clean import plus a green `uv run --frozen pytest tests/e2e -q`.

- [ ] 6. **`images.py`: decode, the one coordinate frame, sha256, blur, crop.** `open_source(path)`
      opens `"rb"` and is the only source reader. `PIL.Image.open` → `ImageOps.exif_transpose` →
      `convert("RGB")`; `pillow_heif.register_heif_opener()` once at import;
      `Image.MAX_IMAGE_PIXELS` raised to 400 MP with anything larger a `DecodeError` →
      `skipped(too_large_pixels)`. **I9:** `width`/`height` and every persisted coordinate and
      `area_frac` are in the EXIF-transposed frame; raw stored dimensions exist nowhere else.
      sha256 streams the **raw file bytes** in 1 MiB chunks. EXIF `DateTimeOriginal` → ISO-8601 and
      GPS → signed decimal degrees are both optional (absent → NULL, never fabricated; out-of-range
      GPS → NULL + WARNING). Blur = variance of Laplacian on grayscale, **downscaled only** when the
      long edge exceeds 512 px, storing `blur_ref_edge` beside `blur_score`; document in the
      docstring that `blur_threshold` is calibrated at the 512 px reference and smaller images are
      scored natively (NIT 12). `crop()` expands by the **artifact's** `crop_margin` and clips;
      a box under 2 px on a side after clipping is `species_status="degenerate"` — excluded from
      classification but **still an animal for dominance** (a floor by the back door is forbidden).
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/images.py`
      Verify: covered by chunk 12 (E11 blur legs) and chunk 20 (E18 orientation-6 leg); interim gate
      is a clean import and a green `uv run --frozen pytest tests/e2e -q`.

- [ ] 7. **`detect/base.py` + `detect/scripted.py`: the detector protocol and the shipped test
      detector.** `base.py` defines the `Detector` protocol and the frozen `Box(cls, conf, x0, y0,
      x1, y1, area_frac)` dataclass in the transposed frame. `scripted.py` reads a
      `<image>.boxes.json` sidecar and returns exactly those boxes — a *shipped* implementation
      selected by the real flag `--detector scripted`, so dominance geometry can be stated exactly.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/detect/__init__.py`,
      `.../detect/base.py`, `.../detect/scripted.py`
      Verify: exercised by chunk 9's E5/E6/E26; interim gate is a clean import and green collection.

- [ ] 8. **`decide.py` + `materialize.py`: the dominance rule and atomic filing.** `decide.py` is the
      §5.7 function verbatim: `species_or_unknown` total (degenerate or `None` conf or below
      `min_species_confidence` → `unknown`), no animals → `junk` if `blur_score < blur_threshold`
      else `landscape`, one animal → species-or-unknown, ≥ 2 animals → `animals[0].area_frac >=
      dominance_ratio * animals[1].area_frac` (multiplication, `>=`) else `multiple`. Nothing else in
      the file compares sizes. `materialize.py` implements copy (default) / `--link` / `--hardlink`,
      each as temp-in-destination-dir + `os.replace`, `shutil.copystat` for copy,
      `finally`-unlink of the temp on failure, a sweep of stale `.ac-tmp-*` at run start, the §5.8
      collision rules (content sha for regular files; `os.readlink` comparison in link mode, never
      dereferencing a dangling link; `<stem>-<sha256[:8]><suffix>` on differing content; fatal if
      that is taken too), the `EXDEV` actionable fatal for `--hardlink`, `--dry-run` writing
      `status='planned'` with a planned `dest_path` and no destination writes, and a refusal of any
      path not under `output_root`. **No code path opens a materialized file for writing or truncates
      it** — on an existing destination only `os.replace`/`os.unlink`.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/decide.py`,
      `.../materialize.py`
      Verify: covered by chunk 9 (E2/E5/E6/E26), chunk 11 (E1/E3/E14/E15); interim gate is green
      collection.

- [ ] 9. **Fixture builder + `classify` wired end-to-end → the first green e2e run.** Write
      `scripts/make_e2e_fixtures.py`, which builds the fixture SD-card tree from the archives in
      `/projects/sandbox/AnimalImageClassifier/data/raw` (session-scoped cache under
      `tests/e2e/_fixtures/`, gitignored) and **fails with the exact download commands** if
      `data/raw` is missing — a silently skipped suite is worse than a red one. It writes the mixed
      card (JPEG/PNG/TIFF/HEIC via `pillow-heif` encoding — fail loudly if HEIF encode is
      unavailable, never substitute AVIF), the `.boxes.json` sidecars for the scripted-detector
      cases, the animal-free sharp photo, the blurred photo, the small (≤512 px) sharp photo and the
      orientation-6 photo. Then replace the placeholder `classify` in `cli.py` with the real
      orchestration: scan → decode/sha256/blur in a `ThreadPoolExecutor(jobs)` → detect → crop →
      (pass-through classifier for now) → decide → materialize → one catalog transaction per image,
      `torch.set_num_threads(jobs)`, bounded queue depth `2 * jobs`, progress written to the `runs`
      row, and the full flag surface of §8 including the hand-rolled mutually-exclusive
      `--link`/`--hardlink` (`typer.BadParameter` → exit 2) and the repeatable `--formats` StrEnum.
      Add the person/vehicle → `landscape` note to `classify --help`.
      Files: `/projects/sandbox/AnimalImageClassifier/scripts/make_e2e_fixtures.py`,
      `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/cli.py`,
      `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/pipeline.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/conftest.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e02_happy_path.py`,
      `.../tests/e2e/test_e05_dominance_boundaries.py`, `.../tests/e2e/test_e06_no_area_floor.py`,
      `.../tests/e2e/test_e26_degenerate_dominant_box.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e02_happy_path.py tests/e2e/test_e05_dominance_boundaries.py
      tests/e2e/test_e06_no_area_floor.py tests/e2e/test_e26_degenerate_dominant_box.py -q` — E2
      exact output tree + `images`/`boxes`/`sources`/`runs` rows + exit 0; E5 ratio 1.6 → dominant,
      1.59 → `multiple`, equal → `multiple`, zero-area second box → dominant with no exception; E6 a
      lone 0.05 %-of-frame animal filed as its species, the 0.1 % vs 0.5 % pair resolved by ratio,
      and `classify --help` exposing no `min_box_area`-like option; E26 label `unknown`,
      `species_status='degenerate'`, file under `unknown/`, exit 0.

- [ ] 10. **Idempotency, resume, `--limit`, `--reclassify`, `failed` retry — DEFECT 1's regression
      test.** Implement/verify the §5.9 bookkeeping in the pipeline and write E13 as one file with
      all legs: `classify` twice over the same fixture tree (**the second run must succeed and be a
      clean no-op** — this is DEFECT 1's regression gate, with a card containing an `.mp4` so the
      `skipped` upsert is exercised, exactly one `skipped` row for that path carrying the *second*
      run's `run_id`); a killed run resuming to completion; two `--limit 2` runs over a 5-image card
      leaving **4** `done` rows; two `--reclassify` runs leaving `COUNT(*)` over `boxes` and
      `candidates` unchanged with `overrides` untouched; two `--reclassify --limit 2` runs touching
      **4 distinct hashes** in `last_updated ASC` order; a `failed` row **retried** on the next run;
      and a `.cr2` skipped without `--raw` ingested by a later `--raw` run (review finding 10).
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/pipeline.py`,
      `.../src/animal_classifier/catalog.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e13_idempotency_limit_reclassify.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e13_idempotency_limit_reclassify.py -q` — all legs pass, and paste the captured
      output showing the second run's exit code and its `already_present` count.

- [ ] 11. **Source immutability, the three modes, duplicates and collisions.** E1 snapshots
      `(relative path, size, mtime_ns, sha256)` over the whole fixture card and asserts byte-identical
      equality after `classify`, a re-tag, `export-trainset` and `verify --fix` — run in **copy** and
      **`--hardlink`** legs (hardlink proves I1 under an aliased inode: source `mtime_ns`/`sha256`
      unchanged and `st_ino` still shared), plus a leg with the card `chmod 0o500` and a leg with the
      card renamed away. Re-tag/`verify --fix`/`export-trainset` legs are marked `xfail(strict=False)`
      **only** until chunks 21/23/24 land, then unmarked. E3 asserts `--link` symlink targets and
      `--hardlink` `st_nlink == 2` + equal `st_ino`. E14 asserts one destination file and two
      `sources` rows for duplicate content, **and** the content-changed-in-place upsert (NIT 17): one
      `sources` row, two `images` rows, both destinations intact. E15 asserts the
      `<stem>-<sha8><suffix>` collision name with both files intact.
      Files: `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e01_source_immutable.py`,
      `.../tests/e2e/test_e03_link_modes.py`, `.../tests/e2e/test_e14_duplicate_content.py`,
      `.../tests/e2e/test_e15_collision_different_content.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e01_source_immutable.py tests/e2e/test_e03_link_modes.py
      tests/e2e/test_e14_duplicate_content.py tests/e2e/test_e15_collision_different_content.py -q`
      — all green (the not-yet-implemented E1 legs reported as xfail, not as passes).

## Phase B — formats, skips, states and fail-loud (chunks 12–13)

- [ ] 12. **Formats, skips, byte cap, symlink policy, landscape/junk, dry-run.** E16 drives a card
      with JPEG/PNG/TIFF/HEIC, an `.mp4`, a `.cr2`, a 0-byte file, a truncated JPEG, a sparse file
      just over `--max-file-bytes`, a symlink to a JPEG inside the card and one to a JPEG outside
      it, plus a `--formats jpeg` run, a `--follow-source-symlinks` run and a `--formats raw` run
      (exit 3 listing the valid set). For the over-cap file assert the **observable** consequences
      (NIT 15): `skipped(too_large)`, no `images`/`sources` row, no `DecodeError` in stderr — never
      "its bytes were not read". Exit code follows chunk 5's benign/abnormal split: this card exits 4
      because of the 0-byte and truncated files. E11 asserts `landscape` vs `junk`, `blur_score` and
      `blur_ref_edge` stored for both, and the small sharp animal-free photo as `landscape` (no
      upscaling bias). E17 asserts `--dry-run` writes no image files, rows are `planned`, exit 0.
      Files: `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e16_formats_and_skips.py`,
      `.../tests/e2e/test_e11_landscape_junk.py`, `.../tests/e2e/test_e17_dry_run.py`, and whatever
      `scan.py`/`images.py`/`pipeline.py` fixes the tests demand
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e16_formats_and_skips.py tests/e2e/test_e11_landscape_junk.py
      tests/e2e/test_e17_dry_run.py -q` green (E17's GUI 409 legs arrive in chunk 20).

- [ ] 13. **Fail-loud config end to end (E22).** One test, one case per row: missing detector
      weights; unknown TOML key; `dominance_ratio=0.5`; `hosted_bird_api` selected;
      `ebird_enrich` without `ANIMAL_CLASSIFIER_EBIRD_API_KEY`; `output` nested inside `source`;
      `--formats raw`; `--device cuda` on this CUDA-less host; `eval --calibrate` without `--out`;
      **and a missing `species_model`, whose message must contain the exact
      `animal-classifier train ...` invocation that produces it** (review finding 11b). Each case
      asserts exit **3**, the offending name in stderr, and **no destination directory created**.
      Files: `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e22_fail_loud_config.py`, plus
      `config.py`/`cli.py` fixes
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e22_fail_loud_config.py -q` — every case exits 3 with the named value in stderr.

## Phase C — the real detector (chunk 14)

- [ ] 14. **MegaDetector v5a for real, and E4 against COCO ground truth.** `detect/megadetector.py`:
      the `sys.modules['models'/'utils'] = yolov5.*` alias shim, `torch.load(..., weights_only=False)`
      with a **size + sha256 gate** against `280766885` /
      `94e88fe97c8050f2e3d0cc4cb4f64729d639d74312dcbe2f74f8eecd3b01b276` (fatal with the download URL
      on mismatch), `ema` preferred over `model`, then `letterbox(new_shape=config.detector_image_size,
      stride=64, auto=False)` → `/255` → `model(x)[0]` → `non_max_suppression(conf_thres=
      config.detector_confidence, iou_thres=config.detector_iou, max_det=config.detector_max_det)` →
      `scale_boxes` onto the transposed size. **NIT 13:** no hard-coded 1280/0.20/0.45 — they come
      from config. **NIT 14:** add `detector_max_det = 100` to `config.py` with validation, and note
      in `decide.py`'s docstring that it is a detector cap, not a size gate. Only `cls == "animal"`
      boxes create labels; person/vehicle boxes are stored and drawn. Write
      `scripts/build_coco_dominance.py`, which emits `coco_dominance.json` (**all** qualifying
      val-bucket images at `threshold_high=3.0` / `threshold_low=1.3`) and `coco_species.json`
      (**all** qualifying single-animal val-bucket images — review finding 8, no literal 20), both
      sorted by `image_id`, both restricted to `split_for(file_name) == "val"`.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/detect/megadetector.py`,
      `.../src/animal_classifier/config.py`,
      `/projects/sandbox/AnimalImageClassifier/scripts/build_coco_dominance.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/data/coco_dominance.json`,
      `.../tests/e2e/data/coco_species.json`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e04_dominance_real_coco.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen python
      scripts/build_coco_dominance.py` then `uv run --frozen pytest
      tests/e2e/test_e04_dominance_real_coco.py -q` — every label in the closed set, `>= ceil(0.8 *
      len(high))` of `high` not `multiple`, `>= ceil(0.8 * len(low))` are `multiple`, thresholds and
      lengths read from the JSON with **no numeric literals in the test**, per-image results printed.
      If the aggregate fails, the only sanctioned remedy is re-freezing at `threshold_high = 4.0` and
      recording it in PROGRESS.md — never lower a ratio in test code, never pad with train-bucket
      images.

## Phase D — training and the species head (chunks 15–17)

- [ ] 15. **Training subsystem skeleton + the fast synthetic smoke path (E10).**
      `training/manifest.py` (JSONL read/write/validate, **fail on the first bad line** with line
      number and field, missing files fatal, plus `split_for()` keyed on the **basename** exactly as
      §7.2), `training/dataset.py` (`ManifestDataset`, crop-on-load at the artifact's `crop_margin`,
      `WeightedRandomSampler(1/sqrt(count))`), `training/transforms.py`, `training/tinycnn.py`,
      `training/synthetic.py` (deterministic seeded shapes), `training/export.py` and
      `classify/artifact.py`. The artifact dict is §7.1 **plus `crop_margin`** (review finding 7);
      `temperature` is always written as `1.0` with `calibrated_from = None`, and on load a
      **missing** `temperature` is a fatal `AssetError` (never `.get(k, 1.0)`). `artifact.load()`
      validates the **whole label space** up front — `slug()` every entry, reject
      `RESERVED_LABELS`, exit 3 naming the label and artifact — and caches the slugs. Wire real
      `train` and `eval` subcommands in `cli.py`.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/training/__init__.py`,
      `.../training/{manifest,dataset,transforms,trainer,evaluate,export,tinycnn,synthetic}.py`,
      `.../src/animal_classifier/classify/__init__.py`, `.../classify/base.py`,
      `.../classify/artifact.py`, `.../src/animal_classifier/cli.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e10_synthetic_smoke.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e10_synthetic_smoke.py -q` — `train --dataset synthetic --arch tinycnn` (4
      classes, 500 train / 100 val) → `eval` → `classify` completes the full train→eval→export→infer
      path with `val_top1 >= 0.90` in **< 90 s wall clock**; paste the timing.

- [ ] 16. **Real transfer learning on real COCO crops (E7).** `scripts/build_coco_manifest.py` reads
      `data/raw/annotations/instances_val2017.json`, takes categories 16–25, drops the 34 `iscrowd=1`
      instances, converts `bbox` to `x0,y0,x1,y1`, honours `--classes`, and **filters training
      manifests to `split == "train"`** (the §7.2 leakage rule). `training/trainer.py` implements the
      two-stage finetune (head-only `AdamW(3e-3)` with the backbone frozen and in `eval()` so BN
      stats hold, then unfreeze the last `--unfreeze-blocks` stages at `3e-4`/`3e-5`), cosine
      schedule, `CrossEntropyLoss(label_smoothing=0.1)`, `timm.create_model(arch, pretrained=False)`
      + explicit local `load_state_dict` (**never `pretrained=True`**; a missing local backbone is
      fatal exit 3), per-epoch checkpointing to `<out>.ckpt` with RNG states, and `--resume` that
      refuses a different manifest sha256 or arch. `training/evaluate.py` writes `metrics.json` +
      `confusion_matrix.csv` (top-1/top-5, macro and per-class recall, support) and implements
      `--calibrate --out <new>` as a **new artifact** (`model_id = "<old>+calN"`,
      `train.calibrated_from`), fatal if `--out` is an existing artifact path (I8). Default
      `--input-size` is **224**; E7 alone passes 128 as a stated CPU budget trade-off.
      Files: `/projects/sandbox/AnimalImageClassifier/scripts/build_coco_manifest.py`,
      `.../src/animal_classifier/training/trainer.py`, `.../training/evaluate.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e07_real_training_species.py`,
      `.../tests/e2e/test_e09_train_resume.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e07_real_training_species.py tests/e2e/test_e09_train_resume.py -q` — E7
      finetunes `efficientnet_b0` on the 7-class train split and asserts `val_top1 >= 0.55`,
      artifact `train.val_top1` equal to `metrics.json` within `1e-6`, `format_version` present, 7
      labels each with `rank`, `temperature == 1.0`, `calibrated_from is None`, **`crop_margin`
      recorded** and honoured over the run config (`classify --crop-margin 0.3` still crops at the
      artifact's 0.08 — finding 7), and wall clock **< 20 min**; per-class recall printed but never
      asserted for `bear` (11 val instances). E9 resumes at the right epoch and exits 3 on a
      mismatched manifest with both ids. Sanctioned remedy if `val_top1` misses: raise
      `--input-size` toward 224 and/or `--epochs-finetune`, recorded in PROGRESS.md — never lower the
      threshold.

- [ ] 17. **Species inference wired into `classify` (E7 identity leg + E12).**
      `classify/own_model.py` loads the artifact, batches an image's crops (≤ 8), forwards, applies
      `softmax(logits / artifact["temperature"])`, returns `Prediction`/`Candidate` exactly as §5.5
      with `rank_level` from the artifact, and gates on `min_species_confidence` → `unknown` while
      still recording the top-5. Replace the pass-through classifier in the pipeline and persist
      `candidates` rows. E7's identity leg asserts `>= ceil(0.7 * len(coco_species.json))` images
      receive their GT species label **through the real CLI**, with the length read from the JSON
      (review findings 8 and 9 — no `20`, no `14` literal; the identity subset is deliberately easier
      than the crop-level average because it is single-animal with `area_frac >= 0.20`, and the
      sanctioned remedy is raising `--input-size`/`--epochs-finetune`, never widening
      `min_species_confidence` for the test).
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/classify/own_model.py`,
      `.../src/animal_classifier/pipeline.py`,
      `.../tests/e2e/test_e07_real_training_species.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e12_unknown_low_confidence.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e12_unknown_low_confidence.py tests/e2e/test_e07_real_training_species.py -q` —
      E12: `classify --min-confidence 0.999` labels `unknown`, top-5 candidates still recorded, file
      still materialized; E7's identity leg green.

## Phase E — birds (chunks 18–19)

- [ ] 18. **The CUB bird head and the bird path (E8) — DEFECT 2's assertion.**
      `scripts/build_cub_manifest.py` extracts CUB-200-2011, emits one sample per image with the box
      from `bounding_boxes.txt` and the official `train_test_split.txt`, and derives labels through
      chunk 4's normalization: directory `022.Chuck_will_Widow` → `key = chuck_will_widow`, display
      names from `taxonomy/data/cub200.csv` — the directory name is **never** used as a label
      directory. `classify/birds/base.py` defines `BirdProvider`/`BirdResult`/`GpsPoint`;
      `classify/birds/own_head.py` is the default provider; the trigger fires when **any** coarse
      top-3 candidate rolls up to `Aves`, or the top-1 is below the gate and any top-3 is `Aves`;
      the merge rule is §5.6 exactly (`refined` replaces the box row + candidates and sets
      `model_id = "<species>+<bird>"`; every other status leaves the coarse row alone and writes only
      `bird_provider`/`provider_status`). `--force-bird-head` logs its one mandated WARNING and is
      recorded in `runs.config_json`. **Review finding 11c:** a missing `bird_model` is fatal only
      when the bird path is reachable — otherwise a startup WARNING that refinement is disabled, with
      `provider_status='no_bird_model'` recorded.
      Files: `/projects/sandbox/AnimalImageClassifier/scripts/build_cub_manifest.py`,
      `.../src/animal_classifier/classify/birds/{__init__,base,own_head}.py`,
      `.../src/animal_classifier/pipeline.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e08_bird_path.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e08_bird_path.py -q` — a 5-species CUB head trains, a CUB test-split image gets
      one of the 5 species, `bird_provider='own_bird_head'`, `provider_status='refined'`,
      `model_id == "<species_id>+<bird_id>"`, box `species_rank='species'`, the `--force-bird-head`
      run invokes the head without relying on coarse ranking, **and the created label directory
      matches `^[a-z][a-z0-9_]*$` with no leading digits** — no `~/animal_pics/022_chuck_will_widow/`
      anywhere under the output root (DEFECT 2's regression gate).

- [ ] 19. **`ebird_enrich`, the alias table, and the hosted stub (E23).**
      `taxonomy/data/ebird_aliases.csv` (`cub_key,ebird_com_name,ebird_sci_name`, hand-authored,
      deliberately partial) with the §5.6 load-time validation (unknown `cub_key` or both name
      columns empty → exit 3 with the CSV line number). `classify/birds/ebird_enrich.py` implements
      the re-ranker: no GPS → canonicalize aliased names only, ranking unchanged,
      `provider_status='no_gps'`, **no network call**; with GPS → one cached call to
      `.../obs/geo/recent?...&dist=50&back=30` (4 s connect / 8 s read, one attempt, no retry, cache
      at `<output_root>/.ebird-cache.json` keyed on `(round(lat,2), round(lon,2))`), sci-name-then-
      folded-common-name matching, aliased-but-unobserved candidates × **0.25**, **alias-less
      candidates exempt**, re-sort, re-apply the confidence gate; unreachable → exactly one WARNING
      per run and the own-head result unchanged with `provider_status='unreachable'`; 401/403 fatal.
      `classify/birds/hosted_api.py` raises `AssetError` at config time naming the env vars and
      endpoint it would need — it never silently no-ops. The client factory must accept an injected
      transport so E23's stub leg can drive it.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/classify/birds/ebird_enrich.py`,
      `.../classify/birds/hosted_api.py`,
      `.../src/animal_classifier/taxonomy/data/ebird_aliases.csv`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e23_ebird_enrich.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e23_ebird_enrich.py -q` — both live runs (host answers `000` here) produce the
      same labels as the own-head run with exactly one warning and `provider_status='unreachable'`;
      the no-GPS run makes no network call and records `no_gps`; the `httpx.MockTransport` leg shows
      the observed candidate's score untouched, the aliased-unobserved candidate at exactly `0.25 ×`
      its original, and the alias-less candidate **bit-identical**.

## Phase F — the GUI (chunks 20–22)

- [ ] 20. **GUI read paths (`gui/app.py`) — E18 and E17's GUI legs.** `create_app(config)` +
      `animal-classifier gui` on `127.0.0.1:8765` (no `--host`). All `GET` routes use read-only
      connections (`file:...?mode=ro`). `/api/labels` is **catalog-authoritative**
      (`SELECT label, COUNT(*) … WHERE status IN ('done','planned','materializing')` — NIT 16 adds
      `materializing` so a pending move cannot vanish from the sidebar) with `files_on_disk` as a
      visible cross-check that ignores tool-internal entries (`.catalog.db*`, `.thumbs/`,
      `.ebird-cache.json`, `.ac-tmp-*`). `/api/images` implements every filter of §6 plus:
      **review finding 3** — `q` is a case-insensitive substring match over
      `images.species_common`, `images.species_scientific` and the `sources.path` basename, max 128
      chars, 422 above that; **review finding 4** — `date_from`/`date_to` are ISO `YYYY-MM-DD`
      (inclusive of the whole `date_to` day, 422 on unparseable or `from > to`), NULL-dated rows
      match only when `include_undated=true` (default false), and **every** response carries both
      `unscored_excluded` and `undated_excluded`. `/thumb` is generated from `dest_path` only on the
      transposed image and cached under `.thumbs/`; `/thumb` and `/full` both return
      `409 {"error": ..., "reason": "planned"|"missing"|"dangling_symlink"}`, never 500. `/api/run`
      returns the newest run row including `source_root` for the header only.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/gui/__init__.py`,
      `.../gui/app.py`, `.../src/animal_classifier/cli.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e18_gui_browse.py`,
      `.../tests/e2e/test_e17_dry_run.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e18_gui_browse.py tests/e2e/test_e17_dry_run.py -q` — E18 (real uvicorn
      subprocess + `httpx`): `/` serves HTML, `/api/labels` returns matching `count` and
      `files_on_disk` excluding tool-internal entries, filters work, thumb/full bytes decode, boxes
      lie inside the stored dimensions, captions carry `species_rank`; the orientation-6 file has
      `images.width < images.height` with its box valid only in the transposed frame and a thumb
      whose dimensions match; a `min_conf` query reports the exact `unscored_excluded` and returns
      those rows under `include_unscored=true`; a date query reports `undated_excluded`; a `q` hit
      and a `q` miss; the dangling-symlink `/thumb` returns 409 `reason="dangling_symlink"`. E17's
      GUI legs: `/full` **and** `/thumb` return 409 `reason="planned"`, and `/api/labels` shows
      `count > 0` with `files_on_disk == 0`.

- [ ] 21. **GUI re-tag: rename inside the output tree, validation, contention (E19–E21).**
      `materialize.retag()` exactly as §5.8: resolve the new destination, **write the override row
      and the `materializing` status first** (I4), then one `os.replace`, then `status='done'`; the
      source is never opened. Collisions reuse §5.8's rules (identical content or identical
      `os.readlink` → unlink the old entry, count `already_present`). **Review finding 6:** a row
      with `status='planned'` is a `409 {"reason": "planned"}` *guard before anything is touched* —
      it must not be flipped to `failed`, and no override row is written; the `failed` marking is
      reserved for a row that was `done` and whose file has genuinely vanished. The `POST` uses a
      short-lived write connection with `busy_timeout = 250 ms` and `BEGIN IMMEDIATE`, returning
      `409 {"error": "a classify run is writing the catalog; retry"}` with nothing changed on disk.
      Label validation: `LABEL_RE`, never `RESERVED_LABELS` (422 even under `--allow-new-labels`),
      known unless `--allow-new-labels`. `--ignore-overrides` suppresses without deleting.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/materialize.py`,
      `.../src/animal_classifier/gui/app.py`, `.../src/animal_classifier/pipeline.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e19_gui_retag.py`,
      `.../tests/e2e/test_e20_override_lifecycle.py`,
      `.../tests/e2e/test_e21_gui_validation.py`, `.../tests/e2e/test_e17_dry_run.py`,
      `.../tests/e2e/test_e01_source_immutable.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e19_gui_retag.py tests/e2e/test_e20_override_lifecycle.py
      tests/e2e/test_e21_gui_validation.py tests/e2e/test_e17_dry_run.py
      tests/e2e/test_e01_source_immutable.py -q` — E19 all three modes (symlink moves *as* a symlink
      with `os.readlink` unchanged and still dangling; hardlink keeps `st_ino`), `overrides` row
      written, `label_source='human'`, `status='done'`, and the contention leg returning 409 within
      ~250 ms with nothing changed; E20's three runs prove the human label is kept, then suppressed
      with the override row still present, then restored; E21 422s for `Brewer's Blackbird`, the
      reserved `unknown` (even with `--allow-new-labels`), a bad sha256, an out-of-range `limit`, a
      non-bool `include_unscored`, and 404 for an unknown hash, with nothing changed on disk; E17's
      new POST leg returns 409 `reason="planned"` with the row still `planned` and no override row;
      **E1's re-tag legs are now unmarked and green**.

- [ ] 22. **GUI frontend (vanilla JS, no build step).** `index.html`, `app.js`, `app.css`: thumbnail
      grid grouped by label with per-label counts in a sidebar (rendering catalog `count`, with a
      warning marker and a "run verify" hint whenever `files_on_disk != count`), label / confidence /
      date filters with the "N images have no confidence score — show them" and undated toggles wired
      to `include_unscored` / `include_undated`, a `q` search box, a full-size detail view drawing
      boxes on a `<canvas>` scaled from the stored coordinates and `width`/`height`, `style=
      "image-orientation: from-image"` explicit on the detail `<img>`, per-box captions showing
      `cls`, `area_frac`, top-1 species **and `species_rank`**, the top-3 candidates with
      confidences, a placeholder tile carrying the 409 `reason` instead of a broken image, live
      progress from `/api/run`, and a label picker that **offers known labels from `/api/labels` plus
      the artifact label space, displaying `common` and submitting the slug `key`** (NIT 19).
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/gui/static/index.html`,
      `.../gui/static/app.js`, `.../gui/static/app.css`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e18_gui_browse.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e18_gui_browse.py -q` — `/` returns the HTML shell and `/static/app.js`,
      `/static/app.css` are served with the right content types (browser rendering is deliberately
      out of scope per §11.2).

## Phase G — the loop closers and polish (chunks 23–26)

- [ ] 23. **`export-trainset` with the explicit label filter (E25).** Walk the catalog for
      `label_source='human'` (plus `--include-model-labels --min-conf`), emit the §7.5 manifest with
      the dominant box for species labels (`box` omitted when the image has no boxes), skip
      `multiple` and `unknown` **always**, skip `landscape`/`junk` unless `--include-non-species`
      (then emitted without a `box`), and print the counted summary
      `exported=… skipped_multiple=… skipped_unknown=… skipped_landscape=… skipped_junk=… no_box=…`.
      Source images are read read-only.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/cli.py`,
      `.../src/animal_classifier/training/manifest.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e25_export_trainset.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e25_export_trainset.py -q` — manifest lines validate and `train` consumes them;
      a `junk` override and a `multiple` override produce **no** line by default and appear in the
      counted summary; with `--include-non-species` the `junk` image appears as class `junk` without
      a `box` while `multiple` is still absent.

- [ ] 24. **`verify` and `verify --fix` with the specific exit codes (E24).** Implement every check
      of §8 (detector weights size + sha256 + loadable; artifacts loadable at an accepted
      `format_version`; backbone weights present; catalog openable at a known schema version;
      `output_root` writable; `output_root` not nested with the source root read from the **newest
      `runs` row**, reported `skipped(no_runs)` when there is none; catalog/filesystem reconciliation
      ignoring tool-internal entries; bird provider config; an informational reachability probe that
      is never fatal) with exit codes **0 / 1 / 3 / 4** exactly. `--fix` repairs exactly two states —
      completing a **pending re-tag** forward from the `overrides` row, and unlinking a duplicate
      under the *old* label that an override justifies — using only `os.replace`/`os.unlink`;
      everything else is reported and left alone.
      Files: `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/verify.py`,
      `.../src/animal_classifier/cli.py`,
      `/projects/sandbox/AnimalImageClassifier/tests/e2e/test_e24_verify.py`,
      `.../tests/e2e/test_e01_source_immutable.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest
      tests/e2e/test_e24_verify.py tests/e2e/test_e01_source_immutable.py -q` — exit **0** on a good
      tree; **4** naming a missing `dest_path`; **3** for an absent artifact with `--fix` not
      attempting it; **4** for a pending re-tag then **0** after `--fix` with the file only under the
      new label and the row `done`; **4** for the catalog-explained duplicate then **0** after
      `--fix` with exactly one copy left; an orphan file reported but left in place; the source tree
      byte-identical throughout, and E1's `verify --fix` leg now unmarked and green.

- [ ] 25. **Produce the shipped artifacts, and document the commands that produce them
      (review finding 11).** Run the two documented training commands to create
      `models/species.acmodel` (COCO animal classes, `--input-size 224`) and `models/birds.acmodel`
      (CUB-200), record the exact invocations, wall clock and resulting `val_top1` in
      `docs/PROGRESS.md` and in a new "Producing the shipped artifacts" section of `README.md`, and
      state there that relative model paths resolve against the current working directory. Both
      artifacts are gitignored (they are weights) — the *commands* are the deliverable.
      Files: `/projects/sandbox/AnimalImageClassifier/README.md`,
      `/projects/sandbox/AnimalImageClassifier/.gitignore`,
      `/projects/sandbox/AnimalImageClassifier/docs/PROGRESS.md`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen animal-classifier verify
      -o /tmp/ac-verify-root --json` exits **0** with the species and bird artifacts reported present
      and loadable; paste the JSON. Then `uv run --frozen animal-classifier classify
      tests/e2e/_fixtures/card -o /tmp/ac-real-run` files real labels with the real models — paste
      the resulting `ls /tmp/ac-real-run`.

- [ ] 26. **Docs, NIT sweep, and the full-suite green run.** `README.md` gains: quick start, the
      person/vehicle → `landscape` note (also in `classify --help`), the `--link`/`--hardlink`
      trade-offs, the GUI walkthrough, the "no area floor — `dominance_ratio` is the only size gate"
      statement, and the E2E-only testing policy. Sweep the remaining NITs that are not already
      done: NIT 12 (blur reference-edge note in §5.7's implementation and `--help`), NIT 13/14
      (already in chunk 14 — confirm no literals remain via `rg -n "1280|0\.20|0\.45|max_det" src/`),
      NIT 16 (already in chunk 20), NIT 18 (document the fixed, non-configurable 2 px degenerate-crop
      precondition in `images.py`'s docstring and README), NIT 19 (already in chunk 22). Move
      anything still outstanding to `docs/FOLLOWUPS.md` with a reason.
      Files: `/projects/sandbox/AnimalImageClassifier/README.md`,
      `/projects/sandbox/AnimalImageClassifier/docs/FOLLOWUPS.md`,
      `/projects/sandbox/AnimalImageClassifier/src/animal_classifier/images.py`
      Verify: `cd /projects/sandbox/AnimalImageClassifier && uv run --frozen pytest tests/e2e -q`
      — **the whole suite green (E1–E26)**; paste the full summary line and the wall clock. Then set
      `"complete": true` in `docs/impl-status.json`.

## Traceability — every folded-in finding and defect has an owning chunk

| Source | Item | Chunk |
|---|---|---|
| **Reproduced defect 1** | `skipped` PRIMARY KEY `IntegrityError` on the second run; upsert + a two-run e2e test | 3, 10 |
| **Reproduced defect 2** | raw dataset class names leaking into output paths (`022_chuck_will_widow`); normalization layer + directory-name assertion | 4, 18 |
| review MEDIUM 1 | `skipped` upsert published and tested | 3, 10 |
| review MEDIUM 2 | CUB label derivation + `cub200.csv` authoritative for display names | 4, 18 |
| review MEDIUM 3 | `q` parameter specified, validated and tested | 20 |
| review MEDIUM 4 | `date_from`/`date_to` + `include_undated` + `undated_excluded` | 20 |
| review MEDIUM 5 | benign vs abnormal skips → exit 0 vs 4 | 5, 12 |
| review MEDIUM 6 | re-tag of a `planned` row → 409 guard, never `failed` | 21 |
| review MEDIUM 7 | `crop_margin` recorded in the artifact and the artifact wins | 15, 16 |
| review MEDIUM 8 | `coco_species.json` length data-driven, no literal 20/14 | 14, 17 |
| review MEDIUM 9 | E7's two thresholds reconciled, both with a sanctioned remedy | 16, 17 |
| review MEDIUM 10 | re-processing policy covers `failed` (retried) and `skipped` (re-walked) | 3, 10 |
| review MEDIUM 11 | shipped artifacts: how they are produced, CWD-relative paths, bird model optional | 2, 13, 18, 25 |
| review NIT 12 | blur reference-edge scale documented | 6, 26 |
| review NIT 13 | no hard-coded detector literals | 14, 26 |
| review NIT 14 | `detector_max_det` named, validated, documented as not a size gate | 14 |
| review NIT 15 | E16 asserts observable consequences, not "bytes never read" | 12 |
| review NIT 16 | `/api/labels` counts `materializing` | 20 |
| review NIT 17 | `sources` upsert covered by E14 | 11 |
| review NIT 18 | 2 px degenerate-crop precondition documented | 26 |
| review NIT 19 | label picker offers known labels, submits slugs | 22 |
