# Build review — iteration 1

**Review timestamp:** 2026-09-15 21:57 UTC
**Branch:** `feat/safari-classifier` @ `c59a656`
**Diff reviewed:** `6a3cc67..HEAD` (commits `a8103ab`, `c59a656`)
**Verdict:** CHANGES_REQUESTED — 1 blocking finding (the checklist is 2 of 26 complete)

---

## 1. What landed this iteration

Two commits, both squarely inside plan chunks 1-2:

| File | Lines | What it is |
|---|---|---|
| `pyproject.toml` | +38/-32 | Pinned dependency contract, `testpaths = ["tests/e2e"]`, three `override-dependencies` neutralising hostile yolov5 transitives |
| `uv.lock` | +1848 | Frozen resolution for the above |
| `src/animal_classifier/errors.py` | +100 | Exception hierarchy carrying the exit-code contract (0/1/2/3/4) |
| `src/animal_classifier/config.py` | +787 | Layered resolution (CLI → env → TOML → default), total validation, frozen `Config` |
| `.python-version` | +1 | 3.12 pin |
| `docs/*` | +284 | PROGRESS, RECON, IMPL-PLAN, impl-status, test-report updates |

No pipeline code exists yet: `src/animal_classifier/` holds `__init__.py`, `cli.py` (still the
original scaffold), `errors.py` and `config.py`. `catalog.py`, `scan.py`, `detect.py`, `decide.py`,
`materialize.py`, the training code, the bird providers and the GUI are all unwritten.

## 2. Does it match the frozen design and the user's stated intent?

Yes, for the surface it covers.

**The TOML key set matches DESIGN.md §3 exactly.** I diffed `DEFAULTS` in `config.py:104-124`
against the TOML block at `DESIGN.md:174-190`: same 17 keys, same values —
`output_root=~/animal_pics`, `mode=copy`, `dominance_ratio=1.6`, `min_species_confidence=0.45`,
`detector_confidence=0.20`, `detector_iou=0.45`, `detector_image_size=1280`, `crop_margin=0.08`,
`blur_threshold=100.0`, `max_file_bytes=536870912`, the three asset paths,
`bird_provider=own_bird_head`, `jobs=7`, `device=auto`, `formats=[jpeg,png,tiff,heic]`. The one
addition, `detector_max_det=100`, is justified in the module docstring (`config.py:19-21`): §5.4
quotes the value inline and the detector should read NMS parameters from `Config` rather than
hard-code them. A documented, deliberate superset — not a drift.

**No area floor exists, under any name.** `rg 'min_box_area|min_animal_area|box_area|area_floor|min_area'`
across the whole repo returns matches only in prose that asserts the *absence* of such a key
(`config.py:14`, `IMPL-PLAN.md:23,77`, `DESIGN.md:194,596,1378`, `PLAN.md:30`). There is no
comparison of a box area against any constant anywhere in the source. `dominance_ratio` is
range-checked with `minimum=1.0` and the rationale string "a ratio below 1.0 would make the smaller
box dominant" (`config.py:243-246`), which is invariant I2 exactly. `max_file_bytes` is kept as a
whole-file scan cap and is structurally unable to see a detection box, as §5.1 requires.

Better still, the unknown-key check is built as the *enforcement mechanism* for I2 rather than as
generic hygiene: `_parse_toml` (`config.py:349-357`) rejects unknown keys and the error message
itself says a box-area floor setting deliberately does not exist. A future attempt to reintroduce a
floor by configuration fails loudly with an explanation.

**Fail-loud on missing required values holds.** I looked specifically for the silent-default
pattern the spec forbids (invariant I7):
- `SOURCE` missing for `classify` → `ConfigError` (`config.py:520-523`), not a default.
- `ebird_enrich` without `ANIMAL_CLASSIFIER_EBIRD_API_KEY`, including an empty string →
  `ConfigError` (`config.py:597-603`), because `_ebird_api_key` normalises `""` to `None` first.
- `hosted_bird_api` → `AssetError` naming the two env vars it would need (`config.py:605-612`),
  rather than pretending to work.
- `--device cuda` with no CUDA device → `ConfigError` (`config.py:474-479`); `auto` resolves and
  logs. A broken torch import is fatal rather than swallowed (`config.py:489-495`).
- Missing detector weights / species model / bird model → `AssetError` carrying the resolved
  absolute path *and* a copy-pasteable `train` command (`config.py:615-644`).
- No `dict.get(key, fallback)` for a required key and no bare `except` in the module.

The values that *do* fall back (`_number`, `_flag`, `_enum_value`) fall back to the documented
built-in default layer, which is the design's bottom layer, not a substitution for something
required.

**Read-only-card intent is respected at the config layer.** `_guard_nesting` (`config.py:558-578`)
rejects `output_root == SOURCE`, `output_root` inside `SOURCE`, and `SOURCE` inside `output_root`,
each with a message explaining which harm it prevents. Nothing in this diff writes to any path.

**Type coercion is careful in the places that usually go wrong.** `_coerce_number`
(`config.py:396-441`) rejects booleans before numeric coercion (so `dominance_ratio = true` cannot
become `1.0`), rejects NaN and ±inf, and rejects a float that is not integral where an int is
required. `Config` is `frozen=True, slots=True`, so a validated value recorded in the `runs` row
cannot be mutated downstream.

## 3. What the real test output proves — and does not

`docs/test-report.md` captures `uv run pytest tests/e2e -v` verbatim: **0 collected, 0 passed,
0 failed, 0 errored**, process exit code 5 (`EXIT_NOTESTSCOLLECTED`).

What that proves: the environment resolves and pytest runs; `testpaths` points at `tests/e2e`; and
a `find` scan confirms **zero `.py` files under `tests/` outside `tests/e2e/`**, so the e2e-only
rule is not violated. `ls tests/e2e/` independently confirms one file, `conftest.py`.

What it does not prove: any behaviour whatsoever. Not one line of `config.py` or `errors.py` is
executed by a test. Every claim in section 2 above rests on my reading of the source, not on
observed behaviour.

I want to be explicit that this is **not** a "green suite without evidence" violation. The report
does not claim green — it states plainly that nothing passed, explains that the suite is empty until
chunk 9, and even warns the caller that pytest's exit 5 will look like a failure to a gate keyed on
non-zero exits. That honesty is the correct behaviour and I am crediting it rather than penalising
it. The first executable proof of the config contract arrives with E22 (fail-loud config) and E6
(no area floor) in chunk 9.

## 4. Blocking findings

### B1 — The implementation is 2 of 26 chunks complete; `impl-status.json` has `complete: false`

`docs/impl-status.json` reports `done_items: 2`, `total_items: 26`,
`current_chunk: "3. catalog.py: schema, indexes, upserts, statuses — including reproduced DEFECT 1"`.
Approval requires `complete == true`. It is false, so the verdict is CHANGES_REQUESTED and the loop
must continue coding.

This is a sequencing finding, not a defect in what landed. Nothing in `errors.py`, `config.py` or
`pyproject.toml` needs to be undone. To close it, continue from chunk 3 through 26 in
`docs/IMPL-PLAN.md`. The load-bearing gates still entirely unproven, each needing real captured
output:

- **Chunk 3** must reproduce and then fix the `skipped`-table PRIMARY KEY `IntegrityError` so a
  second run over the same tree is a clean no-op.
- **Chunk 9** lands the first e2e tests, including E2/E5/E6 (dominance boundaries and the no-area-floor
  proof, with `classify --help` exposing no floor-like option) and E22 (fail-loud config).
- **Chunk 11** brings `materialize.py`: atomic writes, and the `--link` / `--hardlink` / `--dry-run` /
  `--reclassify` contract.
- **Chunks 15-19** bring the species and bird heads, where "low confidence files as `unknown`, never
  as a guessed species" and the human-readable slug rule (`chuck_wills_widow`, never
  `022_chuck_will_widow`) must both be demonstrated.

None of these can be reviewed until they exist. Please do not treat B1 as a request to revisit
chunks 1-2.

## 5. Non-blocking observations

All five are appended to `docs/FOLLOWUPS.md`. None withholds approval.

1. **`output_root` is not symlink-resolved, so `_guard_nesting` can be evaded** (medium; fix by
   chunk 11). `_resolve_source_root` uses `Path.resolve()` (`config.py:524`) but
   `_resolve_output_root` uses `os.path.abspath` (`config.py:551`), which normalises `..` without
   following symlinks. An `output_root` that is itself a symlink into the card would pass all three
   nesting checks and then be written to. `Path(...).resolve()` is non-strict in Python 3.12 and
   resolves the existing ancestors of a not-yet-created directory, so the existence tolerance that
   motivated `abspath` is not actually needed. Worth fixing before `materialize.py` gives the guard
   something to protect; harmless today because nothing writes.
2. **`pytest` is in `[project.optional-dependencies] dev`, but the documented command does not
   request it** (medium). `impl-status.json` quotes `uv run --frozen pytest tests/e2e -q` and the
   report ran `uv run pytest tests/e2e -v`; neither passes `--extra dev`. It worked because the
   venv happens to have pytest. Under `--frozen` on a clean machine it would not. Move pytest to a
   dependency-group or pin the command as `uv run --frozen --extra dev pytest ...` before chunk 9
   makes the suite meaningful.
3. **`_load_toml`'s injected-env branch duplicates `config_search_paths()` and loses its
   deduplication** (nit). `config.py:322-340`: the `env is os.environ` branch calls
   `config_search_paths()`, the else branch rebuilds the same list inline without the dedupe of
   `XDG_CONFIG_HOME == ~/.config`. Harmless (a duplicate `is_file()` probe), but the two branches
   will drift. Parameterising `config_search_paths(env)` collapses them.
4. **pytest's exit 5 on an empty suite will read as a gate failure until chunk 9** (nit, process).
   Already flagged by the e2e step itself. Self-resolving; noted so nobody debugs a phantom failure.
5. **`detector`, `limit`, `port` and the boolean flags are CLI/env-only, not TOML-settable** (nit).
   `KNOWN_TOML_KEYS = frozenset(DEFAULTS)` (`config.py:127`) so these are rejected in a TOML file.
   This matches DESIGN.md, which shows them only as flags (`--detector scripted` in E6), so it is
   correct as written — recorded only so a future reader does not mistake it for an omission.

## 6. Assessment

The code quality here is high and the two properties most at risk in this project — no area floor,
no silent defaulting — are not merely satisfied but built so that violating them later requires
deleting an explanatory error message. Nothing in this diff needs to be reverted or redone.

The blocker is simply that 24 of 26 chunks remain and the suite is still empty, so the design's
behavioural claims have zero executable proof. Continue at chunk 3.
