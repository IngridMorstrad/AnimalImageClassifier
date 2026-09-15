# Build review — iteration 2 (2026-09-15 22:17 UTC)

**Range reviewed:** `f682087..HEAD` (`c87a690` catalog.py, `6377ac8` test report) on
`feat/safari-classifier`.
**Spec of record:** `docs/DESIGN.md` (frozen; not reviewed, not edited).
**Verdict:** CHANGES_REQUESTED — **1 blocking finding**, and it is the same sequencing finding as
iteration 1: `docs/impl-status.json` has `complete: false` at 3 of 26 chunks. Nothing that landed in
this iteration needs to be undone.

## 1. What landed

| Area | Change |
|---|---|
| `src/animal_classifier/catalog.py` | New, 1038 lines. Full §5.9 catalog: schema, statuses, upserts, run lifecycle, `replace_inference`, overrides, lock backoff. |
| `src/animal_classifier/errors.py` | New `CatalogLockedError` (exit 4, `EXIT_PARTIAL`), deliberately not a `CatalogError`. |
| `src/animal_classifier/config.py` | `_resolve_without_requiring_existence` switched from `os.path.abspath` to `Path.resolve()`. |
| `pyproject.toml`, `uv.lock` | `pytest==9.1.1` moved from `[project.optional-dependencies]` to PEP 735 `[dependency-groups]`. |
| `tests/e2e/conftest.py`, `tests/e2e/test_cli_surface.py` | New. E6's CLI-surface leg, 7 tests. |
| `docs/*` | test-report, PROGRESS, FOLLOWUPS, impl-status, IMPL-PLAN checkbox. |

Both mediums that iteration 1 filed as follow-ups were **fixed early**, ahead of their stated
deadlines (chunk 11 and chunk 9 respectively). That is the right response to a follow-up list and is
worth recording as a positive.

## 2. Match against the frozen design and the user's stated intent

**Invariant I2 — no absolute box-area floor (the user's explicit instruction).** Clean, and now for
the first time *executably* guarded. `grep -rniE "min_box_area|min_area|area_floor|min_animal_area|min_box_frac"`
over `src/` and `tests/` matches only (a) prose asserting the absence at `config.py:14`, and (b) the
denylist inside `tests/e2e/test_cli_surface.py`. `grep -rnE "area_frac\s*[<>]"` returns **zero
matches** — no box area is compared to anything. `catalog.py` stores `area_frac` on `boxes` and its
`BoxWrite` docstring states the column exists for the GUI and the dominance audit trail and "is
never compared against a floor". `dominance_ratio` remains the only size gate.

**DEFECT 1 — second run over the same tree must be a clean no-op.** The fix is present and is
structured so it cannot be quietly undone: `SKIPPED_UPSERT_SQL` (`catalog.py:189-197`) is
`INSERT INTO skipped(...) ON CONFLICT(path) DO UPDATE SET ...`, `record_skip` (`catalog.py:795-822`)
is documented as the only sanctioned write path into that table, and `grep -rn "INSERT INTO skipped" src/`
finds exactly one statement — the named upsert constant. `sources` gets the same treatment via
`SOURCES_UPSERT_SQL`. `replace_inference` (`catalog.py:~860-970`) deletes `candidates` then `boxes`
for the hash before re-inserting, inside one `BEGIN IMMEDIATE`, so a re-classified hash cannot
accumulate boxes. `ensure_image` excludes `first_seen` from its `DO UPDATE` list. This is the right
shape. It is **not yet proven by execution** — see §3 and follow-up F1.

**Read-only source card.** Nothing in this diff writes to a source path. `Catalog.open` only ever
creates `path.parent` for the catalog file, which lives under `output_root`. The `config.py` change
strengthens this: `output_root` is now symlink-resolved, so an `output_root` that is itself a symlink
into the card is caught by `_guard_nesting` instead of sailing through three textual checks — the
exact evasion iteration 1 flagged. FOLLOWUPS.md records a real verification run of the rejection.

**Fail loud, never substitute a default for a missing required value (I7).** Holds. `start_run`
rejects an empty `source_root` with a message naming why it matters (`catalog.py:~540-548`) rather
than storing `""` into a `NOT NULL` column. `ensure_image` and `update_image` validate keyword names
against the `UPDATABLE_IMAGE_COLUMNS` allowlist and raise on an unknown column instead of updating
nothing. `update_image` and `replace_inference` both check `cursor.rowcount == 0` and raise, so a
write against a missing row is an error, not a silent zero-row success. `_check_schema_version`
refuses a newer schema untouched, refuses a non-numeric version, and refuses a read-only open of a
file with no `schema_version` row rather than presenting an empty database. The one default in the
module (`status=Status.PLANNED`) is a documented default for a new row, not a stand-in for a value
the caller failed to supply.

**Label precedence (I6).** `replace_inference` never touches `overrides` and never sets
`label_source`, so a human label survives `--reclassify` unless a caller explicitly demotes it.
`insert_override` only appends; nothing in the module deletes or rewrites an override row.

**Lock contention.** `do_write` retries 0.5/1/2/4/8 s and then raises `CatalogLockedError`, which
carries `exit_code = EXIT_PARTIAL` (4), so contention is a per-image outcome and not a dead run —
matching §5.9. `write_tx` takes the lock up front with `BEGIN IMMEDIATE`, so a contended writer fails
before doing partial work, which is what makes retrying the whole body sound.

**e2e-only tests (the user's explicit instruction).** Holds. `tests/` contains exactly two `.py`
files, both under `tests/e2e/`; `pyproject.toml` pins `testpaths = ["tests/e2e"]`. The new tests
drive the installed `animal-classifier` console script as a real subprocess via `run_cli` and assert
on observable stdout/exit code — genuinely end-to-end, not a unit test in an e2e directory. The
`cli_path` fixture raises rather than falling back to `python -m`, which would silently test a
different surface.

**No raw or digit-prefixed class names, no low-confidence guess filed as a species, atomic writes,
`--link`/`--hardlink`/`--dry-run`/`--reclassify` contract.** Not yet reachable: `taxonomy/` is
chunk 4, `materialize.py` is chunk 11, the species and bird heads are chunks 15-19. No code in this
diff creates an output directory or writes an image. Nothing here pre-violates any of them.

## 3. What the real test output proves — and what it does not

`docs/test-report.md` captures a verbatim run of `uv run pytest tests/e2e -v`: **7 collected, 7
passed, 0 failed, 0 errored, exit code 0**, up from the previous run's exit 5 on an empty suite. I did
not re-run the suite.

What that output proves: the CLI exists, `--help` exits 0 and names all six commands, and no
command's `--help` offers any of 10 spellings of an area floor. That is a green executable guard on
invariant I2 — the user's most explicit standing requirement.

What it does **not** prove, and the report says so itself in as many words: nothing about detection,
the dominance rule, materialization, **the catalog**, the GUI, training, or the bird providers. Every
correctness claim in §2 about `catalog.py` — including the DEFECT 1 idempotency fix, the strongest
claim in this iteration — rests on reading the source, not on observed behaviour. The report's
honesty about its own scope is correct behaviour and is not treated as a defect; the consequence is
simply that the fix stays unproven until the first `classify` e2e test lands (chunk 9).

The report also states, verifiably from `docs/impl-status.json` rather than inferred, `done_items: 3`
of `total_items: 26`. `grep -cE "^- \[x\]" docs/IMPL-PLAN.md` returns 3 and `^- \[ \]` returns 23,
so the checklist and the status file agree.

## 4. Blocking findings

### B1 — implementation is 3 of 26 chunks complete; `impl-status.json` has `complete: false`

- **File:** `docs/impl-status.json` (`complete: false`, `done_items: 3`, `total_items: 26`,
  `current_chunk: "4. taxonomy/: slug(), LABEL_RE, RESERVED_LABELS and the static name tables"`).
- **Why blocking:** approval requires `complete == true`. 23 of 26 plan chunks are unwritten and the
  e2e suite covers only the CLI surface, so none of the design's behavioural guarantees — including
  the DEFECT 1 idempotency fix that landed this iteration — has executable proof.
- **Not a defect in what landed.** Nothing in `catalog.py`, `errors.py`, `config.py`,
  `pyproject.toml` or `tests/e2e/` needs to be undone or revisited.
- **What correct looks like:** continue from chunk 4 through chunk 26 of `docs/IMPL-PLAN.md`,
  updating `done_items` and `current_chunk` after each, and set `complete: true` only when all 26
  checklist items are checked and `uv run --frozen pytest tests/e2e -q` is green over E1-E26 with
  real captured output in `docs/test-report.md`. Gates still entirely unproven: chunk 9's E6 full
  form (a real `classify` proving the dominance rule decides alone) plus E22 (fail-loud config) and
  the second-run-is-a-no-op test that finally exercises `record_skip`'s upsert; chunk 11's
  `materialize.py` with atomic writes and the `--link`/`--hardlink`/`--dry-run`/`--reclassify`
  contract; chunks 15-19, where a low-confidence result must file as `unknown` rather than a guessed
  species and output directories must be human-readable slugs (`chuck_wills_widow`, never
  `022_chuck_will_widow`).

## 5. Non-blocking observations

All of these are appended to `docs/FOLLOWUPS.md`. Per the anti-stall rule none of them withholds
approval; B1 is the sole reason the verdict is not APPROVED. F1 is the one worth reading before
writing chunk 5 — it is a latent correctness trap, not polish, but it has no caller today and so
nothing observable is broken.

- **F1 (high priority, act on it in chunk 5 — `scan.py`).** `ensure_image` (`catalog.py:~660-700`)
  upserts with `status` and `run_id` in its `DO UPDATE` list and defaults `status=Status.PLANNED`. If
  chunk 5's scanner calls `ensure_image` unconditionally for every hash it walks, a second run over
  the same card silently resets every `done` row to `planned`, and `plan_disposition` — which decides
  `SKIP_DONE` purely from `row.status` — would then re-process the whole card. That is precisely the
  "second run must be a clean no-op" invariant, defeated from the caller side rather than by the
  `skipped` table. There is no caller yet (`grep` for `ensure_image` outside `catalog.py` returns
  nothing), so nothing is broken now. Either have the scanner consult `plan_disposition` before
  calling `ensure_image`, or make `status`/`run_id` refreshable only on an explicit opt-in argument.
  Chunk 9's idempotency e2e test must cover it either way.
- **F2 (low).** `write_tx`'s `except` branch issues `ROLLBACK` unconditionally
  (`catalog.py:~470-480`). If SQLite has already rolled the transaction back, the `ROLLBACK` raises
  "cannot rollback - no transaction is active" and masks the original exception. Wrap it so the
  original propagates.
- **F3 (low, GUI-facing).** Read accessors are inconsistent in return type: `image()`, `run()` and
  `newest_run()` return frozen value objects, while `boxes()`, `candidates()` and `newest_override()`
  return raw `sqlite3.Row`. Fine within the module; decide before the GUI serializers in chunks 20-22
  so row-index access does not leak into the web layer.
- **F4 (nit).** `update_run_counts` returns silently when every count argument is `None`
  (`catalog.py:~576-586`), so a caller that passes nothing gets a successful no-op. Defensible for an
  all-optional progress publisher; noted only so it is a decision.
- **F5 (nit).** `runs.state` has no CHECK constraint, unlike `images.status` and
  `images.label_source`, so `RunState` is enforced only in Python. The docstring already explains
  that §5.9 declares the column without enumerating values, so this is faithful to the spec, not
  drift.
- **F6 (carried over, still open).** `_load_toml`'s injected-env branch duplicates
  `config_search_paths()` and drops its `XDG_CONFIG_HOME == ~/.config` dedupe
  (`config.py:322-340`). Unchanged this iteration.
- **F7 (informational).** The DEFECT 1 fix has no executable proof yet; it is asserted by source
  reading only. Self-resolving at chunk 9. Recorded so that nobody later mistakes "reviewed" for
  "tested".
