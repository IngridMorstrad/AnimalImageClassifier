# Gate review — iteration 4 (chunk 5, `scan.py`)

**Reviewed:** `49af62f..HEAD` on `feat/safari-classifier`
**Commits in scope:** `4fdc624` (`feat: scan.py — read-only walk, one rule per skip reason, benign/abnormal exit class`), `bc7d401` (`test: record real e2e run at chunk 5`)
**Verdict:** CHANGES_REQUESTED — **1 blocking finding, and it is sequencing only** (5 of 26 chunks complete, `impl-status.json: complete == false`). Nothing in this iteration's diff needs to be undone or revisited.
**Method:** read the diff plus the captured evidence. The test suite was **not** re-run, per instruction. Two narrow spot-checks were used, both read-only: `rg` for callers of `ensure_image` outside `catalog.py` (none), and `rg` in `docs/design-review.md` to confirm the finding number `scan.py`'s docstring cites.

## 1. What landed

`git diff --stat 49af62f..HEAD` — 605 insertions, 42 deletions, 7 files:

| File | Change |
|---|---|
| `src/animal_classifier/scan.py` | **new, 392 lines** — the whole scanner |
| `src/animal_classifier/catalog.py` | +25/-4 — `ensure_image(refresh_state=...)`, closing FOLLOWUPS F1 |
| `docs/test-report.md` | rewritten for the run at `4fdc624` |
| `docs/impl-status.json` | `done_items` 4 → 5, `current_chunk` → chunk 6 |
| `docs/IMPL-PLAN.md` | chunk 5 checkbox ticked |
| `docs/FOLLOWUPS.md` | F1 ticked, with the closure recorded |
| `docs/PROGRESS.md` | +131 lines of timestamped narrative |

`scan.py` is a single-purpose module: `os.walk`, `os.stat`, `Path.is_symlink`, `os.path.realpath` and nothing else. It yields `Candidate(path, size, mtime_ns)` or `Skipped(path, reason, detail)` for **every** entry, and it exposes `is_benign()` / `exit_code_for()` for the run's exit class.

## 2. Does it match the frozen design and the user's intent?

Checked against `DESIGN.md` §5.1, §5.2 and §10.1 — read as the spec of record, not critiqued, and not edited.

**Skip reasons — all 13 present, each with exactly one rule.** The `SkipReason` enum holds §5.1's eleven scan-time reasons plus §5.2's two decode-time ones (`too_large_pixels`, `decode_error`), which belong here because they land in the same `skipped` table. `symlink_loop` is correctly absent: §5.1 deleted it as unreachable under `followlinks=False`, and `walk()` does pass `followlinks=False` (`scan.py:258`). Every rule matches the §5.1 table:

- `hidden` — basename starts with `.` (`scan.py:337`), and the source root itself is exempt because only entries *below* it are judged, which is what lets a user point the tool at `/media/.card`.
- `system_dir` — `.Trash` as a **prefix** (`.Trashes`, `.Trash-1000`), the other three exact (`scan.py:132-136`). Tested before `hidden` (`scan.py:300-303`) because three of the four also start with a dot and the specific record is the more useful one.
- `unsupported_extension` vs `format_disabled` — split on membership in `EXTENSION_FAMILY`, which is derived from `config.FAMILY_EXTENSIONS`, i.e. §3.1 (`scan.py:125-129, 345-356`). "Belongs to no family" vs "known family, not requested" is exactly §5.1's distinction.
- `video` — the seven extensions from §5.1, always skipped (`scan.py:113-117`).
- `raw_not_enabled` — the eight RAW extensions, governed solely by `--raw`, and deliberately *outside* the `formats` family check (`scan.py:118-122, 343-344`), which is §3.1's rule that RAW is not a family.
- `zero_bytes`, `too_large` — `st_size == 0` and `st_size > max_file_bytes`, both from one `stat()` (`scan.py:383-389`). §5.1's requirement that an over-cap file's **bytes are never read** holds structurally: the only reads in this module are `stat`.
- `unreadable` — `OSError` on `stat` (`scan.py:376-382`) *and* on descending a directory, via `os.walk(onerror=...)` (`scan.py:246-256`). §5.1 names both sources and both are covered; `os.walk` silently swallows `scandir` failures unless `onerror` is supplied, so photos in an unlistable directory would otherwise have vanished with no row at all.
- `symlink` / `symlink_escape` — default-skip for symlinked files, `--follow-source-symlinks` to ingest, and even then `os.path.realpath` must be `is_relative_to(root)` (`scan.py:353-368`). Symlinked **directories** are pruned as `symlink` under every flag (`scan.py:304-310`), matching §5.1's "never descended under any flag". The escape rejection carries §5.1's stated reason in a comment: ingesting it would put a `sources.path` inside the card for bytes that live outside it.

**Invariant I2 — no area floor.** Honoured, and the docstring is careful about the one thing that could be mistaken for a violation: `max_file_bytes` is a whole-file `st_size` cap applied *before any decode*, so it structurally cannot see a detection box (`scan.py:23-30`, and `DESIGN.md:295` defines it in exactly those terms). §5.2's pixel cap is the separate reason `too_large_pixels`, so the catalog can never confuse a huge file with a huge raster. `dominance_ratio` remains the only size gate, still guarded executably by the six parametrised CLI tests. No `min_box_area` or absolute box-area floor is introduced anywhere in this diff.

**Read-only guarantee.** Not violated, and improved. The module has no write call of any kind. Pruning mutates `dirnames` in place (`scan.py:313`), so a refused directory's contents are never even *listed* — the cheapest possible way to keep the promise. `root.resolve()` is computed once and used as the containment boundary for the escape check.

**I7, fail loud.** `scan()` raises `ConfigError` when `config.source_root is None` (`scan.py:205-209`) instead of substituting the working directory, and `walk()` raises `ConfigError` when the root is not a directory (`scan.py:239-243`), matching §10.1's "source missing / not a dir → fatal, exit 3". No silent default is introduced.

**Benign vs abnormal exit class.** `BENIGN_REASONS` / `ABNORMAL_REASONS` partition the enum, and the partition is enforced **at import time** (`scan.py:103-110`): a reason added to the enum without being classified raises `AssertionError` on import rather than silently inheriting "benign" at the first `not in ABNORMAL_REASONS` test. That is the right shape — the failure mode it prevents (a new real failure quietly becoming exit 0) is invisible in testing. The classification matches §10.1's table: the rows marked "not an error"/`DEBUG` (`format_disabled`, `too_large`, `symlink`, `video`, `hidden`, `system_dir`, `unsupported_extension`, `raw_not_enabled`) are benign; the rows that say "recoverable"/exit 4 (`symlink_escape`, `zero_bytes`, `decode_error`, `too_large_pixels`) plus `unreadable` are abnormal. `exit_code_for` also takes `n_failed` for the per-image failures that are not skips at all. One deviation from a literal DESIGN sentence is discussed in §5 below (F15) — it resolves a contradiction the cold design review already flagged, and it is not blocking.

**FOLLOWUPS F1 is genuinely closed, and closed the right way.** The previous review's blocking text asked for this before `scan.py` was written, and it was. `ensure_image` no longer refreshes everything-but-`first_seen`: `refreshed = [*columns, "last_updated"]`, with `status`/`run_id` joining that list only under an explicit `refresh_state=True` (`catalog.py:711-716`). Two things make this correct rather than merely plausible:

- `status` and `run_id` are **explicit keyword parameters**, so neither can ever arrive through `**columns`. The `refreshed` list therefore cannot contain a duplicate assignment, and every name in it is guaranteed to be present in the INSERT's `values` dict (so `excluded.<name>` always resolves). `refreshed` is never empty because `last_updated` is unconditional.
- `first_seen` is still never in the UPDATE list, so a hash keeps its discovery time through every `--reclassify`.

The effect is that a re-walk cannot demote a `done` row to `planned` from the caller side, which is where DEFECT 1's `skipped` upsert could not protect the second-run-is-a-no-op invariant. `rg` confirms there is **no caller of `ensure_image` outside `catalog.py`** today, so the narrowing cannot have broken an existing call site. The empirical check is recorded in `PROGRESS.md:1282` — a real catalog where a `done`/`r1` row survived `ensure_image(sha, run_id=r2)` unchanged (`disposition: skip_done`) and moved to `planned`/`r2` only with the opt-in, `first_seen` unchanged throughout.

## 3. What the real test output proves — and what it does not

From `docs/test-report.md` (run at `4fdc624`, verbatim `uv run pytest tests/e2e -v` captured):

```
collected 7 items ... 7 passed in 0.97s
```

Process exit code **0**; confirmed reproducible by a second `-q` run. No failures, no collection errors, nothing skipped, so there is no failing test name or assertion to report.

**Proven:** the CLI surface exists and exits 0 for `--help` on all six commands, and **none of the six commands offers an absolute area floor** under any of ten spellings (`min-box-area`, `min_box_area`, `min-area`, `min_area`, `area-floor`, `area_floor`, `min-animal-area`, `min_animal_area`, `min-box-frac`, `min_box_frac`). These are genuinely end-to-end in mechanism: `conftest.py` resolves the installed `animal-classifier` console script from `PATH` and runs it as a real subprocess. That is an executable regression guard on the user's standing requirement, and it is the most important thing the suite currently protects.

**Not proven, and the report says so plainly:** nothing in this run touches `scan.py`, `catalog.py` or `taxonomy/labels.py` — none is reachable from a `--help` invocation. Detection, the dominance rule, materialization, the catalog, the GUI, training and the bird providers are all untested at this commit. So this iteration's subject module has **no executable coverage**; its correctness rests on this review's reading plus the ad-hoc runs in `PROGRESS.md`. That is a sequencing fact at chunk 5 of 26, not a defect: the plan puts `scan.py`'s first e2e coverage at chunk 9 (E6's full form) and all thirteen skip reasons at chunk 12 (E16). It is recorded as F17 below so it cannot be forgotten.

**Test hygiene — all clean.** The report captures a `find` over `tests/` for anything outside `tests/e2e/` that printed **no output at all**; `tests/e2e/` holds only `conftest.py` and `test_cli_surface.py`; `pyproject.toml` pins `testpaths = ["tests/e2e"]`. No unit test exists, no test lives outside `tests/e2e/`, no test was deleted or weakened to make the suite green, and the green claim is backed by verbatim captured output with an exit code. The test set and result are identical to the previous recorded run (7 passed), which is the expected outcome for a chunk that added no test.

## 4. Blocking findings

### B1 — Implementation is 5 of 26 chunks complete; `impl-status.json` has `complete == false`

- **File:** `docs/impl-status.json` (`complete: false`, `done_items: 5`, `total_items: 26`, `current_chunk: "6. images.py: decode, the one coordinate frame, sha256, blur, crop"`). `docs/IMPL-PLAN.md` agrees: chunk 5 ticked, 21 unticked.
- **Why blocking:** approval requires `complete == true`. Twenty-one plan chunks are unwritten and the e2e suite still covers only the CLI surface, so no behavioural guarantee in `DESIGN.md` has executable proof yet.
- **Explicitly not a defect in this diff.** `scan.py` and the `catalog.py` narrowing are correct as written; nothing here needs rework. This finding is the loop's progress counter, not a criticism of chunk 5.
- **What correct looks like:** continue from chunk 6 through chunk 26, updating `done_items` and `current_chunk` after each, and set `complete: true` only when all 26 items are ticked **and** `uv run --frozen pytest tests/e2e -q` is green over E1–E26 with real captured output in `docs/test-report.md`. The gates still entirely unproven by tests: chunk 9 must land E6's full form (a real `classify` over the fixture card proving `dominance_ratio` decides alone) plus the second-run-is-a-clean-no-op test that exercises both halves of the idempotency fix (`record_skip`'s `ON CONFLICT(path)` upsert *and* `ensure_image`'s new `refresh_state` opt-in — the caller must pass `refresh_state=True` only after `plan_disposition` returns `PROCESS`); chunk 11 brings `materialize.py` with atomic writes and the `--link`/`--hardlink`/`--dry-run`/`--reclassify` contract; chunk 12's E16 must cover all thirteen skip reasons and pin the exit class; E22 covers fail-loud config; chunks 15–19 bring the species and bird heads, where a low-confidence result must file as `unknown` rather than a guessed species, and where E8 must assert human-readable output directories (`chuck_wills_widow`, never `022_chuck_will_widow`).

No other blocking finding. Specifically checked and clean this iteration: no `min_box_area` or absolute area floor; no unit test and no test outside `tests/e2e/`; the green claim has real captured output; no write to the source tree; no silent default for a missing required value; the second-run-no-op invariant is *strengthened*, not weakened; no raw or digit-prefixed class name can reach a directory (untouched this chunk); no materialization contract exists yet to break; no species filing yet to mis-file.

## 5. Non-blocking observations (appended to `FOLLOWUPS.md` as F15–F20)

- **F15 — exit class for benign skips deviates from `DESIGN.md:1211`'s literal wording.** §10.1 says "`4` completed with per-image failures **or skips**", and `scan.py` exits 0 when only benign skips happened. This is the right call: `docs/design-review.md:115` finding 5 is titled exactly "exit code 4 vs 0 for benign skips is self-contradictory, and E2/E16 cannot both pass", and §10.1's own table marks `video`/`format_disabled`/`too_large`/`symlink` as "not an error"/`DEBUG`. A card full of videos is a successful run. Recorded because the resolution now lives only in a module docstring — E2 and E16 must assert the chosen semantics so a future reader cannot re-derive the literal sentence.
- **F16 — `_judge_file`'s rule order is not literally §5.1's table order.** `video` and `raw_not_enabled` are tested before `unsupported_extension`/`format_disabled` (`scan.py:319-336`), while the table lists them after. Behaviourally right — `.mp4` and `.cr2` belong to no `formats` family, so the table order would record the *less* specific `unsupported_extension` — and both candidates are benign, so the exit class is identical either way. Only the docstring's claim of "the fixed precedence order documented there" overstates it.
- **F17 — `scan.py` has no executable proof at this commit** (self-resolving at chunks 9/12; same shape as F7/F12). None of the thirteen reasons, the directory pruning, the `onerror` drain ordering, or `exit_code_for` is reachable from a `--help` invocation.
- **F18 — an unreadable *root* surfaces as `Skipped(unreadable)` (exit 4), not §10.1's fatal exit 3.** `walk()` checks `root.is_dir()` but not readability; a card root that exists and is a directory yet cannot be listed goes through `_on_error` as a single abnormal skip. It still fails loudly and non-zero, so nothing is silent — but §10.1's "source missing / not a dir / **not readable** → fatal, exit 3" wants the up-front check. Do it in chunk 9 where `classify` wires the exit codes.
- **F19 — `pending_errors` drain is duplicated and O(n).** The `while pending_errors: yield pending_errors.pop(0)` block appears twice (`scan.py:266-268, 281-282`); a `deque` or a `yield from`-then-`clear` would read better. No behaviour change — the ordering comment explaining *why* the drain sits at the top of the loop body is worth keeping either way.
- **F20 — two spellings of one predicate.** `Skipped.benign` (property) and module-level `is_benign()` both answer the same question; the property has no caller today. `is_benign()`'s string-accepting form is the load-bearing one (it classifies a reason read back out of the catalog and raises `ValueError` on an unrecognised value rather than defaulting to benign).

`docs/DESIGN.md` was not edited, and was read only as the spec of record — the design phase is over and the design itself is not under review.
