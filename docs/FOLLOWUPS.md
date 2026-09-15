# Follow-ups

Non-blocking observations raised during build review. Nothing here withholds approval.
Append only; note the review iteration that raised each item.

## From build review iteration 1 (2026-09-15 21:57 UTC, `c59a656`)

- [x] **`output_root` is not symlink-resolved, so `_guard_nesting` can be evaded** (medium — fix by
      chunk 11, before `materialize.py` exists). `_resolve_source_root` uses `Path.resolve()`
      (`src/animal_classifier/config.py:524`) but `_resolve_output_root` uses `os.path.abspath`
      (`config.py:551`), which normalises `..` without following symlinks. An `output_root` that is
      a symlink into the SD-card tree passes all three checks in `_guard_nesting` and would then be
      written to, violating the read-only-card invariant. `Path(...).resolve()` is non-strict in
      3.12 and resolves existing ancestors of a not-yet-created directory, so the
      does-not-exist-yet tolerance that motivated `abspath` is not needed. Harmless today because
      nothing in the tree writes.

- [x] **`pytest` is in `[project.optional-dependencies] dev` but the documented test command does
      not request it** (medium — before chunk 9). `docs/impl-status.json` quotes
      `uv run --frozen pytest tests/e2e -q`; `docs/test-report.md` ran `uv run pytest tests/e2e -v`.
      Neither passes `--extra dev`. It succeeded only because the existing venv has pytest; under
      `--frozen` on a clean machine it would not resolve. Either move pytest to a dependency-group
      or fix the canonical command to `uv run --frozen --extra dev pytest tests/e2e -q`.

- [ ] **`_load_toml`'s injected-env branch duplicates `config_search_paths()` and drops its
      dedupe** (nit). `config.py:322-340`: the `env is os.environ` branch calls
      `config_search_paths()`; the else branch rebuilds the same candidate list inline without
      deduplicating `XDG_CONFIG_HOME == ~/.config`. Currently only costs a redundant `is_file()`
      probe, but the two branches will drift. Parameterise as `config_search_paths(env)` and call it
      from both.

- [x] **pytest exits 5 on an empty suite, which reads as failure to an exit-code gate** (nit,
      process). Already flagged by the e2e step in `docs/test-report.md`. Self-resolves when chunk 9
      lands the first test. Recorded so nobody debugs a phantom failure in the meantime.
      **Closed in chunk 3**, earlier than planned: E6's CLI-surface leg
      (`tests/e2e/test_cli_surface.py`) landed now because it is provable the moment a CLI exists,
      so the suite is 7 passed / exit 0 instead of exit 5. Chunk 9 still lands E6's full form (a
      real `classify` run proving the dominance rule decides alone).

## Closed in chunk 3 (2026-09-15 22:11 UTC)

The first two mediums above are fixed in this commit, ahead of their deadlines:

- `_resolve_without_requiring_existence` now uses `Path(path).resolve()`, so an `output_root` that
  is a symlink into the card is rejected by `_guard_nesting`. Verified with a real run: a symlink
  `pics_link -> card/DCIM/out` now raises `ConfigError: output_root /…/card/DCIM/out is inside
  SOURCE /…/card`, where before the fix it was accepted.
- `pytest==9.1.1` moved from `[project.optional-dependencies] dev` to `[dependency-groups] dev`
  (PEP 735). uv installs that group by default, so the canonical
  `uv run --frozen pytest tests/e2e -q` resolves without `--extra dev`. `uv lock` + `uv sync
  --frozen` re-locked cleanly (113 packages resolved).

- [x] **`detector`, `limit`, `port` and the boolean flags are CLI/env-only, not TOML-settable**
      (nit — no action needed). `KNOWN_TOML_KEYS = frozenset(DEFAULTS)` (`config.py:127`) means a
      TOML file containing these is fatal. This matches DESIGN.md, which presents them only as
      flags (e.g. `--detector scripted` in the E6 row). Closed as correct-as-written; recorded only
      so a future reader does not mistake it for an omission.

## From build review iteration 2 (2026-09-15 22:17 UTC, `6377ac8`, chunk 3 / `catalog.py`)

Nothing below withholds approval. **F1 is the one to read before writing chunk 5** — it is a latent
correctness trap rather than polish, though it has no caller today so nothing is broken yet.

- [x] **F1 — `ensure_image`'s upsert refreshes `status`/`run_id` and defaults to `planned`, which a
      naive scanner would use to reset `done` rows** (high priority — act on it in chunk 5,
      `scan.py`). `ensure_image` (`src/animal_classifier/catalog.py:~660-700`) puts `status` and
      `run_id` in its `ON CONFLICT DO UPDATE` list and defaults `status=Status.PLANNED`. If chunk 5's
      scanner calls it unconditionally for every hash it walks, a second run over the same card
      silently resets every `done` row to `planned`; `plan_disposition` decides `SKIP_DONE` purely
      from `row.status`, so the whole card would be re-processed. That defeats the
      second-run-is-a-clean-no-op invariant from the *caller* side rather than via the `skipped`
      table, so DEFECT 1's upsert does not protect against it. No caller exists today (`grep` for
      `ensure_image` outside `catalog.py` returns nothing). Fix: either consult `plan_disposition`
      before calling `ensure_image`, or make `status`/`run_id` refreshable only under an explicit
      opt-in argument. Chunk 9's idempotency e2e test must cover it either way.
      **Closed in chunk 5**, before `scan.py` was written, by the second option *plus* a narrowing:
      on a re-seen hash `ensure_image` now refreshes only the columns the caller actually named (plus
      `last_updated`), and `status`/`run_id` join that list only under an explicit
      `refresh_state=True`. Verified with a real catalog: a `done` row written under run `r1`, then
      re-seen by `ensure_image(sha, run_id=r2)` with no opt-in, stayed `status=done run_id=r1
      label=lion` and `plan_disposition` still returned `skip_done`; the same call with
      `refresh_state=True` did move it to `planned`/`r2`, and `first_seen` was unchanged throughout.
      Chunk 9's idempotency e2e test still owns the end-to-end assertion.

- [ ] **F2 — `write_tx`'s unconditional `ROLLBACK` can mask the original exception** (low).
      `catalog.py:~470-480`: the `except BaseException` branch issues `ROLLBACK` before re-raising.
      If SQLite has already rolled the transaction back, that statement raises
      `cannot rollback - no transaction is active` and the real error is lost. Guard it so the
      original propagates.

- [ ] **F3 — read accessors return inconsistent types** (low, GUI-facing). `image()`, `run()` and
      `newest_run()` return frozen value objects (`ImageRow`, `RunRow`); `boxes()`, `candidates()`
      and `newest_override()` return raw `sqlite3.Row`. Fine inside the module. Decide before the GUI
      serializers land in chunks 20-22 so row-index access does not leak into the web layer.

- [ ] **F4 — `update_run_counts` is a silent no-op when every count is `None`** (nit).
      `catalog.py:~576-586` returns early rather than raising, so a caller that passes nothing gets a
      successful no-op. Defensible for an all-optional progress publisher; recorded so it reads as a
      decision, not an oversight.

- [ ] **F5 — `runs.state` has no CHECK constraint** (nit). `images.status` and `images.label_source`
      are CHECK-constrained; `runs.state` is not, so `RunState` is enforced only in Python. Faithful
      to §5.9, which declares the column without enumerating its values — recorded as spec fidelity,
      not drift.

- [ ] **F7 — DEFECT 1's fix has no executable proof yet** (informational, self-resolving at chunk 9).
      The `skipped` upsert, `replace_inference`'s replace-not-append semantics and the `first_seen`
      protection are all asserted by source reading; the 7 green tests exercise `--help` only.
      Recorded so nobody later mistakes "reviewed" for "tested".

## From build review iteration 3 (2026-09-15 22:40 UTC, `7ada6a0`)

Raised against `6e448c2` (`taxonomy/`). None of these withholds approval and none needs attention
before chunk 5.

- [ ] **F8 — `merged()` is order-dependent and its docstring is garbled** (nit).
      `taxonomy/labels.py:~430`: `combined.update(cub200().by_key)` means a future `cub200.csv` row
      keyed `bird` would silently shadow COCO's coarse `bird` class instead of raising. No collision
      exists today (verified: `merged size 210` = 10 + 200). The docstring sentence "The keys are
      disjoint apart from nothing at all" is also unreadable — say plainly that COCO contributes
      `bird` and CUB contributes species keys, and consider raising on an unexpected overlap.

- [ ] **F9 — `find_by_*` indexes silently keep the first duplicate** (low). `labels.py:~415-420`
      builds `_by_common` / `_by_scientific` with `setdefault`, while duplicate `key` and duplicate
      `label` both raise. Reachable for `scientific`: two rows at coarser ranks can legitimately
      share a genus or family name, and the second becomes unreachable by scientific-name lookup
      with no diagnostic. Decide whether that is intended before the GUI search box (chunks 20-22)
      depends on it.

- [ ] **F10 — `TaxonTable`'s private indexes are constructor parameters** (nit). `_by_common` and
      `_by_scientific` are declared as dataclass fields, so `TaxonTable(...)` takes underscore-named
      keyword arguments. Harmless — `load_table` is the only constructor — but it reads as a leak.

- [ ] **F11 — `Taxon.scientific` type/storage mismatch is undocumented at the boundary** (nit). The
      CSV column holds `""` for unknown; the dataclass holds `None`. The conversion is in exactly one
      place (`scientific or None`, `labels.py:~430`), which is correct. Write the invariant down
      before the GUI serializers land so nobody re-introduces `""` on the way out.

- [ ] **F12 — `taxonomy/` has no executable proof yet** (informational, self-resolving at chunk 18).
      Neither `catalog.py` nor `taxonomy/labels.py` is reachable from a `--help` invocation, so the
      7 green tests do not touch either. This chunk's correctness is currently evidenced by the
      review's spot-check (`chuck_wills_widow`, `arctic_tern`, `brewers_blackbird`, 210/210 labels
      legal) and `PROGRESS.md`'s ad-hoc runs — not by anything in `docs/test-report.md`. E8's
      label-directory assertion at chunk 18 is what converts this to a regression test. Same shape
      as F7.

- [ ] **F13 — `cub200.csv` ↔ `classes.txt` order equality is asserted only ad hoc** (medium — do it
      at chunks 15-18). `impl-status.json` and `PROGRESS.md` record that the 200 keys match
      `CUB_200_2011.tgz`'s `classes.txt` exactly and in order, from a one-off `python -c`. That
      ordering is load-bearing: a silent reordering would mislabel every bird while every label
      stayed syntactically legal. Assert it where the `.acmodel` label list is loaded.

- [ ] **F14 — `config.py:14` mentions `min_box_area` in a negation** (nit, no action). The docstring
      sentence asserts that no such knob exists; it is the documented absence of the gate, not the
      gate. Recorded only so a future grep-based audit of the "no area floor" invariant does not
      misread the guard as a violation.
