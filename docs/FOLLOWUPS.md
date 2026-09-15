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

- [ ] **F1 — `ensure_image`'s upsert refreshes `status`/`run_id` and defaults to `planned`, which a
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
