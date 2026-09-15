# Follow-ups

Non-blocking observations raised during build review. Nothing here withholds approval.
Append only; note the review iteration that raised each item.

## From build review iteration 1 (2026-09-15 21:57 UTC, `c59a656`)

- [ ] **`output_root` is not symlink-resolved, so `_guard_nesting` can be evaded** (medium — fix by
      chunk 11, before `materialize.py` exists). `_resolve_source_root` uses `Path.resolve()`
      (`src/animal_classifier/config.py:524`) but `_resolve_output_root` uses `os.path.abspath`
      (`config.py:551`), which normalises `..` without following symlinks. An `output_root` that is
      a symlink into the SD-card tree passes all three checks in `_guard_nesting` and would then be
      written to, violating the read-only-card invariant. `Path(...).resolve()` is non-strict in
      3.12 and resolves existing ancestors of a not-yet-created directory, so the
      does-not-exist-yet tolerance that motivated `abspath` is not needed. Harmless today because
      nothing in the tree writes.

- [ ] **`pytest` is in `[project.optional-dependencies] dev` but the documented test command does
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

- [ ] **pytest exits 5 on an empty suite, which reads as failure to an exit-code gate** (nit,
      process). Already flagged by the e2e step in `docs/test-report.md`. Self-resolves when chunk 9
      lands the first test. Recorded so nobody debugs a phantom failure in the meantime.

- [x] **`detector`, `limit`, `port` and the boolean flags are CLI/env-only, not TOML-settable**
      (nit — no action needed). `KNOWN_TOML_KEYS = frozenset(DEFAULTS)` (`config.py:127`) means a
      TOML file containing these is fatal. This matches DESIGN.md, which presents them only as
      flags (e.g. `--detector scripted` in the E6 row). Closed as correct-as-written; recorded only
      so a future reader does not mistake it for an omission.
