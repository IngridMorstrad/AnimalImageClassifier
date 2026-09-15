# E2E test report

**Run timestamp:** 2026-09-15 22:14 UTC
**Branch:** `feat/safari-classifier` @ `c87a690`

## 1. Commands run

```
cd /projects/sandbox/AnimalImageClassifier && uv run pytest tests/e2e -v
```

`uv sync` was **not** needed — the environment resolved and ran as-is. Two supporting read-only
commands were run to characterise the result (no source or test file was edited):

```
find /projects/sandbox/AnimalImageClassifier/tests -type f -name '*.py' | sort
git -C /projects/sandbox/AnimalImageClassifier status --short
```

## 2. Verbatim output

`uv run pytest tests/e2e -v` — complete, untruncated:

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /projects/sandbox/AnimalImageClassifier/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /projects/sandbox/AnimalImageClassifier
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 7 items

tests/e2e/test_cli_surface.py::test_top_level_help_lists_every_command PASSED [ 14%]
tests/e2e/test_cli_surface.py::test_no_command_offers_an_area_floor[classify] PASSED [ 28%]
tests/e2e/test_cli_surface.py::test_no_command_offers_an_area_floor[gui] PASSED [ 42%]
tests/e2e/test_cli_surface.py::test_no_command_offers_an_area_floor[train] PASSED [ 57%]
tests/e2e/test_cli_surface.py::test_no_command_offers_an_area_floor[eval] PASSED [ 71%]
tests/e2e/test_cli_surface.py::test_no_command_offers_an_area_floor[export-trainset] PASSED [ 85%]
tests/e2e/test_cli_surface.py::test_no_command_offers_an_area_floor[verify] PASSED [100%]

============================== 7 passed in 1.00s ===============================
```

Process exit code: **0**.

Test-file inventory under `tests/` — complete output:

```
/projects/sandbox/AnimalImageClassifier/tests/e2e/conftest.py
/projects/sandbox/AnimalImageClassifier/tests/e2e/test_cli_surface.py
```

`git status --short` printed nothing (clean tree) before this step's own writes.

## 3. Summary

| Result | Count |
|---|---|
| passed | **7** |
| failed | 0 |
| errored | 0 |
| skipped | 0 |
| **collected** | **7** |

**No failures and no collection errors**, so there is no failing test name, assertion message, or
responsible source file to report. Exit code 0, up from the previous run's 5 (empty suite).

**What these 7 tests actually prove — and what they do not.** All 7 come from one file,
`tests/e2e/test_cli_surface.py`, and all 7 exercise `--help` output only:

- `test_top_level_help_lists_every_command` — `--help` exits 0 and names all six commands
  (`classify`, `gui`, `train`, `eval`, `export-trainset`, `verify`).
- `test_no_command_offers_an_area_floor[<command>]` (×6) — per-command `--help` exits 0 and
  contains none of 10 spellings of an absolute area floor (`min-box-area`, `min_box_area`,
  `min-area`, `min_area`, `area-floor`, `area_floor`, `min-animal-area`, `min_animal_area`,
  `min-box-frac`, `min_box_frac`).

That is the executable guard on DESIGN.md invariant I2 (no `min_box_area`; `dominance_ratio` is the
only size gate), which is a standing user requirement — so it is worth having green this early. But
it is the CLI-surface leg of **E6 only**. This run does **not** exercise detection, classification,
the dominance rule, materialization, the catalog, the GUI, training, or the bird providers. Per the
file's own docstring, E6's full form (a real `classify` run over the fixture card asserting the
dominance rule decides alone) lands in chunk 9. No claim about pipeline behaviour is supported by
this output.

**Progress context (from `docs/impl-status.json`, not inferred).** `done_items: 3` of
`total_items: 26`, `current_chunk: "4. taxonomy/: slug(), LABEL_RE, RESERVED_LABELS and the static
name tables"`. 7 passing CLI tests at chunk 3 is the expected state; E1–E26 are not yet written.

**Spec compliance (e2e only).** No violation. `tests/` contains exactly two `.py` files, both under
`tests/e2e/` (`conftest.py`, `test_cli_surface.py`). No unit-test file exists anywhere under
`tests/` outside `tests/e2e/`, and `pyproject.toml` pins `testpaths = ["tests/e2e"]`. Nothing was
deleted.

**Blocked:** nothing in this step. The suite runs, collects, and passes. Coverage breadth is a
sequencing fact (chunk 3 of 26), not a blocker on testing.
