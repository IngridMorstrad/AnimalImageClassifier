# E2E test report

**Run timestamp:** 2026-09-15 22:53 UTC
**Branch:** `feat/safari-classifier` @ `4fdc624`

## 1. Commands run

```
cd /projects/sandbox/AnimalImageClassifier && uv run pytest tests/e2e -v
```

`uv sync` was **not** needed — the environment resolved and ran as-is. The suite was run **twice**
(once with `-v`, once with `-q`) to check the result is reproducible; both runs reported 7 passed,
exit 0. These read-only commands were run to characterise the result (no source or test file was
edited):

```
git -C /projects/sandbox/AnimalImageClassifier status --short
git -C /projects/sandbox/AnimalImageClassifier log --oneline -3
find /projects/sandbox/AnimalImageClassifier/tests -type f ! -path '*/e2e/*'
find /projects/sandbox/AnimalImageClassifier/src -type f -name '*.py' | sort
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

============================== 7 passed in 0.97s ===============================
```

Process exit code: **0**.

Confirmation run, `uv run pytest tests/e2e -q` (tail):

```
.......                                                                  [100%]
7 passed in 0.97s
```

Test-file inventory outside `tests/e2e/` — the `find` above printed **no output at all** (no such
file exists). `git status --short` printed nothing (clean tree) before this step's own writes.

## 3. Summary

| Result | Count |
|---|---|
| passed | **7** |
| failed | 0 |
| errored | 0 |
| skipped | 0 |
| **collected** | **7** |

**No failures and no collection errors**, so there is no failing test name, assertion message, or
responsible source file to report. Exit code 0. Identical test set and identical result to the
previous recorded run at `6e448c2` (7 passed); wall time 0.97s vs 0.94s.

**What these 7 tests actually prove — and what they do not.** All 7 come from one file,
`tests/e2e/test_cli_surface.py`, and all 7 exercise `--help` output only:

- `test_top_level_help_lists_every_command` — `--help` exits 0 and names all six commands
  (`classify`, `gui`, `train`, `eval`, `export-trainset`, `verify`).
- `test_no_command_offers_an_area_floor[<command>]` (×6) — per-command `--help` exits 0 and
  contains none of 10 spellings of an absolute area floor (`min-box-area`, `min_box_area`,
  `min-area`, `min_area`, `area-floor`, `area_floor`, `min-animal-area`, `min_animal_area`,
  `min-box-frac`, `min_box_frac`).

They are genuinely end-to-end in mechanism: `conftest.py` resolves the installed
`animal-classifier` console script from `PATH` and invokes it as a real subprocess, so the surface
under test is the one a user runs. That is the executable guard on DESIGN.md invariant I2 (no
`min_box_area`; `dominance_ratio` is the only size gate), a standing user requirement. But it is the
CLI-surface leg of **E6 only**. This run does **not** exercise detection, classification, the
dominance rule, materialization, the catalog, the scanner, the GUI, training, or the bird providers.

**Untested source at this commit (fact, from the `find` output above).** Eight modules exist under
`src/animal_classifier/`: `__init__.py`, `cli.py`, `config.py`, `errors.py`, `catalog.py`,
`scan.py`, `taxonomy/__init__.py`, `taxonomy/labels.py`. Only `cli.py`'s `--help` output is touched
by this suite. In particular `scan.py` (chunk 5, this commit's subject), `catalog.py` (chunk 3) and
`taxonomy/labels.py` (chunk 4) have **no e2e coverage in this run** — none is reachable from a
`--help` invocation. Their correctness is currently evidenced only by the ad-hoc verification
scripts recorded in `PROGRESS.md`, not by anything in this report. Per `test_cli_surface.py`'s
docstring and the plan, E6's full form lands in chunk 9 (where `scan.py` first gets executable e2e
coverage), every skip reason is covered by E16 in chunk 12, and the label-directory assertion (E8)
lands in chunk 18.

**Progress context (read from `docs/impl-status.json`, not inferred).** `done_items: 5` of
`total_items: 26`, `current_chunk: "6. images.py: decode, the one coordinate frame, sha256, blur,
crop"`. 7 passing CLI tests at chunk 5 is the expected state; E1–E26 are not yet written.

**Spec compliance (e2e only).** No violation. The explicit search for files under `tests/` outside
`tests/e2e/` returned nothing, `tests/e2e/` holds exactly `conftest.py` and `test_cli_surface.py`
(plus a `__pycache__/` directory), and `pyproject.toml` pins `testpaths = ["tests/e2e"]`. Nothing
was deleted.

**Blocked:** nothing in this step. The suite runs, collects, and passes reproducibly. Coverage
breadth is a sequencing fact (chunk 5 of 26), not a blocker on testing.
