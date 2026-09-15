# Build review — iteration 3 (2026-09-15 22:40 UTC)

**Range reviewed:** `b8ec047..HEAD` — `6e448c2` (`taxonomy/`) and `7ada6a0` (test report) on
`feat/safari-classifier`.
**Spec of record:** `docs/DESIGN.md` (frozen; not reviewed, not edited).
**Verdict:** CHANGES_REQUESTED — **1 blocking finding**, and it is the same sequencing finding as
iterations 1 and 2: `docs/impl-status.json` has `complete: false` at 4 of 26 chunks. **Nothing that
landed in this iteration needs to be undone or revisited.** The code that landed is correct, and the
central defect this chunk existed to fix is now fixed and empirically demonstrated.

## 1. What landed

| Area | Change |
|---|---|
| `src/animal_classifier/taxonomy/labels.py` | New, 446 lines. `slug()`, `LABEL_RE`, `RESERVED_LABELS`, `validate_label()`, `cub_key_from_dirname()`, `Rank`, `Taxon`, `TaxonTable`, `load_table()`, and the cached `coco_animals()` / `cub200()` / `merged()` accessors. |
| `src/animal_classifier/taxonomy/__init__.py` | New, 55 lines. Re-exports the whole surface; documents that this is the import point, not `.labels`. |
| `src/animal_classifier/taxonomy/data/coco_animals.csv` | New, 10 rows + header. |
| `src/animal_classifier/taxonomy/data/cub200.csv` | New, 200 rows + header. |
| `.gitignore` | `data/` → `/data/`. Anchored so the unanchored pattern stops matching `src/animal_classifier/taxonomy/data/`. |
| `pyproject.toml` | `[tool.hatch.build.targets.wheel] artifacts = [".../taxonomy/data/*.csv"]`. |
| `docs/*` | test-report (real run), PROGRESS (+181 lines), impl-status (`done_items` 3 → 4), IMPL-PLAN chunk 4 checked. |

## 2. Does it match the frozen design and the user's stated intent?

Yes, on every point I could check against the diff.

**DEFECT 2 — raw / digit-prefixed class names as output directories — is fixed.** This is the
category the gate calls out explicitly (`022_chuck_will_widow` must never be a folder), so I spent my
one permitted spot-check here rather than accepting a source reading. Loading both real tables
through the real code:

```
cub rows 200 coco rows 10
chuck_will_widow -> chuck_wills_widow
artic_tern       -> arctic_tern
brewer_blackbird -> brewers_blackbird
forsters_tern    -> forsters_tern
invalid/digit-prefixed/reserved labels: []
cub_key_from_dirname('022.Chuck_will_Widow') -> chuck_will_widow
merged size 210
label_for('nope') raises ConfigError
```

That is the exact transformation the gate names, produced by the shipped tables: the ordinal prefix
is stripped into a *key*, and the *label* comes from the authoritative display name. All 210 labels
across both tables satisfy `LABEL_RE`, none begins with a digit, none collides with a reserved
outcome. CUB's own misspelling (`141.Artic_Tern`) is preserved in the key — correct, that is what
the dataset is keyed by — while the folder is corrected to `arctic_tern`. The key/label split in
`labels.py:20-45` is exactly the right shape for this, and `_reject_raw_dataset_id`
(`labels.py:~300`) turns a future regression into a startup `ConfigError` rather than a directory on
disk.

**No `min_box_area` and no absolute area floor.** `git grep` across all tracked files for ten
spellings of an area floor returns exactly one hit outside the docs: `config.py:14`, a docstring
sentence asserting that no such knob exists. That is the documented *absence* of the gate, not the
gate. `dominance_ratio` remains the only size gate. The 6 parametrised CLI-surface tests continue to
assert this executably.

**Fail-loud on missing required values (I7) is honoured, not eroded.** `TaxonTable.__getitem__`
raises `ConfigError` naming the file and the fix; `label_for` / `common_for` / `rank_for` all route
through it, so a missing class key can never become a placeholder directory. `slug()` raises rather
than returning a sanitized guess. `load_table` validates the header exactly, rejects blank rows,
wrong field counts, empty `key`, empty `common`, empty `class`, an unknown `rank`, duplicate keys,
and — the check I most wanted to see — **duplicate labels**, which would otherwise silently merge two
species into one folder with no later stage able to notice (`labels.py:~395`). The one field allowed
to be absent, `scientific`, is genuinely optional in the contract and is stored as `None`, with the
reasoning recorded in the module docstring. That is an honest absence, not a substituted default.

**The `.gitignore` / `hatchling` fix is a real data-loss catch, not bookkeeping.** An unanchored
`data/` matched `src/animal_classifier/taxonomy/data/`, and hatchling honours VCS ignore files when
selecting wheel contents. Left alone, the CSVs would have been dropped from both `git add -A` and
the wheel, and since no taxonomy API is reachable (§13.3) the installed package would have failed at
runtime with no local reproduction. The commit anchors the pattern *and* names the artifacts
explicitly — belt and braces, correct.

## 3. What the real test output proves

`docs/test-report.md` captures a verbatim, untruncated `uv run pytest tests/e2e -v` at `6e448c2`:
**7 collected, 7 passed, 0 failed, 0 errored, 0 skipped, exit code 0**, 0.94s. No test was claimed
green without captured output.

The report is also honest about its own reach, which I checked rather than took on faith:

- All 7 tests are `--help` assertions from one file, `tests/e2e/test_cli_surface.py`. They prove the
  CLI-surface leg of E6 and the standing "no area floor" invariant. They exercise no detection, no
  dominance rule, no materialization, no catalog, no GUI, no bird provider.
- **`taxonomy/` has zero e2e coverage at this commit** — `catalog.py` and `taxonomy/labels.py` are
  both unreachable from a `--help` invocation. The report says so explicitly. Recorded as F12 below;
  E8's label-directory assertion is scheduled for chunk 18, and my spot-check above is a review
  artifact, not a regression test.
- No test exists anywhere outside `tests/e2e/`. I confirmed independently: `find tests -name '*.py'
  -not -path 'tests/e2e/*'` returns nothing; the tree holds exactly `conftest.py` and
  `test_cli_surface.py`; `pyproject.toml` pins `testpaths = ["tests/e2e"]`. No unit tests. Nothing
  was deleted to make the suite green.

## 4. Blocking findings (1)

### B1 — Implementation is 4 of 26 chunks complete; `impl-status.json` has `complete == false`

**File:** `docs/impl-status.json`

This is a sequencing finding, carried forward. `complete: false`, `done_items: 4`,
`total_items: 26`, `current_chunk: "5. scan.py: ..."`; `IMPL-PLAN.md` checkbox counts agree (4
checked, 22 unchecked). Approval requires `complete == true`, so the loop must keep going. **No part
of this iteration's diff is implicated** — `labels.py`, the two CSVs, `.gitignore` and
`pyproject.toml` are all correct as written.

Behavioural gates still entirely unproven by executable tests: E6's full form (a real `classify`
over the fixture card proving `dominance_ratio` decides alone) at chunk 9; the
second-run-is-a-clean-no-op test that finally exercises `record_skip`'s `ON CONFLICT(path)` upsert
end to end (chunk 9); `materialize.py`'s atomic writes and the
`--link` / `--hardlink` / `--dry-run` / `--reclassify` contract (chunk 11); E22 fail-loud config;
the species and bird heads at chunks 15-19, where a low-confidence result must file as `unknown`
rather than a guessed species, and where E8 must assert the human-readable output directories that
this chunk made possible.

**Immediate next action, unchanged and now overdue:** before writing `scan.py`, act on `FOLLOWUPS.md`
**F1**. `ensure_image`'s upsert refreshes `status`/`run_id` and defaults to `PLANNED`, so an
unconditional call from the new scanner would reset `done` rows to `planned` and defeat the
second-run-is-a-no-op invariant from the caller side, where DEFECT 1's upsert cannot protect it.

## 5. Non-blocking observations

All appended to `docs/FOLLOWUPS.md` as F8-F14. None withholds approval; none needs attention before
chunk 5.

- **F8** — `merged()`'s docstring sentence "disjoint apart from nothing at all" is garbled, and the
  merge is order-dependent: `cub200()` overwrites `coco_animals()`, so a future CUB row keyed `bird`
  would silently shadow COCO's without an error. No collision exists today (verified: 10 + 200 =
  `merged size 210`).
- **F9** — `find_by_common` / `find_by_scientific` build their indexes with `setdefault`, so a
  duplicate silently keeps the first row while duplicate keys and labels raise. Reachable for
  `scientific` (two rows can legitimately share a genus/family name at coarser ranks).
- **F10** — `TaxonTable`'s `_by_common` / `_by_scientific` are dataclass fields, so the underscore
  names appear in the constructor signature.
- **F11** — `Taxon.scientific` is typed `str | None` while the CSV column is `""`; the conversion
  happens in one place (`scientific or None`), so document the invariant before the GUI serializers
  land.
- **F12** — `taxonomy/` has no e2e coverage at this commit (same shape as F7 for `catalog.py`).
  Correctness is currently evidenced by this review's spot-check and `PROGRESS.md`'s ad-hoc runs.
  Self-resolving at chunk 18 (E8).
- **F13** — the claim that `cub200.csv`'s 200 keys match `CUB_200_2011.tgz`'s `classes.txt` exactly
  and in order rests on an ad-hoc run recorded in `PROGRESS.md`. Chunks 15-18 should assert it where
  the artifact's label list is loaded, since a silent reordering would mislabel every bird.
- **F14** — `config.py:14` names `min_box_area` inside a sentence asserting it does not exist.
  Recorded only so a future grep-based audit does not misread the guard as the gate.
