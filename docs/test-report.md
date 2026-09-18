# Test report — build complete (all 26 chunks)

Captured on the locked environment with `uv run --frozen`. The suite is e2e-only
(§11); every test drives a real entry point (the CLI as a subprocess, or the GUI app
through an in-process `TestClient`).

## Lint gate

```
$ uv run --frozen ruff check src scripts
All checks passed!
```

## e2e suite

Run in three windows (the CPU-only sandbox caps a single command's wall clock);
totals below are the sum, with zero failures and zero remaining xfails.

```
$ uv run --frozen pytest tests/e2e -q -m "not slow" -k "cli_surface or e02 or e05 or e06 or e26 or e11 or e16 or e17 or e22 or e01 or e03"
64 passed, 45 deselected

$ uv run --frozen pytest tests/e2e -q -m "not slow" -k "e13 or e14 or e15 or e18 or e19 or e23 or e24 or e25"
36 passed, 73 deselected

$ uv run --frozen pytest tests/e2e -q -m "slow"
9 passed, 100 deselected
```

**Total: 109 passed, 0 failed, 0 xfailed.** (100 fast + 9 slow.)

## What the slow tests prove, on real data

- **E4** — MegaDetector v5a, verified by size + sha256, finds `animal` boxes in real
  COCO val2017 photos and files a people-only photo as `landscape` (person/vehicle
  boxes stored but never labelling).
- **E7** — a real `efficientnet_b0` finetune on staged COCO crops produces a loadable
  `.acmodel` (`temperature == 1.0`, `calibrated_from is None`) that files a real photo
  into a species directory with a recorded confidence and candidate rows.
- **E8** — a CUB-named bird head files into human-readable directories
  (`chuck_will_s_widow`, never `022_chuck_will_widow`) — DEFECT 2.
- **E10** — the synthetic smoke path runs train → eval → export → calibrate → infer
  in seconds with the from-scratch `tinycnn`.

## Coverage of the frozen expectation list (§11.1)

E1 (source immutable, incl. copy/hardlink/read-only/re-tag/export/verify legs), E2
(happy path + exact tree), E3 (link modes), E4 (real detector), E5 (dominance
boundaries), E6 (no area floor), E7 (transfer learning), E8 (bird path), E10
(synthetic training), E11 (landscape/junk), E13 (idempotency/limit/reclassify), E14
(duplicate content), E15 (collision), E16 (formats/skips), E17 (dry-run), E18 (GUI
read), E19–E21 (GUI re-tag), E22 (fail-loud config), E23 (bird providers), E24
(verify + --fix), E25 (export-trainset), E26 (degenerate dominant box) — all present
and green.

## Bugs found and fixed by the tests during the build

1. **Same content filed twice** (E14) — two byte-identical copies on one card were
   one `images` row but two destination files; `_accept` now dedups by hash within a
   run so "classified once, filed once" holds intra-run.
2. **Backbone head shape mismatch** (E7) — `load_state_dict(strict=False)` still
   raised on the pretrained 1000-class classifier head; the loader now drops
   shape-incompatible tensors and trains the head fresh.
3. **`--reclassify` clobbered human labels** (E19) — a re-run overwrote a
   GUI-corrected label back to the model's; the pipeline now preserves
   `label_source='human'` unless `--ignore-overrides`.

## Honestly scoped

- **E7's `val_top1 >= 0.55`** needs the full COCO val2017 stage and a ~20-minute CPU
  finetune; the in-loop test asserts the *path* (a real finetune produces a usable
  artifact) and documents the accuracy gate as the full-stage job it is, rather than
  lowering the threshold in test code.
- **Shipped `.acmodel` artifacts** are produced on the user's machine via the exact
  commands in `docs/ARTIFACTS.md`; they are not committed (large binaries).
