# Test report — build complete, full expectation list covered

Captured on the locked environment with `uv run --frozen`. The suite is e2e-only
(§11); every test drives a real entry point (the CLI as a subprocess, or the GUI app
through an in-process ASGI client).

## Lint gate

```
$ uv run --frozen ruff check
All checks passed!
```

Covers `src`, `scripts` **and** `tests` — a test that lints cleanly is one whose
fixtures and bindings actually resolve (an unused-variable finding in `tests/` caught
a real `NameError` during this build).

## e2e suite — 147 tests, 0 failures, 0 xfail

Run in windows (the CPU-only sandbox caps a single command's wall clock); totals are
the sum.

```
$ pytest -m "not slow" -k "cli_surface or e02 or e05 or e06 or e26 or e11 or e16 or e17 or e22 or e01 or e03"
64 passed

$ pytest -m "not slow" -k "e13 or e14 or e15 or e18"
23 passed

$ pytest -m "not slow" -k "e19 or e20 or e21 or e23 or e24 or e25"
37 passed

$ pytest -m "slow" -k "not real_coco and not dominance_real"
15 passed

$ pytest tests/e2e/test_e04_dominance_real_coco.py          # real MegaDetector, 52 images
3 passed in 109s

$ pytest tests/e2e/test_e07_real_coco_species.py            # real finetune + identity
5 passed
```

**Total: 147 passed (124 fast + 23 slow), 0 failed, 0 xfailed.**

## The real-data results

### E4 — dominance on real COCO with real MegaDetector v5a

Frozen lists built by `scripts/build_coco_dominance.py` from
`instances_val2017.json`, val-bucket only, `iscrowd` dropped. Thresholds and list
lengths are read from the JSON; there are no numeric literals in the test.

| bucket | GT ratio | images | required | measured |
|---|---|---|---|---|
| `high` (should not be `multiple`) | > 3.0 | 27 | ≥ 22 | **26** |
| `low` (should be `multiple`) | < 1.3 | 25 | ≥ 20 | **21** |

### E7 — real transfer learning and species identity

`build_coco_manifest.py` (7 classes, `--split train`) → `train efficientnet_b0
--input-size 128` with the timm backbone → `eval --split val` → `classify --detector
megadetector --species-model`. Support: **1,625 train / 349 val** crops, exactly
§11.2's numbers.

- **`val_top1 = 0.797`** against §11.2's gate of **≥ 0.55** (chance 0.143,
  majority-class baseline 0.229)
- **identity: 18/20** frozen single-animal photos received their ground-truth species
  through the real CLI, against a gate of **≥ 16**

The two misses are the documented failure modes, not pipeline faults: one photo where
MegaDetector found only `person` boxes (→ `landscape` per §5.4; the annotator saw a
bear it did not), and one `bear` classified as `cattle` — `bear` is the smallest class
(60 train crops) and §11.2 already declines to assert its per-class recall.

## Coverage of the frozen expectation list (§11.2)

E1, E2, E3, **E4 (real)**, E5, E6, **E7 (real)**, E8, **E9**, E10, E11, **E12**, E13,
E14, E15, E16, E17, E18 (incl. the orientation-6 frame leg), E19 (incl. all three
modes and the contention 409), **E20**, **E21**, E22, E23, E24, E25, E26 — all
present and green.

## Bugs found and fixed by the tests during the build

1. **Same content filed twice** (E14) — two byte-identical copies on one card were one
   `images` row but two destination files; the within-a-run dedup was missing.
2. **Backbone head shape mismatch** (E7) — `load_state_dict(strict=False)` still raises
   on a *shape* mismatch, so the pretrained 1000-class head crashed every finetune.
3. **`--reclassify` clobbered human labels** (E19) — a re-run overwrote a GUI-corrected
   label back to the model's guess.
4. **`--ignore-overrides` destroyed the override** (E20) — the re-apply path keyed off
   `images.label_source`, so once the flag demoted a row the human decision could never
   return. It now reads the append-only `overrides` table, which is the durable record.
5. **GUI re-tag did not fail fast** (E19 modes) — it routed through `classify`'s
   0.5/1/2/4/8 s backoff, so a contended re-tag returned 200 after ~10 s instead of the
   deterministic 409 §5.9 requires. `Catalog` now takes `retry_writes`.
6. **Manifest paths were CWD-relative** (E7, E10) — `load_manifest` resolves relative
   paths against the manifest's own directory, so both builders emitted paths that did
   not resolve. Absolute now.

Five of the six were found by the test written for the behaviour, not by review.

## Recorded deviations

- **`coco_species.json` re-frozen at `min_area_frac = 0.10`** rather than §11.1's 0.20.
  Re-measured: at 0.20 only **9** val-bucket single-animal images exist, too thin to
  assert identity on; at 0.10 there are exactly the **20** §11.1 intends. The test reads
  the length from the JSON, so nothing is hard-coded — this is the design's own
  re-freeze rule, used as intended.
- **E4's measured list sizes (27/25) differ from §11.1's recorded 22/29.** The builder
  emits what the data contains and the test asserts `≥ ceil(0.8 × len)`, which is
  exactly why §11.1 forbids hard-coded lengths.
- **`--input-size 128`** in E7 is §7.3's stated test-budget trade-off against the
  shipped 224, not a silent weakening.
