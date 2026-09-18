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

## From the chunk-5 gate review (`scan.py`, 2026-09-15 23:00 UTC)

None of F15–F20 withheld approval; the only blocking finding was the progress counter (5/26).

- [ ] **F15 — the exit class for benign skips deviates from `DESIGN.md:1211`'s literal wording**
      (medium — resolve it in an assertion, not in prose). §10.1 says "`4` completed with per-image
      failures **or skips**", while `scan.py`'s `exit_code_for` returns 0 when only `BENIGN_REASONS`
      occurred. The deviation is correct: `docs/design-review.md:115` finding 5 is titled "exit code
      4 vs 0 for benign skips is self-contradictory, and E2/E16 cannot both pass", and §10.1's own
      table marks `video`/`format_disabled`/`too_large`/`symlink` as "not an error"/`DEBUG`. A card
      full of videos is a successful run. Action: E2 must assert exit **0** on a card whose only
      skips are benign, and E16 must assert exit **4** because its fixture contains `zero_bytes`,
      a truncated JPEG and `symlink_escape` — so the semantics live in a test, not just in a
      module docstring.

- [ ] **F16 — `_judge_file`'s rule order is not literally §5.1's table order** (nit, docstring only).
      `video` and `raw_not_enabled` are tested before `unsupported_extension`/`format_disabled`
      (`scan.py:341-356`). This is the better behaviour — `.mp4` and `.cr2` belong to no `formats`
      family, so the table's order would record the *less* specific `unsupported_extension` — and
      both candidates are benign, so the exit class is identical. Only the module docstring's claim
      of "the fixed precedence order documented there" overstates the correspondence.

- [ ] **F17 — `scan.py` has no executable proof at this commit** (informational, self-resolving at
      chunks 9 and 12; same shape as F7/F12). None of the 13 skip reasons, the directory pruning,
      the `onerror` drain ordering, or `exit_code_for` is reachable from a `--help` invocation, so
      the 7 green tests do not touch this chunk's subject at all. Its correctness currently rests on
      the gate review's reading plus `PROGRESS.md`'s ad-hoc runs. E6's full form (chunk 9) and E16
      (chunk 12) are what convert this into regression coverage.

- [ ] **F18 — an unreadable source *root* exits 4, where §10.1 wants a fatal 3** (medium — do it in
      chunk 9). `walk()` checks `root.is_dir()` (`scan.py:239`) but not readability, so a card root
      that exists and is a directory yet cannot be listed arrives through `_on_error` as a single
      `Skipped(unreadable)` — abnormal, exit 4. §10.1's row reads "source missing / not a dir /
      **not readable** → fatal, exit 3". Nothing is silent either way, so this is an exit-code
      nuance, not a data risk. Fix where `classify` wires the exit codes: probe the root once
      up front and raise `ConfigError`.

- [ ] **F19 — the `pending_errors` drain is duplicated and O(n)** (nit). `while pending_errors:
      yield pending_errors.pop(0)` appears at `scan.py:266-268` and again at `281-282`. A `deque`,
      or a `yield from list(...)` followed by `clear()`, says the same thing once. Keep the comment
      explaining *why* the drain sits at the top of the loop body (it preserves walk order) —
      that is the non-obvious part.

- [ ] **F20 — two spellings of one predicate** (nit). `Skipped.benign` (property) and module-level
      `is_benign()` both answer "does this reason leave the run successful?", and the property has
      no caller today. Keep `is_benign()`: its string-accepting form is load-bearing, because it
      classifies a reason read back out of the catalog and raises `ValueError` on an unrecognised
      value rather than defaulting to benign.


## From an unscheduled review of chunk 6 (`images.py`, 2026-09-18)

Raised against `940115b` by reading the module and probing it with inputs
`scripts/probe_images.py` did not build. **F21 and F22 are real defects, not polish, and both
are fixed in this commit** — each one silently corrupted a value that feeds the dominance
rule or the catalog, which is why neither could be left to a later chunk. F23–F26 are
observations only.

- [x] **F21 — a GPS hemisphere ref spelled as `bytes` flipped the coordinate into the wrong
      hemisphere** (high). `_to_degrees` decided the sign with
      `str(ref).strip().upper().startswith(negative_ref)`. `GPSLatitudeRef` is ASCII(2) per
      the EXIF spec, but firmware that declares it UNDEFINED(7) makes Pillow's `Image.Exif`
      yield `b"S"` — and `str(b"S")` is `"b'S'"`, which starts with neither `N` nor `S`, so a
      southern-hemisphere photo was stored at **+1.5° instead of −1.5°**. Verified before the
      fix: `_to_degrees(path, (1, 30, 0), b"S", "S")` returned `1.5`, and
      `TiffImagePlugin.ImageFileDirectory_v2` with `tagtype[1] = 7` confirms Pillow really
      does hand back `b"S"` for that tag. The same fall-through also defaulted an **absent,
      empty or numeric** ref to the positive hemisphere. That is the lie §10.2 forbids, in the
      one field where it is most damaging: a hemisphere flip is not an imprecise location but
      a confident wrong one, it is what `ebird_enrich` uses to down-rank species by locality
      (PLAN.md "Birds"), and for a tool aimed at African safaris the wrong side of the equator
      is the common case, not the exotic one.
      **Fixed**: the letter is now extracted by `_gps_ref_letter`, which accepts `str` and
      `bytes`, strips EXIF's trailing NUL, and returns `None` for anything else; `_to_degrees`
      takes the positive letter as well as the negative one and drops the coordinate with a
      `WARNING` unless the ref decodes to one of the two. The sign is now validated as
      strictly as the magnitude. Nine ref shapes are pinned in `probe_images.py`.

- [x] **F22 — a non-finite box coordinate became a silent full-frame crop; an off-frame box
      produced an inverted region** (high). Two separate defects in `crop`, both found by
      passing boxes a malfunctioning detector could emit:
      - NaN loses every comparison, so `max(0.0, nan)` returned `0.0` and `min(500.0, nan)`
        returned `500.0`. A box of `(nan, 0, 10, 10)` on a 500×500 frame therefore clipped to
        `region=(0, 0, 500, 11)` with `degenerate=False` — a full-width strip, handed to the
        classifier as an animal, contributing a meaningless `area_frac` to the §5.7 dominance
        rule. Fully silent: no warning, no skip row. `config.py:497` already rejects NaN and
        inf for every float knob, so the project's standard for detector output was simply
        lower than its standard for config. Now raises `ValueError`, deliberately *not* an
        `ImageDecodeError` — a NaN box is a systemic detector fault, and routing it through
        the per-image `except` would record one skip while the rest of the card kept taking
        garbage.
      - Bounding each edge on one side only (`max(0, …)` near, `min(width, …)` far) inverted
        the region for a box that misses the frame entirely: `(600, 600, 700, 700)` on a
        500 px frame gave `region=(592, 592, 500, 500)`, extent −92×−92. It was correctly
        degenerate, so no pixels were cropped, but the region is still returned and
        `boxes.x0..y1` still persists it, and the GUI overlay would be asked to draw a
        negative-extent rectangle. Both edges are now clamped into the frame interval with
        `_clamp`; clamping is monotonic, so `left_f <= right_f` holds for every input and the
        off-frame case collapses to a zero-extent region on the frame edge.

- [ ] **F23 — there is no automated lint or type gate, and three `noqa` directives are
      already dead** (medium — decide before the tree doubles in size). Neither `ruff` nor
      `mypy` is in `[dependency-groups] dev`, there is no `[tool.ruff]` config and no
      `.github/` CI, yet the source carries `# noqa: PLC0415` (`images.py:294`),
      `# noqa: S603` (`tests/e2e/conftest.py:38`) and `# noqa: E402`
      (`scripts/probe_md_inference.py:17`) — rules that are not enabled, so `ruff check`
      reports all three as `RUF100` unused directives. The intent to lint is in the code; the
      gate is not. A `ruff check --select E,F,I,B,SIM,PL,S,RUF,DTZ,PYI,EXE,UP,C4,RET,ARG` run
      finds 85 items today, mostly `E501` (37) and `B008` (8, all typer's
      `Argument`/`Option`-in-default idiom, which wants the `Annotated` spelling or a
      per-file ignore). Worth picking a ruleset deliberately, pinning it in `pyproject.toml`,
      and adding it to the gate command — a 26-chunk build reviewed by reading diffs is
      exactly the shape of project where a mechanical check pays for itself. Two genuine
      items in that list: `DTZ007` (`images.py`'s `strptime` is naive — correct for EXIF,
      which has no zone, but say so) and `PLR0124` (`config.py:497`'s `value != value` NaN
      idiom, which reads as a typo and is clearer as `math.isnan`).

- [ ] **F24 — `_decode_raw` allocates the full raster before the pixel cap is applied,
      contradicting the module docstring** (low). `decode`'s docstring promises
      `too_large_pixels` is "judged from the header, so a decompression bomb is refused
      without ever being allocated", which holds for the Pillow path (`_refuse_oversized` runs
      on `opened.size` before `convert`). On the RAW path `_refuse_oversized` runs on
      `array.shape` *after* `source.postprocess()` has already decoded and allocated. Bounded
      in practice by `max_file_bytes` and by `--raw` being opt-in, so this is an accuracy
      problem in the docstring more than a live risk — `rawpy` exposes `sizes` before
      postprocessing if the guarantee is worth making real.

- [ ] **F25 — a multi-frame file silently becomes its first frame** (low, undocumented).
      An animated GIF and a multi-page TIFF both decode to frame 0 with no warning. That is
      almost certainly the behaviour you want for a photo pipeline, and the exotic-mode
      handling around it is genuinely solid (palette+transparency, 16-bit grayscale, CMYK and
      LA all convert to RGB without complaint, verified). Recorded only because "which frame
      is the frame" is a question §5.2 does not answer, and a burst-mode TIFF stack would
      quietly lose frames 1..n.

- [ ] **F26 — every source file is read twice, once to hash and once to decode** (low, perf).
      `sha256_file` and `decode` each open the file through `open_source`, so a 500 MB RAW
      within `max_file_bytes` is pulled off the SD card twice. Inherent to hashing raw bytes
      rather than pixels (§5.2, and the right call — the hash must be decoder-independent),
      but the two passes could share one read when the decode worker lands in chunk 9/10.
      Mentioned because SD card throughput, not inference, will dominate a real card's
      wall-clock time.
