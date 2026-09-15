# Cold design review — DESIGN.md iteration 2

Reviewed 2026-09-15 16:16–16:40 UTC, without the context that produced the design. Inputs:
`docs/DESIGN.md` (iteration 2, 1,101 lines), `docs/PLAN.md`, `docs/RECON.md`. Every numeric and
feasibility claim I could check, I re-measured myself in this sandbox rather than taking the
document's word for it (commands in [Verified assumptions](#verified-assumptions)).

**Verdict: CHANGES_REQUESTED — 1 HIGH, 13 MEDIUM, 6 NIT.**

None of the hard blocking conditions apply. Specifically, checked and clear:

| Blocking condition | Result |
|---|---|
| `min_box_area` / any absolute box-area floor | **Absent.** The only four mentions are explicit negations (§3, §5.7, I2, E6); `dominance_ratio` (1.6) is the only size gate, and it is a pure multiplication comparison |
| Unit tests or any layer other than `tests/e2e` | **Absent.** §2 and §11 state e2e only, no `tests/unit`, no assertions on private helpers |
| Dependence on an asset RECON records as unreachable | **None.** MegaDetector + timm backbones come from GitHub release assets, datasets from `s3.amazonaws.com` — all re-verified below. `huggingface.co`/LILA appear only as "blocked here, available on the user's machine"; `api.ebird.org` is opt-in and off by default |
| Writes/moves/renames/deletes in the source tree | **None.** Three independent guards (§5.1) plus I1 and E1's byte-for-byte snapshot. `verify --fix` is explicitly scoped away from the source |
| Silent default for a missing required value | **None found.** I7 plus §10.1/§10.2; `--device cuda` on a CUDA-less host, missing eBird key, missing artifact and `--calibrate` without `--out` are all fatal exit 3. The single degradation (`ebird_enrich` unreachable) is optional enrichment and is recorded as `provider_status='unreachable'` |
| Required capability omitted | **None.** MegaDetector v5a ✓ · own finetuned species model + training subsystem + `train` ✓ · three bird providers with `own_bird_head` default ✓ · label set `species\|multiple\|landscape\|junk\|unknown` ✓ · atomic temp+`os.replace` ✓ · copy/`--link`/`--hardlink` ✓ · sha256-keyed idempotent resumable ✓ · SQLite catalog ✓ · FastAPI + vanilla-JS GUI on `127.0.0.1:8765` with re-tag that moves files and writes `overrides` ✓ · typer CLI `classify\|gui\|train\|eval\|export-trainset\|verify` ✓ · JPEG/PNG/TIFF/HEIC default, RAW behind `--raw`, video always skipped ✓ |

The design is in good shape: the dependency contract, the detector load path, the dominance rule
and the training subsystem are all specified at implementation depth and backed by measurements
that reproduce exactly. What remains are unspecified behaviours that a coder would have to invent —
concentrated in the re-tag write path, re-inference bookkeeping, GUI/catalog source-of-truth, and a
handful of enumerations with no defining rule.

---

## Findings

### 1. HIGH — Re-tag never says *what bytes it materializes from*, and the naive reading fails in the normal case

**Where:** §5.8 "Re-tag (from the GUI) is a move, per spec: materialize into `<new_label>/` first
(same atomic path), then unlink the old destination"; §6 `POST /api/images/{sha256}/label`.

Re-tag is defined by reference to `materialize.py`, whose three documented modes all read from the
**source**: copy streams from the source file, `--link` symlinks `abs_source`, `--hardlink` calls
`os.link(source, tmp)`. The GUI's whole purpose is reviewing a card that has usually already been
unplugged, so "materialize into `<new_label>/`" as written is unimplementable at exactly the moment
it will be used — and in `--link` mode it is ambiguous whether the *symlink* moves or a fresh
symlink is created to a source path that may be gone. §5.8 also promises the crash window leaves
"the image present in two label dirs", which only makes sense if the new file is created before the
old is removed; a plain rename has no such window, so the two statements describe different
algorithms. Nothing in the design tells the implementer which.

**Fix — specify re-tag as a pure output-tree operation that never reads the source:**

```python
# gui/app.py -> materialize.retag(sha256, new_label)
old = row.dest_path                      # must be under output_root, else 409 + row 'failed'
new_dir = output_root / new_label        # created 0o755 if absent
new = collision_resolve(new_dir / old.name, sha256, mode=row.mode)
os.replace(old, new)                     # same filesystem by construction; moves a regular
                                         # file, a hardlink or the symlink itself (no deref)
catalog.record_override(sha256, old_label, new_label, note)   # after the rename
```

- Applies to all three modes unchanged: `os.replace` renames the directory entry, so a symlink
  moves as a symlink (never dereferenced, so a dangling link is fine) and a hardlink keeps its
  inode.
- Collision at the new destination reuses §5.8's rules: identical content (or, in link mode,
  identical `os.readlink`) → remove the old entry, count `already_present`; different content →
  `<stem>-<sha256[:8]><suffix>`.
- Then delete §5.8's "present in two label dirs" sentence and the corresponding `verify --fix`
  justification for the *re-tag* case, or keep the copy-then-unlink ordering and say explicitly
  that the copy's bytes come from `old_dest` — but pick one. If you keep copy-then-unlink, state
  that the source path is never opened during a re-tag, because that is the property E1 asserts.

---

### 2. MEDIUM — `--reclassify` has no rule for the stale `boxes` / `candidates` / override rows it replaces

**Where:** §5.9 "Idempotency, resume and `--limit`"; schema in §5.9.

`boxes` has no unique constraint on `(sha256, idx)` and `candidates` none on `(box_id, rank)`, and
no section says re-inference deletes the previous rows. A second `classify --reclassify` therefore
doubles every box and every candidate for that image, which silently corrupts `/api/images`, the
box overlay and any `export-trainset` that reads "the dominant box". E13 only covers the *skip*
path, so no test would catch it.

**Fix:** state the write order for a re-inferred hash, inside the single per-image transaction:

```sql
BEGIN;
DELETE FROM candidates WHERE box_id IN (SELECT id FROM boxes WHERE sha256 = :sha);
DELETE FROM boxes      WHERE sha256 = :sha;
-- re-insert boxes + candidates, then UPDATE images SET ... , last_updated = :now
COMMIT;
```

and add to E13 a `--reclassify` leg asserting `COUNT(*)` over `boxes` and `candidates` is unchanged
after the second run.

---

### 3. MEDIUM — `--limit` budget rule contradicts `--reclassify`

**Where:** §5.9: "`--limit N` caps images newly submitted to inference; hashes already at
`status='done'` are skipped *without consuming budget*."

Under `--reclassify` those hashes are precisely what *is* submitted to inference, so the rule as
written is self-contradictory: either `--reclassify --limit 2` re-does 2 images (budget spent on
done rows) or it does nothing forever (done rows skipped without budget). Both are defensible; the
design must choose, because it changes observable behaviour on a 5,000-image card.

**Fix:** define the budget in terms of inference, not status: "`--limit N` caps the number of images
submitted to inference in this run. Without `--reclassify`, `status='done'` hashes are skipped
before the budget is consulted. With `--reclassify`, every image processed — including previously
`done` ones — consumes budget, and the run advances in catalog `last_updated` order (oldest first)
so repeated `--reclassify --limit N` runs sweep the whole card." Extend E13 with two
`--reclassify --limit 2` runs asserting 4 distinct hashes got a new `model_id`/`last_updated`.

---

### 4. MEDIUM — `temperature` is required at inference but nothing defines what `train` writes

**Where:** §5.5 ("softmax with the artifact's calibration temperature"), §7.1 (`"temperature": 1.37`
shown as an illustrative value), §7.4 (only `eval --calibrate` fits a temperature).

`train` is the command that produces `models/species.acmodel` and `models/birds.acmodel`, and E7
classifies with a freshly trained artifact — so an uncalibrated artifact must still have a defined
temperature. Today the implementer either invents `artifact.get("temperature", 1.0)` (a silent
default for a value inference depends on, contra I7) or `train` produces artifacts that `classify`
cannot load.

**Fix:** make it explicit in §7.1: `train` always writes `"temperature": 1.0` and
`"train": {..., "calibrated_from": None}`; `artifact.load()` treats a **missing** `temperature` key
as a fatal `AssetError` naming `eval --calibrate`; only `eval --calibrate` may write a value ≠ 1.0,
into a new file (I8). Add to E7's assertions: `temperature == 1.0` and `calibrated_from is None`.

---

### 5. MEDIUM — `slug()` is defined for spaces only, and failing its regex has no defined outcome

**Where:** §5.8 "Species names are lowercased with spaces → `_` by `taxonomy.labels.slug()` and
validated against the same regex"; §5.7 `taxonomy.slug(box.species_common)`; label regex
`^[a-z0-9][a-z0-9_-]{0,63}$`.

Two gaps. (a) Real label sources contain characters the rule does not mention: CUB-200 common names
carry apostrophes and periods once humanised (`Brewer's Blackbird`, `Le Conte's Sparrow`), and
`--allow-new-labels` lets a user introduce anything. Lowercase + space→`_` turns
`Brewer's Blackbird` into `brewer's_blackbird`, which fails the regex. (b) The design says the
result is "validated" but never says what validation failure *does* — and this is a required value
(the destination directory), so I7 says it cannot fall back to something.

**Fix:** publish the function next to the regex, and make failure fatal at artifact-load time (not
mid-run, when files are already being written):

```python
_KEEP = re.compile(r"[^a-z0-9]+")
def slug(common: str) -> str:
    s = unicodedata.normalize("NFKD", common).encode("ascii", "ignore").decode()
    s = _KEEP.sub("_", s.lower()).strip("_")[:64]
    if not LABEL_RE.fullmatch(s):
        raise ConfigError(f"label {common!r} does not slugify to a valid directory name ({s!r})")
    return s
```

and add to §7.1: `artifact.load()` slugs every label entry up front and fails fatally (exit 3) on
any that does not validate, so a bad label space can never reach `materialize`.

---

### 6. MEDIUM — The coordinate frame for `width`/`height`, box coordinates and thumbnails is never stated

**Where:** §5.2 (`ImageOps.exif_transpose` on decode), §5.4 (boxes "in original-image pixel
coordinates"), §5.9 (`images.width`, `images.height`), §6 (canvas overlay "scaled from the stored
pixel coordinates and the stored `width`/`height`", `/thumb`, `/full`).

"Original-image pixel coordinates" is ambiguous exactly for the images that matter: a portrait
photo with EXIF orientation 6 has different pre- and post-transpose dimensions. Detection runs on
the transposed image, so boxes are in the transposed frame — but `/full` serves the **original file
bytes**, whose EXIF the browser applies on its own, and `/thumb` is generated server-side with no
stated orientation handling. Get any one of the three wrong and every rotated photo draws its boxes
sideways, with no test that would notice (E18 only checks that coordinates lie inside the stored
dimensions, which holds in either frame).

**Fix:** state once, in §5.2, that all persisted geometry is in the **EXIF-transposed** frame:
`images.width/height` are `exif_transpose(im).size`, all `boxes` coordinates and `area_frac` are in
that frame, `/thumb` is generated from the transposed image, and `/full` relies on the browser's
default `image-orientation: from-image` (add `<img style="image-orientation: from-image">` so it is
explicit). Extend E18 with one EXIF-orientation-6 fixture asserting
`images.width < images.height` for a physically landscape file plus a box whose coordinates fall
inside the transposed frame only.

---

### 7. MEDIUM — Two scan skip reasons have no defining rule (`too_large`, `symlink_loop`), and symlinked *files* have no policy

**Where:** §5.1 skip-reason enumeration; §5.2 (`too_large` used for > 400 MP at decode).

`too_large` is listed as a *scan* reason but scan has no size rule — the only threshold in the
design is decode's 400 MP, so an implementer must invent a byte cap (or leave a dead reason).
`symlink_loop` cannot occur at all with `os.walk(followlinks=False)`. And the policy for a
symlinked regular file inside the card is unspecified: `os.walk` yields it, `images.open_source`
will happily follow it, so a link pointing outside the card is silently ingested with the wrong
provenance in `sources`.

**Fix:** pick one of these and write it down:

- Delete `too_large` from the scan list (it is decode's reason only), or define it — e.g. "scan
  skips regular files larger than `max_file_bytes` (default 512 MiB) with reason `too_large`" and
  add `max_file_bytes` to §3 and §10.2.
- Delete `symlink_loop`, or keep it for the one reachable case: `os.path.realpath` of a symlinked
  file resolving outside `source`.
- Add explicitly: "a symlinked file inside the source tree is skipped with reason `symlink` unless
  `--follow-source-symlinks` is passed; symlinked directories are never descended into
  (`followlinks=False`)." Add the `.mp4`-style case to E16.

---

### 8. MEDIUM — `ebird_enrich` names a static table that does not exist in the layout, and never defines "candidate absent from the response"

**Where:** §5.6 (`ebird_enrich`: "canonicalize names against the static eBird-style table only";
"Every candidate **absent** from the response has its score multiplied by 0.25"); §4 lists only
`taxonomy/data/coco_animals.csv` and `taxonomy/data/cub200.csv`.

Two unresolved dependencies. (a) The "static eBird-style table" is not in the module layout, has no
schema, and no stated behaviour for a predicted name that is missing from it (canonicalise to
what?). (b) The down-rank rule turns on string matching between our candidate labels and the eBird
`obs/geo/recent` payload, and the matching rule is the entire behaviour of the provider — eBird
returns `comName`/`sciName` in its own orthography (`"Zebra Dove"`, `"Streptopelia chinensis"`),
which will not equal a CUB label (`"Mourning Dove"` vs `"Zenaida macroura"`, and CUB ships
`scientific = NULL` for many classes per §13.3). Since RECON confirms nothing in-sandbox can reach
eBird, this cannot be discovered later by running it — it has to be pinned in the design.

**Fix:** (a) add `taxonomy/data/ebird_aliases.csv` to §4 with columns
`cub_key,ebird_com_name,ebird_sci_name`, and state that a candidate absent from the alias table is
left untouched and logged at `DEBUG` (no canonicalisation, no down-rank). (b) define matching
precisely: "a candidate matches an observation when its `ebird_sci_name` equals the observation's
`sciName` case-folded, else when its `ebird_com_name` equals `comName` case-folded with
`[^a-z0-9]` stripped. Candidates with no alias row are **exempt** from the 0.25 multiplier, because
absence from our alias table is not evidence about the bird's range." Add that exemption to E23's
assertions.

---

### 9. MEDIUM — `/api/labels` counts come from the filesystem while `/api/images` comes from the catalog

**Where:** §6 (`GET /api/labels` "enumerated from `output_root` subdirectories"), `GET /api/images`
(catalog rows), E18.

Two sources of truth for the same fact. They diverge in cases the design itself creates: after
`--dry-run` every row is `planned` with no directories at all, so the sidebar is empty while
`/api/images` returns rows (E17 exercises exactly that catalog state); a hash-suffixed collision
file (§5.8) counts as one file but is one row too, fine; but a `verify`-detected missing
`dest_path` inflates the row count over the file count with no way for the GUI to show it.

**Fix:** make the catalog authoritative and the filesystem a cross-check:
`GET /api/labels` returns `SELECT label, COUNT(*) FROM images WHERE status IN ('done','planned')
GROUP BY label`, each entry carrying `{"label", "count", "files_on_disk"}` where the second number
is the filesystem count (tool-internal entries ignored per §5.8) so a mismatch is visible in the UI
and reproduced by `verify`. Update E18 to assert both numbers, and add a dry-run leg asserting the
sidebar lists labels with `files_on_disk == 0`.

---

### 10. MEDIUM — `/thumb` has no defined input, and no defined behaviour when there is no materialized file

**Where:** §6 `GET /api/images/{sha256}/thumb` — "320 px JPEG, generated on demand, cached at
`<output_root>/.thumbs/...`".

Generated from what? `/full` is explicitly `dest_path`-only (and 409s for `planned` rows), but the
thumb route says nothing, so the obvious implementations are (a) `dest_path` — then dry-run and
missing-file rows have no defined response, and (b) the source path — which would contradict §6's
"the GUI never serves bytes from the source tree" and put a read of the SD card behind every grid
tile. Also unspecified: whether a dangling symlink destination (card unplugged, `--link` mode) 409s
or 500s, which is the *normal* state for link-mode users.

**Fix:** state: "`/thumb` is generated from `dest_path` only, with `ImageOps.exif_transpose` applied
(finding 6), and cached. A row with `status='planned'`, a missing `dest_path`, or a dangling symlink
destination returns `409 {"error": "no readable materialized file for this image"}`; the grid renders
a placeholder tile for 409s." Add the dangling-symlink and planned cases to E17/E18.

---

### 11. MEDIUM — `export-trainset` is undefined for the labels it will mostly encounter

**Where:** §7.5 ("walks the catalog for `label_source='human'` … emits a manifest of the dominant
box of each image"), E25.

Human overrides are drawn from the *whole* closed label set, so the common cases are exactly the
ones with no species and often no box: `multiple`, `landscape`, `junk`, `unknown`. "The dominant box
of each image" does not exist for `landscape`/`junk` (no boxes) or for `multiple` (no dominant box
by definition), and emitting `label: "junk"` lines into a species manifest would train the head on
a non-species class — while §7.2 says a missing `box` is legal, so nothing would reject it.

**Fix:** define the filter and the mapping in §7.5: "Only images whose label is a species slug are
exported by default. `multiple`, `landscape`, `junk` and `unknown` are skipped and counted in the
summary line (`--include-non-species` emits `landscape`/`junk` as their own classes for users
training a scene filter; `multiple` and `unknown` are never exported). For an exported image the
`box` is the dominant box; a species-labelled image with no boxes (possible only after a human
override) is exported with `box` omitted." Extend E25 to assert a `junk` override produces no
manifest line and that the counted-skip summary reports it.

---

### 12. MEDIUM — E4's sanctioned remedy is arithmetically impossible (measured: 19 candidates for a 20-image list)

**Where:** §11.2 closing note: "the sanctioned remedy is to re-freeze the list at a stricter GT
ratio (19 val-bucket images are available at ratio > 4.0)"; §11.1 fixes both frozen lists at 20
entries.

I recomputed this from `instances_val2017.json` with the design's own `split_for`: ratio > 4.0
yields **79 images total, 19 in the val bucket** — the design's own number. A 20-entry `high` list
cannot be built from 19 images, so the documented escape hatch fails the moment it is needed, and
the only remaining moves are the two the design forbids (lower the ratio in test code, or break the
leakage rule).

**Fix:** make the remedy executable — state it as "re-freeze `high` at GT ratio > 4.0 with
`len(high) = 19` (measured availability), and lower E4's high-side assertion to ≥ 15/19", or make
list length data-driven from the start: "`high` contains **all** val-bucket images above the
threshold (22 at > 3.0, 19 at > 4.0); the assertion is `≥ ceil(0.8 * len(high))`". Either removes
the hard-coded 20.

---

### 13. MEDIUM — In `--hardlink` mode the destination shares the SD card's inode, which the path-based proof of I1 does not cover

**Where:** §5.8 (`os.link(source, tmp)`), I1 ("materialize.py refuses paths outside `output_root`"),
E3 (asserts `st_ino` equality — i.e. aliasing is a *tested* property).

I1's argument is entirely path-based, but a hardlinked destination *is* the card's file: any future
in-place write through the `~/animal_pics` path would modify the SD card while passing every guard.
Nothing in the design writes into a materialized file today, so this is not a bug — it is an
unstated precondition for the strongest invariant in the document, and re-tag (finding 1) is the
first feature that touches existing destinations.

**Fix:** add to §5.8 and I1: "In `--hardlink` mode a destination and its source file are the same
inode. Therefore no code path ever opens a materialized file for writing or truncates it: the only
operations permitted on an existing destination are `os.replace` (rename) and `os.unlink`, both of
which affect the directory entry only. `verify --fix` and GUI re-tag comply." Add to E1 a hardlink
leg (E1 currently runs in the default copy mode), which is what makes this checkable.

---

### 14. MEDIUM — Confidence filtering has no defined behaviour for rows whose confidence is NULL

**Where:** §6 `GET /api/images` (`min_conf`, `max_conf`), §10.2 (validates the parameters' range but
not their semantics), §5.9 (`images.confidence REAL`, nullable).

`landscape`, `junk`, `multiple` and degenerate-box `unknown` rows have no meaningful confidence —
presumably NULL, though the design never says so. In SQL, `confidence >= :min` silently excludes
NULL, so "filter by confidence band" would make an entire class of images invisible with no
indication, which is the opposite of the "nothing is discarded" property the label set exists to
guarantee.

**Fix:** state in §5.9 that `images.confidence` is NULL exactly for `landscape`, `junk`, `multiple`
and any `unknown` with no classifiable box, and in §6 that "a confidence filter matches NULL-
confidence rows only when `include_unscored=true` (default `false`), and the response always
returns `unscored_excluded: <count>` so the UI can show 'N images have no confidence score'." Assert
that count in E18.

---

### 15. NIT — §5.9 says `gui` reads `runs.source_root`; §6 says the GUI never touches the source tree

§5.9: "`runs.source_root` … is the **only** provenance of the source tree for later commands: `gui`
and `verify` take no `SOURCE` argument and read it from the newest `runs` row." But §6 removed the
GUI's source-serving mode entirely (review item 10), so the GUI has no use for the value.
**Fix:** drop `gui` from that sentence, or say what it uses it for (e.g. displaying the card path in
the header, which is harmless and probably useful).

### 16. NIT — `verify`'s failure exit code is not mapped onto the documented code table

§8 says `verify` exits "non-zero"; the table defines `1` unexpected/IO, `3` missing/invalid asset,
`4` completed with per-image failures. **Fix:** state it: "missing or unloadable asset → 3;
catalog/filesystem inconsistency (reconcilable, or fixable with `--fix`) → 4." E24 should assert the
specific codes rather than "non-zero".

### 17. NIT — `--force-bird-head` can overwrite a confident non-bird species with a CUB label

§5.6 runs the bird head "on every animal crop regardless of the coarse prediction", and the merge
rule replaces the coarse row whenever the bird head clears the gate — so a zebra crop can be filed
as a CUB species in a diagnostic run. **Fix:** add "diagnostic only: `--force-bird-head` logs one
`WARNING` per run stating that non-bird crops may be relabelled, and the run's `runs.config_json`
records it (already specified) so such labels are identifiable afterwards."

### 18. NIT — `run_id` format is unspecified

It is a primary key written into every `images` row. **Fix:** `run_id = uuid4().hex` (or
`f"{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"` if you want sortability, which `GET /api/run`'s
"newest row" query would appreciate — that query also needs a defined ordering: `started_at DESC`).

### 19. NIT — `candidates` has no primary key or index

§5.9 gives `candidates(box_id, rank, ...)` with only a foreign key, so duplicate `(box_id, rank)`
pairs are legal and per-box lookups scan. **Fix:** `PRIMARY KEY(box_id, rank)`, plus
`CREATE INDEX idx_boxes_sha ON boxes(sha256)` and `CREATE INDEX idx_images_label ON images(label)`
for the GUI's grouped queries.

### 20. NIT — A species slug could collide with a reserved label

`multiple`, `landscape`, `junk` and `unknown` are directories like any species. No current label
space collides, but nothing forbids it, and `--allow-new-labels` accepts any regex-valid string.
**Fix:** add to `artifact.load()` (alongside finding 5's slug validation) and to the GUI's label
validator: a label equal to any reserved name is rejected — fatal exit 3 for an artifact, HTTP 422
for the GUI.

---

## Verified assumptions

Everything below I re-measured in this sandbox during the review; all of it matches DESIGN.md
exactly, including the numbers in its "Facts newly measured for this iteration" table.

| Claim (DESIGN.md) | My measurement | Verdict |
|---|---|---|
| MegaDetector v5a on disk, 280,766,885 B, sha256 `94e88fe9…b01b276` | byte-identical size and sha256 recomputed from `models/md_v5a.0.0.pt` | ✅ |
| Checkpoint loads with the `models`/`utils` alias shim and `weights_only=False`; `names=['animal','person','vehicle']`, `stride=[8,16,32,64]`; forward → `(1, 25500, 8)` | reproduced exactly, on CPU | ✅ |
| §2.1: MegaDetector loads **and forwards** with `roboflow` and `sahi` absent | reproduced with a `sys.meta_path` blocker raising `ImportError` for both roots: `import yolov5`, `letterbox`, `non_max_suppression`, `scale_boxes`, checkpoint load and a 640×640 forward all succeed, and neither module ends up in `sys.modules` | ✅ |
| §2.1 override trick drops `opencv-python`, `roboflow`, `sahi` and keeps `typer==0.27.2` | `uv pip compile` on the exact §2.1 `pyproject.toml`: resolution contains `opencv-python-headless==5.0.0.93`, `typer==0.27.2`, `torch==2.14.0`, `timm==1.0.29`, `pillow==12.3.0`, `pillow-heif==1.7.0`, `numpy==2.5.3`, `setuptools==80.10.2`, `ultralytics==8.4.153`, `rawpy==0.27.1`, `pytest==9.1.1` — and **no** `opencv-python`, `roboflow` or `sahi`. 110 lines with `--all-extras` vs the design's 109 base packages, consistent | ✅ |
| `pillow-heif==1.7.0` encodes HEIF on py3.12 + pillow 12.3.0 (the iteration-1 blocker for E16) | fresh venv: `pillow_heif 1.7.0`, `libheif 1.23.3`, `save(format="HEIF")` then re-open → `(64,48) RGB HEIF` | ✅ |
| COCO animal instances: 2,700 − 34 `iscrowd` = **2,666** over **1,016** images; per-class `bird 427, cow 372, sheep 354, horse 272, zebra 266, elephant 252, giraffe 232, dog 218, cat 202, bear 71` | identical, all ten counts | ✅ |
| Sub-2px animal boxes: **7**, smallest `area_frac = 0.00001243` | 7 boxes, min `1.2426814988e-05` | ✅ |
| Dominance ground truth at 1.6: 471 multi-animal → **235 dominant / 236 multiple** | identical | ✅ |
| `split_for` on the 1,016 animal images → **813 train / 203 val** | identical (basename-keyed sha1 % 5) | ✅ |
| Frozen-list availability: ratio > 3.0 → 104 total, **22 val**; ratio < 1.3 → 149 total, **29 val** | identical, so the 20 + 20 lists are buildable under the leakage rule | ✅ |
| E7: 7-class crops **1,625 train / 349 val**, majority baseline **0.229**, chance 0.143 | identical; val per-class `bird 80, cow 75, giraffe 50, sheep 49, zebra 48, elephant 36, bear 11` — confirming `bear` is the class that cannot carry a recall threshold | ✅ |
| `coco_species.json` availability: 24 single-animal, `area_frac ≥ 0.20`, val-bucket, in E7's seven | 24 | ✅ |
| Assets present for offline work: `models/backbones/{efficientnet_b0_ra-3dd342df,convnext_nano_d1h-7eb4bdea}.pth`, `data/raw/{CUB_200_2011.tgz,val2017.zip,annotations_trainval2017.zip}`, `data/raw/annotations/instances_val2017.json` | all present at the RECON sizes | ✅ |
| RECON's blocked hosts are not depended on | no design path requires `huggingface.co`, `download.pytorch.org`, `storage.googleapis.com`, `lila.science`, `api.inaturalist.org`; `api.ebird.org` is opt-in, off by default, and its unreachability is a recorded degradation | ✅ |

Reproduction: the three probes I ran were `sys.meta_path`-blocked MegaDetector load/forward against
`.venv`, `uv pip compile` on a scratch copy of §2.1's `pyproject.toml`, a scratch venv with
`pillow==12.3.0 pillow-heif==1.7.0` doing a HEIF round-trip, and a plain-`json` pass over
`data/raw/annotations/instances_val2017.json` reimplementing `split_for` from §7.2.

## Unverified / wrong assumptions

Nothing in the design measured **wrong**. These are the claims that remain unproven, with the risk
each one carries:

1. **E7's accuracy threshold (`val_top1 ≥ 0.55` from 2 + 2 epochs at input 128 on 1,625 crops,
   < 20 min on 8 CPU cores).** Not runnable inside a review. The class balance I measured makes it
   plausible (majority baseline 0.229, 7 classes, ImageNet-pretrained backbone), but both the
   threshold and the wall clock are estimates. Keep §11.2's rule — if it fails, re-freeze or
   re-state the threshold in the design, never silently lower it in test code.
2. **E10's synthetic threshold (`val_top1 ≥ 0.90` in < 90 s with a ~180 k-param `tinycnn`).**
   Unproven; low risk, since the dataset is generated and separable by construction.
3. **E4's aggregate thresholds (≥ 16/20 each way) against *real* MegaDetector output.** The ground
   truth is verified, but MegaDetector's boxes legitimately differ from the annotator's, and no one
   has yet run the detector over these 40 images. Finding 12 is about the remedy path being
   unbuildable, and it matters precisely because this threshold is the most likely to move.
4. **`efficientnet_b0` at `--input-size 128`.** timm's `efficientnet_b0` default config is 224; a
   128 px input works architecturally (global pooling) but the pretrained features are being used
   off-resolution, which is a real accuracy risk for item 1. Worth stating in §7.3 that the
   resolution is a deliberate CPU-budget trade-off for E7 only, and that the shipped default stays
   224.
5. **`ebird_enrich`'s 0.25 multiplier and the `dist=50&back=30` query.** Unverifiable here by
   RECON's own findings (`api.ebird.org` → `000`), and §13.4 already labels the constant as chosen,
   not fitted. Fine as an honest assumption — but see finding 8: the *matching rule* it operates on
   must still be specified, since it cannot be discovered by running it.
6. **`typer` 0.27.2's exact CLI shape for `--formats` (repeatable) and the mutually exclusive
   `--link`/`--hardlink` pair.** Not exercised. Both are ordinary typer patterns; the only note is
   that mutual exclusion is hand-rolled in a callback (exit 2 per §10.2), not something typer
   enforces for you.
7. **The GUI's 409-on-lock-contention path (E19).** Depends on SQLite `BEGIN IMMEDIATE` semantics
   under WAL with a `busy_timeout` of 10 s in the writer and (unstated) a short one in the GUI. The
   design should name the GUI's `busy_timeout` (suggest 250 ms) so E19 can provoke the 409
   deterministically instead of racing a 10 s wait.
