# Cold design review — DESIGN.md iteration 3

Reviewed 2026-09-15 16:43–21:11 UTC, without the context that produced the document. Inputs:
`docs/DESIGN.md` (1,542 lines, iteration 3), `docs/PLAN.md` (agreed behaviour), `docs/RECON.md`
(reachability facts). Everything checkable was re-measured in-sandbox rather than taken on the
document's word; the probe is committed as `scripts/recon/verify_design_facts.py`.

**Verdict: CHANGES_REQUESTED** — 11 MEDIUM, 8 NIT, 0 HIGH. **No blocking condition is tripped.**

## Blocking-condition audit (all clear)

| Condition | Result |
|---|---|
| `min_box_area` / minimum box area / any absolute box-area floor under any name | **Clear.** Every one of the 7 occurrences of the phrase is an explicit negation (§3, §5.3, §5.7, I2, E6). `dominance_ratio` is the only size gate, the comparison is a multiplication (no `ZeroDivisionError`), degenerate boxes still count as animals for dominance, and E6 asserts a lone 0.05 %-of-frame animal is filed as its species. `max_file_bytes` is correctly walled off in `scan.py` as a whole-file cap that structurally cannot see a box |
| Unit tests, or any layer other than `tests/e2e` | **Clear.** `tests/e2e` only, stated in §2, §11 and §11.2's "two sanctioned seams" paragraph. The two doubles (`--detector scripted`, E23's `httpx.MockTransport`) both replace an external boundary while the real console script runs; neither imports a private helper |
| Dependence on a RECON-unreachable asset without a buildable fallback | **Clear.** Detection = local `models/md_v5a.0.0.pt` (present, 280,766,885 B); backbone = local `models/backbones/efficientnet_b0_ra-3dd342df.pth` (present); data = local `data/raw/{CUB_200_2011.tgz,val2017.zip,annotations_trainval2017.zip}` (all present). huggingface.co appears twice, both times as "never do this" (`pretrained=True`) or as the user's own machine. `api.ebird.org` is opt-in, off by default, degrades to one WARNING, and its stub is fatal-by-design |
| Writes/moves/renames/deletes in the source SD-card tree | **Clear.** Three mechanisms (§5.1) + I1 extended past its path-based argument to the `--hardlink` aliased-inode case (only `os.replace`/`os.unlink` on an existing destination), and E1 runs a hardlink leg plus a leg with the card chmod `0o500` and then renamed away |
| Silent default for a missing required value | **Clear.** I7 is structural: missing `temperature` is fatal (not `.get(k, 1.0)`), `--device cuda` on a CUDA-less host is fatal, `slug()` raises instead of guessing, missing `--config` is fatal, `ebird_enrich` without a key is fatal. The one degradation (eBird unreachable) is an *optional* enrichment and is recorded in `provider_status` |
| Omission of any mandated component | **Clear.** MegaDetector v5a §5.4; own finetuned species model + full `training/` subsystem + `train` §7; three-implementation bird provider with `own_bird_head` default §5.6; label set `species\|multiple\|landscape\|junk\|unknown` §5.7; atomic temp+`os.replace` §5.8; copy/`--link`/`--hardlink` §5.8; sha256-keyed idempotent resumable runs §5.9; SQLite catalog §5.9; FastAPI + vanilla-JS GUI on 127.0.0.1:8765 with re-tag that MOVES files and records overrides §6; typer CLI with all six commands §8; format policy JPEG/PNG/TIFF/HEIC default, RAW behind `--raw`, videos always skipped §3.1/§5.1 |

The document is unusually strong on the axes it was previously pushed on. What remains are gaps
concentrated in the areas iteration 3 *added* (the GUI query surface, the CUB bird label space, the
second-run/skip bookkeeping) plus two self-contradictions the iteration-2 fixes left behind.

---

## Findings

### 1. MEDIUM — `skipped(path TEXT PRIMARY KEY)` has no upsert, so the second run over any card containing a skipped file raises `IntegrityError`

§5.9 publishes an explicit `ON CONFLICT(path) DO UPDATE` upsert for `sources` but leaves `skipped`
as a bare `PRIMARY KEY(path)` with no insert statement specified. Skips are the *normal* case (every
card has an `.mp4`, a hidden file or an oversize file), and §5.9 requires re-runs to re-walk the tree,
so the same path is re-skipped on run 2. Verified empirically against the exact published schema:

```
schema: CREATED OK (sqlite 3.40.0)
second run re-skips same path: IntegrityError -> UNIQUE constraint failed: skipped.path
```

E13 ("`classify` twice") and E16 (a card whose whole point is skipped files) both walk into this.

**Fix.** Publish the statement next to the `sources` one:

```sql
INSERT INTO skipped(path, reason, detail, run_id, seen_at) VALUES(?,?,?,?,?)
ON CONFLICT(path) DO UPDATE SET reason=excluded.reason, detail=excluded.detail,
                                run_id=excluded.run_id, seen_at=excluded.seen_at;
```

and add to E13: a second run over a card containing an `.mp4` exits 0/4 as specified with exactly one
`skipped` row for that path, whose `run_id` is the second run's.

### 2. MEDIUM — the CUB bird label space is underspecified, and the rationale given for `slug()` is factually wrong about CUB's actual names

§5.8 motivates `slug()` with "humanised CUB-200 names carry apostrophes and periods (`Brewer's
Blackbird`, `Le Conte's Sparrow`)". CUB-200-2011's real `classes.txt`, read from the archive in
`data/raw`, contains no apostrophes at all — the names are already underscored and carry a numeric
prefix:

```
1 001.Black_footed_Albatross
22 022.Chuck_will_Widow
193 193.Bewick_Wren
```

§7.2 says `build_cub_manifest.py` "emits one sample per image with the species from its directory
name". Taken literally, the bird artifact's label `key`/`common` become `022.Chuck_will_Widow`, which
`slug()` maps to `022_chuck_will_widow` — so the user gets `~/animal_pics/022_chuck_will_widow/`.
That contradicts the product goal in PLAN.md ("dirs like `~/animal_pics/lion`") and the promise of
"common names or scientific names". `taxonomy/data/cub200.csv` exists in §4 but the document never
says it is authoritative for display names, nor how `ebird_aliases.csv`'s `cub_key` is spelled.

**Fix.** State the derivation once, in §7.2 and §4:

- `022.Chuck_will_Widow` → strip the `^\d{3}\.` prefix → `key = "chuck_will_widow"` (this is also the
  `cub_key` spelling in `ebird_aliases.csv`).
- `common` comes from `taxonomy/data/cub200.csv`, which is authoritative for display names and
  scientific names (`scientific = NULL` where unknown, per §13.3); the directory name is used only to
  derive the key, never as the label directory.
- `artifact.load()` already validates the space, so add to E8: the resulting label directory matches
  `^[a-z][a-z0-9_]*$` and contains no leading digits.

Verified separately that `slug()` as published is otherwise correct — `Brewer's Blackbird` →
`brewer_s_blackbird`, `Ñandú` → `nandu`, `'熊'` and `'  '` raise, and the 64-char truncation lands
exactly on the `LABEL_RE` limit.

### 3. MEDIUM — the `q` query parameter is named once and never defined

§6's `/api/images` row lists `q` among the query parameters. It appears nowhere else in 1,542 lines:
no semantics (does it match filename? species common name? scientific name? note text?), no
validation row in §10.2 (which validates every other parameter), no test in E18. An implementer must
invent it, and whatever they invent is untested.

**Fix.** Either delete `q` from the route, or specify it: `q` is a case-insensitive substring match
over `images.species_common`, `images.species_scientific` and `sources.path` basename, max 128 chars,
`422` above that; add it to §10.2 and assert one hit + one miss in E18.

### 4. MEDIUM — `date_from`/`date_to` reintroduce exactly the NULL-silent-drop bug that `include_unscored` was added to fix

§5.2/§10.2 make `exif_datetime` genuinely optional — absent EXIF stores `NULL` and is never
fabricated. §6's date filter is unspecified beyond its name, so the natural SQL
(`exif_datetime >= :from`) silently drops every NULL-dated image. That is the same failure the
document diagnoses at length for `confidence` ("would hide an entire class of images the moment a
user touched the slider, the exact opposite of the *nothing is discarded* property"), left unfixed one
row above the fix. Cards from cameras with a dead clock, and every screenshot or exported frame, land
in that bucket.

**Fix.** Mirror the confidence treatment exactly: accept `date_from`/`date_to` as ISO-8601 dates
(`YYYY-MM-DD`, inclusive of the whole `date_to` day, `422` on unparseable, `422` when
`date_from > date_to`); NULL-dated rows match only when `include_undated=true` (default `false`); every
response carries `undated_excluded: <count>` alongside `unscored_excluded`. Add both to §10.2 and
assert the count in E18.

### 5. MEDIUM — exit code 4 vs 0 for benign skips is self-contradictory, and E2/E16 cannot both pass

§8's table defines exit `4` as "completed with per-image **failures or skips**". §10.1 classifies
`video`, `format_disabled`, `too_large`, `symlink` and `raw_not_enabled` as "**not an error**", logged
at `DEBUG`. Both statements cover the same rows. E16 asserts exit **4** for a card whose only
abnormality is such skips; E2 asserts exit **0** for a "mixed fixture card" and E17 exit **0** for a
dry run — with no rule saying which skips count, a mixed card that includes a video is
simultaneously required to exit 0 and 4.

**Fix.** Split the skip classes explicitly in §8:

- exit `0` — only *benign* skips occurred: `hidden`, `system_dir`, `unsupported_extension`,
  `format_disabled`, `video`, `raw_not_enabled`, `too_large`, `symlink`.
- exit `4` — at least one *abnormal* outcome: any `failed` image, or a skip in
  `{zero_bytes, unreadable, symlink_escape, too_large_pixels, decode_error}`.

Then state E2's card contains benign skips only (exit 0) and E16's contains the 0-byte and truncated
files (exit 4), which is what each test already implies.

### 6. MEDIUM — re-tagging a `--dry-run` row marks a healthy row `failed` and hides the image from the GUI

§5.8's `retag()` treats "`dest_path` … missing on disk" as `409` **plus** "row marked `failed`", and
§10.1 repeats it. But §5.8 also has `--dry-run` write the *planned* destination into `dest_path` with
`status='planned'`, so for every dry-run row the file legitimately does not exist. A user browsing a
dry-run catalog (E17's exact state) who clicks a label therefore flips a correct `planned` row to
`failed` — and `/api/labels` counts only `status IN ('done','planned')`, so the image disappears from
the sidebar and `verify` gains a finding, all from a read-only-intent click.

**Fix.** Guard on status before touching anything, and never mutate on this path:

```python
if row.status == "planned":
    raise HTTPException(409, {"error": "this image was only planned (--dry-run); "
                              "run classify without --dry-run before re-tagging",
                              "reason": "planned"})
```

Reserve the `failed` marking for a row that was `done` and whose file has genuinely vanished. Add the
POST leg to E17: `409 reason="planned"`, row still `planned`, no `overrides` row written.

### 7. MEDIUM — `crop_margin` is user-configurable but is not recorded in the artifact, so train/inference framing can silently skew

§7.2 says `dataset.py` "crops with the same margin as inference (0.08) so train and inference see the
same framing" — but §3 exposes `crop_margin` as config, §10.2 admits `[0.0, 0.5]`, and §5.3 crops with
whatever the run's value is. A user who sets `crop_margin = 0.25` gets inference crops that no longer
match the artifact's training distribution, with no warning and no way to detect it afterwards. The
`.acmodel` is otherwise scrupulously self-describing (`input_size`, `normalize`, `temperature`), and
this is the one geometric parameter missing from it.

**Fix.** Add `"crop_margin": 0.08` to the artifact dict beside `input_size`, have `train` write the
value it actually used, and have inference use **the artifact's** value rather than the run config's —
logging at `INFO` when the config value differs, since the artifact wins. Assert in E7 that the
artifact records `crop_margin` and that `classify --crop-margin 0.3` still crops at the artifact's
0.08.

### 8. MEDIUM — `coco_species.json` is still frozen at a hard-coded 20 of 24, contradicting §11.1's own headline and §11.2's rule

§11.1's bolded rule is "**List lengths are data-driven, not hard-coded — and so is E4's assertion**",
and §11.2 forbids "never hard-code a list length again". The fix was applied to
`coco_dominance.json` only. The very next row freezes `coco_species.json` at "**20** images" while
recording that **24** qualify, and E7 asserts "**≥ 14/20**". Re-measured: 24 candidates exist, exactly
as claimed. So iteration 2's defect survives in the second list — if the `area_frac ≥ 0.20` or the
7-class filter is ever tightened and fewer than 20 qualify, the same dead end returns.

**Fix.** Apply the identical treatment: `build_coco_dominance.py` emits **all** qualifying
val-bucket single-animal images (24 today), and E7 asserts `>= ceil(0.7 * len(species_list))` with the
length read from the JSON (17 of 24 today). No literal `20` or `14` in test code.

### 9. MEDIUM — E7's two thresholds are not reconciled, and only one of them has a sanctioned remedy

E7 asserts both `val_top1 >= 0.55` (crop-level, 7 classes, 128 px) and "≥ 14/20 of the frozen
single-animal val images receive their GT species label **through the real CLI**" — i.e. 0.70
image-level accuracy, *after* the `min_species_confidence = 0.45` gate can convert a correct-but-
unconfident top-1 into `unknown`, and after MegaDetector's box replaces the GT box. A model that
lands exactly at the sanctioned 0.55 will very likely miss 14/20, so the test can fail while every
stated threshold is met. §11.2 supplies a remedy for the `val_top1` leg only ("raise `--input-size`
toward 224"), and none for the identity leg.

**Fix.** State the relationship and give the second leg a remedy: note that the identity subset is
deliberately easier than the crop-level average (single animal, `area_frac ≥ 0.20`), set the identity
threshold as a derived fraction (finding 8's `ceil(0.7 * len(list))`), and record the sanctioned
remedy — raise `--input-size` toward 224 and/or `--epochs-finetune`, never lower the fraction and
never widen `min_species_confidence` for the test.

### 10. MEDIUM — the re-processing policy covers `done`/`planned`/`materializing` but not `failed` or `skipped`

§5.9 specifies that `done` hashes are skipped and rows left `materializing`/`planned` are
re-processed. `images.status` also admits `failed` (per-image inference failure, exit 4) and
`skipped`, and the `skipped` table is keyed by path. Nothing says whether the next run retries a
`failed` image, whether a transient decode failure is retried forever, or whether a path in `skipped`
is re-examined at all (it must be — `--raw` or `--formats` may have changed since). E13 tests the
killed-run resume but not the retry path.

**Fix.** Add one row per status to §5.9: `done` → skipped unless `--reclassify`; `planned` /
`materializing` → re-processed; `failed` → **retried** (the failure may have been transient, e.g. a
lock or an OOM), and the retry is what consumes `--limit` budget; `skipped` paths → always re-walked
and re-evaluated against the current format policy, with the `skipped` row upserted per finding 1.
Assert in E13 that a `failed` row is retried on the next run and that a `.cr2` skipped without
`--raw` is ingested by a later `--raw` run.

### 11. MEDIUM — how `models/species.acmodel` and `models/birds.acmodel` come to exist is never specified, and the default paths are relative

§3's defaults are `species_model = "models/species.acmodel"` and `bird_model =
"models/birds.acmodel"` — relative paths, with no statement of what they are relative to (CWD? the
package root? `output_root`?). §1 says both artifacts are "produced by the training subsystem in this
repo", but §12's build order only produces E7's and E8's artifacts inside tests, and step 3 lands
`classify` with "a pass-through classifier". So it is undefined what a fresh clone does on
`animal-classifier classify /card`: exit 3 for a missing artifact is the honest answer, but the
document never says so, never names the command that produces the shipped pair, and the missing
`bird_model` would be fatal even for a user who never photographs a bird.

**Fix.** State three things: (a) relative model paths resolve against the **current working
directory**, and the resolved absolute path appears in every error message; (b) a missing
`species_model` is fatal exit 3 with the exact `animal-classifier train` invocation that produces it
(§12 gains a documented "produce the shipped artifacts" step with its command line); (c) a missing
`bird_model` is fatal **only when the bird path is reachable** — otherwise it is a startup `WARNING`
that bird refinement is disabled, with `provider_status='no_bird_model'` recorded, since PLAN.md makes
birds a refinement of an otherwise-working pipeline. Cover (b) in E22.

---

### NITs

12. **NIT — `blur_threshold` is an absolute constant compared against scores from two different reference edges.** §5.2 downscales only when the long edge exceeds 512 px and stores `blur_ref_edge` "so two scores are only ever compared with their reference edge in view", but §5.7 then compares every score against a fixed `100.0` regardless of that edge. A 320 px sharp image and a 512 px-normalised one are not on the same scale. Fix: define `blur_threshold` as calibrated at the 512 px reference, and state explicitly that smaller images are scored at native resolution and are therefore compared on a slightly different scale (accepted, with E11's small-sharp-image leg as the guard) — or normalise the score by `blur_ref_edge / 512`.
13. **NIT — §5.4's inference snippet hard-codes values that are config keys.** `new_shape=1280`, `conf_thres=0.20`, `iou_thres=0.45` appear as literals where `detector_image_size`, `detector_confidence` and `detector_iou` exist in §3. Fix: write them as `config.detector_image_size` etc. so no one ships the literal.
14. **NIT — `max_det=100` is undocumented and not configurable.** With no area floor (correctly), a herd photo can exceed 100 detections; the 101st box is dropped by NMS, which can in principle flip a `multiple` decision. Fix: name it in §3 as `detector_max_det = 100`, validate it, and note in §5.7 that it is a detector cap, not a size gate.
15. **NIT — E16 asserts "its bytes never read", which is not portably observable.** Fix: assert the observable consequences instead — `skipped(too_large)`, no `images`/`sources` row for that path, and no `DecodeError` in stderr.
16. **NIT — `/api/labels` omits `materializing` rows.** `status IN ('done','planned')` excludes exactly the pending-move state §5.8 creates, so a crashed re-tag makes an image vanish from the sidebar while its file is still on disk under the old label, inflating the `files_on_disk != count` warning. Fix: include `materializing` in the count, or surface it as a separate "in flight" number.
17. **NIT — the `sources` upsert path has no e2e test.** §5.9 specifies the content-changed-in-place upsert and that the previous `images` row and destination are retained; no test in E1–E26 exercises it. Fix: extend E14 — rewrite a card file's content in place between two runs, then assert one `sources` row, two `images` rows, and both destinations intact.
18. **NIT — the 2 px degenerate-crop constant is absent from §10.2's validation table.** It is the one size-ish constant in the pipeline; it is well justified in §5.3 (and correctly does not affect dominance), but it is invisible where every other threshold is listed. Fix: add a row stating it is a fixed, non-configurable resize precondition, not a gate.
19. **NIT — the GUI re-tag API demands a slug the user cannot be expected to type.** E21 asserts `Brewer's Blackbird` → 422, so correcting a bird means typing `brewer_s_blackbird`. Fix: have the label picker offer known labels from `/api/labels` + the artifact label space (display `common`, submit `key`), and state that the API deliberately accepts only slugs.

---

## Verified assumptions

Every number below was re-measured in-sandbox by `scripts/recon/verify_design_facts.py` (read-only,
committed) against `data/raw/annotations/instances_val2017.json`, using the design's own published
`split_for()`. **All 15 matched the document exactly** — the iteration-3 fact table is trustworthy.

| Design claim | Measured | Match |
|---|---|---|
| 2,666 non-crowd animal instances; 34 crowd dropped | 2,666 / 34 | ✅ |
| 1,016 images with ≥ 1 animal | 1,016 | ✅ |
| per-class counts `bird 427 … bear 71` | identical, all ten | ✅ |
| COCO split 813 train / 203 val via `split_for` | 813 / 203 | ✅ |
| 471 multi-animal images | 471 | ✅ |
| dominance GT at the 1.6 gate = 235 dominant / 236 multiple | 235 / 236 | ✅ |
| val-bucket ratio > 3.0 → 22 (E4 asserts ≥ 18) | 22, `ceil(0.8×22)=18` | ✅ |
| val-bucket ratio > 4.0 → 19 (remedy, asserts ≥ 16) | 19, `ceil(0.8×19)=16` | ✅ |
| val-bucket ratio < 1.3 → 29 (E4 asserts ≥ 24) | 29, `ceil(0.8×29)=24` | ✅ |
| `coco_species.json` candidates = 24 | 24 | ✅ (but see finding 8) |
| E7 7-class crops = 1,625 train / 349 val | 1,625 / 349 | ✅ |
| E7 majority-class baseline 0.229, chance 0.143 | bird 80/349 = 0.229, 1/7 = 0.143 | ✅ |
| `bear` has 11 val instances (so no recall threshold on it) | 11 | ✅ |

Also verified directly, not from the document:

- **Assets exist locally and match RECON.** `models/md_v5a.0.0.pt` = 280,766,885 B;
  `models/backbones/{efficientnet_b0_ra-3dd342df,convnext_nano_d1h-7eb4bdea}.pth`;
  `data/raw/{CUB_200_2011.tgz, val2017.zip, annotations_trainval2017.zip}`. Nothing in the design
  needs a blocked host at build or test time.
- **The published SQL schema is valid** and creates cleanly on sqlite 3.40.0, including the CHECK
  constraints and all four indexes. `rank` and `idx` are usable as column names (both are
  non-reserved in SQLite) — worth confirming, since `rank` is a window-function keyword.
- **`slug()` as published behaves as claimed**: `Brewer's Blackbird` → `brewer_s_blackbird`,
  `Chuck-will's-widow` → `chuck_will_s_widow`, `Ñandú` → `nandu`, `'熊'` and whitespace-only raise
  rather than returning a guess, and a 90-char name truncates to exactly 64 — the `LABEL_RE` bound.
  `Unknown`/`junk` do slugify onto reserved names, which is precisely why the `RESERVED_LABELS` check
  at `artifact.load()` is load-bearing; it is present.
- **`decide.py` is total.** Walking the published function: no-animals → blur branch; one animal →
  species-or-unknown; ≥ 2 → multiplication comparison (a zero-area second box makes the first
  dominant, no division). Every path returns a member of the closed set.
- **Git state**: branch `feat/safari-classifier`, HEAD `05029d3` is the iteration-3 design commit.

## Unverified / wrong assumptions

**Wrong:**

1. **CUB-200 names carry apostrophes and periods** (§5.8's rationale for `slug()`). They do not —
   `classes.txt` ships `022.Chuck_will_Widow`, already underscored, with a numeric prefix. The
   function is still needed (GUI input, mammal table), but the label-derivation consequence is a real
   gap → finding 2.
2. **"List lengths are data-driven, not hard-coded"** (§11.1) is true of `coco_dominance.json` only;
   `coco_species.json` is still fixed at 20 of 24 → finding 8.
3. **"train and inference see the same framing"** (§7.2) holds only while `crop_margin` is left at its
   default, which the config explicitly permits changing → finding 7.

**Unverifiable here, and correctly flagged as such by the document (no action):**

4. E4's aggregate thresholds against **real** MegaDetector output, E7's `val_top1 >= 0.55` and
   < 20 min, E10's `>= 0.90` and < 90 s. These can only be measured by running the suite; §11.2's
   remedy rule is the right mitigation. Finding 9 is about the *arithmetic between* E7's two
   thresholds, which is reviewable on paper, not about the value of either.
5. `ebird_enrich`'s `0.25` multiplier and `dist=50&back=30` — unfittable in-sandbox
   (`api.ebird.org` → `000`), honestly documented as chosen constants in §13.5. The matching rule they
   act on *is* fully specified and E23's stubbed leg covers it.
6. The 109-package resolution, `pillow-heif` HEIF **encoding**, and MegaDetector loading with
   `roboflow`/`sahi` overridden out. Claimed as measured in iteration 2 and consistent with RECON's
   environment notes; I did not re-resolve the lock file. §12.1 correctly makes re-running the three
   probes a precondition before any other work, which is the right guard.
7. Playwright is deliberately out of scope, so GUI *rendering* is untested by design. The reasoning
   (the risky behaviour is the re-tag rename, the 409 paths and validation, all covered at the HTTP +
   filesystem boundary) is sound.
