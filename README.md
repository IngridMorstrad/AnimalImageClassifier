# animal-classifier

Walks an SD-card directory tree of safari photographs, detects and classifies the animals in
each image, and files each photo under `~/animal_pics/<label>/` by copy (default) or symlink
(`--link`). A local GUI shows every image with the label it received.

See [docs/PLAN.md](docs/PLAN.md) for the design and [docs/PROGRESS.md](docs/PROGRESS.md) for
the timestamped build log.

## Status: `classify` runs, but it cannot recognise a species yet

9 of the 26 chunks in [docs/IMPL-PLAN.md](docs/IMPL-PLAN.md) are done.

**`classify` works end to end.** It walks the card read-only, decodes every JPEG/PNG/TIFF/HEIC,
measures sharpness, applies the dominance rule and files each photo into
`<output>/<label>/` by copy, `--link` or `--hardlink`, recording everything in a SQLite
catalog. Re-running is a cheap no-op. `--dry-run` writes nothing.

**Two things are missing, and both are load-bearing:**

- **No detector runs on real photographs.** `--detector megadetector` exits 3 rather than
  pretending; only `--detector scripted` works today, and it reads boxes from a
  `<image>.boxes.json` sidecar. Without it every photo is filed as `landscape` or `junk`,
  so the tool cannot find animals on your card yet. Real MegaDetector v5a lands at chunk 14.
- **No species model exists.** Every detected animal is therefore filed as `unknown`, never as
  `lion` or `plains_zebra`. The dominance rule itself is fully working and tested — a single
  dominant animal lands in `unknown/`, a crowded frame in `multiple/` — it just has no names to
  attach. Species labels land at chunk 17.

`gui`, `train`, `eval`, `export-trainset` and `verify` are still placeholders that print
`not implemented yet` and exit 1.

## Setup

```bash
uv sync
```

That creates `.venv`, installs the project and its locked dependencies, and is the only
setup step. `uv` is the only Python tool used in this project.

## Commands

Run everything through `uv run`, which resolves the console script without activating
anything:

```bash
uv run animal-classifier classify /Volumes/SDCARD --output ~/animal_pics --detector scripted [--link]
uv run animal-classifier gui --output ~/animal_pics
uv run animal-classifier train <manifest>
uv run animal-classifier eval <manifest> --model models/species.pt
uv run animal-classifier export-trainset --output ~/animal_pics --destination trainset/
uv run animal-classifier verify
```

### Labels

Each photo is filed under exactly one label: a species, or one of `multiple`, `landscape`,
`junk`, `unknown`. Nothing is ever discarded.

> Images containing only people or vehicles are filed as `landscape` (or `junk` if blurry);
> their person/vehicle boxes are still kept in the catalog and drawn in the GUI.

There is **no minimum box size.** The only size comparison in the pipeline is
`--dominance-ratio` (default 1.6), and it is relative: a lone animal is filed as its species
however small it is in frame, and a distant impala beside a lion portrait leaves the photo
filed as the lion.

`uv sync` does install the script at `.venv/bin/animal-classifier`, so if you would rather
type the bare command, activate the venv first — otherwise your shell will not find it:

```bash
source .venv/bin/activate        # .venv\Scripts\activate on Windows
animal-classifier verify
```

## Tests

End-to-end only, no unit tests. The canonical command, matching the one the build gate uses:

```bash
uv run --frozen pytest tests/e2e -q
```
