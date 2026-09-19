# animal-classifier

Walks an SD-card directory tree of safari photographs, detects and classifies the animals in
each image, and files each photo under `~/animal_pics/<label>/` by copy (default) or symlink
(`--link`). A local GUI shows every image with the label it received.

See [docs/PLAN.md](docs/PLAN.md) for the design and [docs/PROGRESS.md](docs/PROGRESS.md) for
the timestamped build log.

## Status: complete

All 26 build chunks in [docs/IMPL-PLAN.md](docs/IMPL-PLAN.md) are done. Every command is
real: `classify`, `gui`, `train`, `eval`, `export-trainset`, `verify`.

The tool ships **no** model weights in git (they are large and reproducible); see
[docs/ARTIFACTS.md](docs/ARTIFACTS.md) for the exact commands that produce the MegaDetector
checkpoint and the species/bird heads. With no species model present, `classify` still runs
usefully: it detects animals, applies the dominance rule, and files each photo — a detected
animal with no trained head is filed as `unknown` rather than a guessed species.

## Quick start

```bash
uv sync                                                          # create .venv, install
uv run animal-classifier classify /Volumes/SDCARD -o ~/animal_pics
```

That is the whole setup. The first run downloads the MegaDetector v5a detector
(280 MB, once) and verifies it against a pinned sha256 before using it; later runs
reuse it. Pass `--no-download` on an offline or metered machine to refuse instead.

```bash
# Review and correct labels in the browser
uv run animal-classifier gui --output ~/animal_pics     # http://127.0.0.1:8765

# Turn your corrections into the next training set
uv run animal-classifier export-trainset --output ~/animal_pics --destination trainset.jsonl

# Check assets and catalog/filesystem consistency
uv run animal-classifier verify --output ~/animal_pics
```

**What you get without a species model:** real animal detection and the dominance
rule, so photos land in `unknown/` (one dominant animal), `multiple/`, `landscape/`
or `junk/`. To get actual species names (`zebra/`, `elephant/`) train a head — see
[docs/ARTIFACTS.md](docs/ARTIFACTS.md) — then add `--species-model models/species.acmodel`.

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

## Tests and lint

End-to-end only, no unit tests. Every test drives a real entry point. The canonical gate:

```bash
uv run --frozen ruff check              # the lint gate
uv run --frozen pytest tests/e2e -q     # the e2e suite
```

Tests marked `slow` train a model or run the real detector; they skip cleanly when the
280 MB checkpoint or the network is unavailable. Run only the fast ones with
`-m "not slow"`.
