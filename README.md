# animal-classifier

Walks an SD-card directory tree of safari photographs, detects and classifies the animals in
each image, and files each photo under `~/animal_pics/<label>/` by copy (default) or symlink
(`--link`). A local GUI shows every image with the label it received.

See [docs/PLAN.md](docs/PLAN.md) for the design and [docs/PROGRESS.md](docs/PROGRESS.md) for
the timestamped build log.

## Status: under construction — no command does its job yet

**Every command below is a placeholder that prints `not implemented yet` and exits 1.**
6 of the 26 chunks in [docs/IMPL-PLAN.md](docs/IMPL-PLAN.md) are done: `config.py`,
`catalog.py`, `taxonomy/`, `scan.py` and `images.py` exist and are exercised, but `detect/`,
`classify/`, `decide.py`, `materialize.py`, `training/` and `gui/` do not. Nothing writes to
the output directory yet.

`classify` first does real work at chunk 9, with the scripted test detector. Real
MegaDetector weights land at chunk 14 and real species labels at chunk 17.

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
uv run animal-classifier classify /Volumes/SDCARD --output ~/animal_pics [--link]
uv run animal-classifier gui --output ~/animal_pics
uv run animal-classifier train <manifest>
uv run animal-classifier eval <manifest> --model models/species.pt
uv run animal-classifier export-trainset --output ~/animal_pics --destination trainset/
uv run animal-classifier verify
```

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
