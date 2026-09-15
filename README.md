# animal-classifier

Walks an SD-card directory tree of safari photographs, detects and classifies the animals in
each image, and files each photo under `~/animal_pics/<label>/` by copy (default) or symlink
(`--link`). A local GUI shows every image with the label it received.

See [docs/PLAN.md](docs/PLAN.md) for the design and [docs/PROGRESS.md](docs/PROGRESS.md) for
the timestamped build log.

## Setup

```bash
uv venv
uv pip install -e .
```

`uv` is the only Python tool used in this project.

## Commands

```bash
animal-classifier classify /Volumes/SDCARD --output ~/animal_pics [--link]
animal-classifier gui --output ~/animal_pics
animal-classifier train <manifest>
animal-classifier eval <manifest> --model models/species.pt
animal-classifier export-trainset --output ~/animal_pics --destination trainset/
animal-classifier verify
```

## Tests

End-to-end only:

```bash
uv run pytest
```
