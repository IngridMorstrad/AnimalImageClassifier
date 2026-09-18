# Producing the shipped artifacts

The tool ships **no** model weights in git — they are large binaries, and the
sandbox that builds this repo cannot reach every training dataset. Instead, this
document is the reproducible recipe for producing each artifact on a machine where
the data and backbone weights are reachable (your laptop, a GPU box). Every command
is the real CLI; none of it is pseudo-code.

## 1. The detector — MegaDetector v5a (`models/md_v5a.0.0.pt`)

Not trained; downloaded. The loader verifies its size and sha256 before use, so any
other file is refused.

```bash
mkdir -p models
curl -L -o models/md_v5a.0.0.pt \
  https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt
# expected: 280766885 bytes,
#   sha256 94e88fe97c8050f2e3d0cc4cb4f64729d639d74312dcbe2f74f8eecd3b01b276
```

## 2. The species head (`models/species.acmodel`)

Finetune `efficientnet_b0` on COCO's ten animal categories. First stage the data and
build the manifest, then train.

```bash
# COCO val2017 images + annotations (≈1 GB)
mkdir -p data/raw && cd data/raw
curl -LO http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip
cd ../..

# The timm-native backbone (its key names match timm's efficientnet_b0)
mkdir -p models/backbones
curl -L -o models/backbones/efficientnet_b0_ra-3dd342df.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth

# Build the training manifest (train split only — §7.2's leakage rule)
uv run animal-classifier's builder:
uv run --frozen python scripts/build_coco_manifest.py \
  --annotations data/raw/annotations/instances_val2017.json \
  --images data/raw/val2017 \
  --out data/manifests/coco_species.jsonl --split train

# Train (224 px is the shipped resolution; ~20 min CPU, minutes on a GPU)
uv run --frozen animal-classifier train \
  --manifest data/manifests/coco_species.jsonl \
  --arch efficientnet_b0 --out models/species.acmodel \
  --backbone-weights models/backbones/efficientnet_b0_ra-3dd342df.pth \
  --input-size 224 --epochs-head 3 --epochs-finetune 5

# Evaluate and (optionally) calibrate into a NEW artifact
uv run --frozen python scripts/build_coco_manifest.py \
  --annotations data/raw/annotations/instances_val2017.json \
  --images data/raw/val2017 --out data/manifests/coco_val.jsonl --split val
uv run --frozen animal-classifier eval --model models/species.acmodel \
  --manifest data/manifests/coco_val.jsonl --split val \
  --calibrate --out models/species.cal.acmodel
```

## 3. The bird head (`models/birds.acmodel`)

Finetune on CUB-200-2011. The download host (`data.caltech.edu`) rate-limits; a
HuggingFace mirror works too. Build the manifest with `scripts/build_cub_manifest.py`
(honours the official `train_test_split.txt`), then `train --arch convnext_nano`.

```bash
# CUB-200-2011 (≈1.1 GB) into data/raw/CUB_200_2011/
uv run --frozen python scripts/build_cub_manifest.py \
  --root data/raw/CUB_200_2011 --out data/manifests/cub_birds.jsonl
uv run --frozen animal-classifier train \
  --manifest data/manifests/cub_birds.jsonl --arch convnext_nano \
  --out models/birds.acmodel --input-size 224
```

## Smoke path (no data required)

For a seconds-long end-to-end check that the whole train → eval → export → infer
path works, use the synthetic dataset and the from-scratch `tinycnn`:

```bash
uv run --frozen animal-classifier train --dataset synthetic --arch tinycnn \
  --out /tmp/synth.acmodel --input-size 64 --epochs-head 2 --epochs-finetune 2
```

## Why these are not in git

`.pt`, `.pth` and `.acmodel` files are gitignored. They are reproducible from the
commands above, they are large, and — for the detector — pinned by hash so a fetched
copy is provably the right one. `animal-classifier verify` reports which artifacts
are present and loadable.
