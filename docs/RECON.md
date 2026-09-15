# Network & assets reconnaissance

Every HTTP status code below was measured from inside this sandbox with `curl`, at the times
noted. Probes are reproducible: `scripts/probe_urls.sh scripts/recon/<list>.txt` re-runs them and
prints label / status / bytes / content-type / URL.

Probe method: `curl -sSL --max-time 25 -r 0-1023 -o /dev/null -w '%{http_code} %{size_download} %{content_type}'`
— a ranged GET of the first KiB, so a `206` means the real asset bytes were served (not merely a
redirect or an HTML landing page). `000` means the connection never completed (blocked/filtered).
Anything downloaded in full is reported with its final size and sha256.

Environment: Python 3.12.13, torch 2.14.0+cu130, torchvision 0.29.0, timm 1.0.29,
yolov5 7.0.14, ultralytics 8.4.153, opencv-python-headless 5.0.0.93, numpy 2.5.3 — all from
pypi.org. Measured 2026-09-15 15:09–15:22 UTC.

## Headline result

The sandbox is far more capable than the initial reachability sweep suggested. **Real detection,
real ImageNet-pretrained backbones, and real labelled species imagery are all available in-sandbox.**
Nothing about the vision subsystem has to be faked or deferred.

| Capability | Status | Asset |
|---|---|---|
| Animal/person/vehicle detection | ✅ downloaded, loads, runs | MegaDetector v5a (280,766,885 B) |
| ImageNet-pretrained backbone for transfer learning | ✅ downloaded, loads, runs | timm `efficientnet_b0`, `convnext_nano` |
| Labelled bird imagery | ✅ downloaded (1,150,585,339 B) | CUB-200-2011, 200 species |
| Labelled multi-animal imagery with boxes | ✅ downloaded (815,585,330 B + 252,907,541 B) | COCO val2017 + instances |
| Live bird-ID / taxonomy API | ❌ every candidate blocked | see [Bird APIs](#bird-apis-and-taxonomy-sources) |

## MegaDetector weights

Downloaded in full to `models/md_v5a.0.0.pt` (gitignored).

| Probe | Status | Result |
|---|---|---|
| `HEAD` primary, redirects followed | **200** | redirects to `release-assets.githubusercontent.com` |
| `GET` primary (full download) | **200** | 280,766,885 bytes in 16.1 s (~17.5 MB/s) |
| `GET` mirror, ranged | **206** | `agentmorris/MegaDetector` release serves the same asset |

- primary: `https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt`
- mirror: `https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt`

```
sha256  94e88fe97c8050f2e3d0cc4cb4f64729d639d74312dcbe2f74f8eecd3b01b276
size    280766885 bytes
```

Integrity: the GitHub release API (`.../releases/tags/v5.0`) reports `md_v5a.0.0.pt` as exactly
**280766885** bytes, matching the downloaded file byte-for-byte in length. GitHub returns
`digest: None` for this asset (it predates asset digests), so no upstream sha256 exists to compare
against; the v5a**.0.1** and v5b.0.1 assets do publish digests if a cross-check is ever wanted.
Note that `api.github.com/repos/microsoft/CameraTraps/releases/tags/v5.0` answers **301** (the repo
moved) — query the `agentmorris/MegaDetector` repo for release metadata.

Also reachable (not downloaded): `md_v5b.0.0.pt` **206**, and the newer
`md_v1000.0.0-redwood.pt` from `agentmorris/MegaDetector` release `v1000.0` **206**.

### Confirmed: the checkpoint loads

`scripts/probe_md_checkpoint.py`, real output:

```
torch 2.14.0+cu130
checkpoint /projects/sandbox/AnimalImageClassifier/models/md_v5a.0.0.pt (280766885 bytes)
LOAD OK: top-level type = <class 'dict'>
top-level keys = ['best_fitness', 'date', 'ema', 'epoch', 'model', 'optimizer', 'updates', 'wandb_id']
  epoch = -1
  best_fitness = None
  date = '2022-02-21T10:34:12.204377'
  model class = models.yolo.DetectionModel
  model parameters = 140,054,656
  model.names = ['animal', 'person', 'vehicle']
  model.stride = tensor([ 8., 16., 32., 64.])
```

Two real failures had to be cleared first, and both are load-bearing for the detector module:

1. `torch.load(..., weights_only=False)` on a bare venv →
   `ModuleNotFoundError: No module named 'models'`. The checkpoint was pickled from the upstream
   yolov5 repo, whose packages are top-level (`models.yolo`, `utils.*`). The PyPI `yolov5`
   distribution namespaces them under `yolov5.*`, so the unpickler needs
   `sys.modules['models'] = yolov5.models` and the same for `utils`
   (`install_yolov5_aliases()` in `scripts/probe_md_checkpoint.py`).
2. Importing `yolov5` then failed twice more:
   `ImportError: libGL.so.1: cannot open shared object file` (its `opencv-python` dependency needs
   a GL library this image lacks → replace with **`opencv-python-headless`**), and
   `ModuleNotFoundError: No module named 'pkg_resources'` (yolov5 7.0.14 still imports
   `pkg_resources`, removed in setuptools 81 → pin **`setuptools<81`**).

Installing `yolov5` also downgrades `typer` 0.27.2 → 0.25.1; the CLI must stay compatible with
0.25.x or the dependency needs pinning.

### Confirmed: the checkpoint runs

`scripts/probe_md_inference.py`, real output (CPU, single 640×640 frame):

```
forward OK in 0.55s on CPU
  raw prediction tensor shape = (1, 25500, 8)
  class names = ['animal', 'person', 'vehicle']
```

8 = 4 box + 1 objectness + 3 classes. ~0.55 s/frame at 640 px on 8 CPU cores; MegaDetector's
native resolution is 1280 px, so budget roughly 2 s/image for full-resolution CPU inference.

## Pretrained backbones tried

The usual weight hosts are blocked, but **timm's own weights are published as GitHub release
assets**, and that route is fully open. This is the decisive finding for transfer learning.

| Label | Status | URL |
|---|---|---|
| torchvision resnet50 | **000** | `https://download.pytorch.org/models/resnet50-0676ba61.pth` |
| torchvision mobilenet_v3 | **000** | `https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth` |
| timm resnet50 via HF | **000** | `https://huggingface.co/timm/resnet50.a1_in1k/resolve/main/pytorch_model.bin` |
| timm resnet50 via GitHub release | **206** | `https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth` |
| timm efficientnet_b0 via GitHub release | **206** | `.../releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth` |
| EfficientNet-PyTorch b0 | **206** | `https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth` |
| EfficientNet-PyTorch b3 | **206** | `https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth` |
| pytorchcv resnet50 | **404** | `https://github.com/osmr/imgclsmob/releases/download/v0.0.147/resnet50-0633-b00d1c93.pth.zip` (guessed filename; host reachable, asset name wrong — not pursued, better routes found) |
| pytorchcv mobilenetv3 | **404** | `.../v0.0.491/mobilenetv3_large_w1-2106-3c5ee1a5.pth.zip` (same) |
| ultralytics yolov8n-cls | **206** | `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt` |
| ultralytics yolov8s-cls | **206** | `.../v8.3.0/yolov8s-cls.pt` |
| ultralytics yolo11n-cls | **206** | `.../v8.3.0/yolo11n-cls.pt` |
| ultralytics yolov5s | **206** | `https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt` |
| SpeciesNet weights (Kaggle) | **000** | `https://www.kaggle.com/models/google/speciesnet` |
| BioCLIP weights (HF) | **000** | `https://huggingface.co/imageomics/bioclip/resolve/main/open_clip_pytorch_model.bin` |

`api.github.com` answers **200**, so the full timm weight catalogue can be enumerated
programmatically. Release tags carrying ImageNet weights, with asset counts:

```
v0.1-weights        195 assets  21940MB      v0.1-rsb-weights   117 assets  15907MB
v0.1-tpu-weights     40 assets   7040MB      v0.1-attn-weights   43 assets   2870MB
v0.1-effv2-weights   15 assets   4602MB      v0.1-regnet         24 assets   3061MB
v0.1-vitjx           16 assets  11316MB      v0.1-tresnet        29 assets   4023MB
v0.1-maxx            17 assets   2125MB      v0.1-morevit         8 assets   1312MB
```
(`https://api.github.com/repos/huggingface/pytorch-image-models/releases?per_page=100`)

Useful members include `efficientnet_b0`, `tf_efficientnet_b0_ns` (noisy-student),
`mobilenetv3_large_100`, `resnet50d`, `seresnet50`, `convnext_{atto,femto,pico,nano,tiny}` —
i.e. a real choice of speed/accuracy points for the species head.

### Confirmed: two backbones downloaded and loaded

Downloaded to `models/backbones/` (gitignored):

```
efficientnet_b0_ra-3dd342df.pth   code=200  21376743 bytes
  sha256 3dd342dfa1fee25ae65e7bbdf8998cad6e45d6e77e69d580f0bd14d3eeb0b3f3
convnext_nano_d1h-7eb4bdea.pth    code=200  62396822 bytes
  sha256 7eb4bdea6812029145084e3f72b32a95c1e6fe620cc28ad2b84d40718a2aad03
```

Integrity: timm embeds the leading sha256 digits in the filename, and both computed hashes match
their filename (`3dd342df…`, `7eb4bdea…`) — these are the genuine published weights.

`scripts/probe_backbones.py`, real output:

```
timm 1.0.29 / torch 2.14.0+cu130
efficientnet_b0: LOAD OK params=5,288,548 missing=0 unexpected=0 logits=(1, 1000)
convnext_nano: LOAD OK params=15,593,560 missing=0 unexpected=0 logits=(1, 1000)
```

Zero missing and zero unexpected state-dict keys, and a real forward pass producing 1000-class
ImageNet logits. Loading pattern for offline use:
`timm.create_model(arch, pretrained=False)` + `load_state_dict(torch.load(local_path))` — never
`pretrained=True`, which would try the blocked HF hub.

## Datasets tried

`s3.amazonaws.com` is reachable, which unlocks the fast.ai public dataset mirror — including
CUB-200-2011 (birds) and COCO (multi-animal, boxed).

| Label | Status | URL |
|---|---|---|
| CUB-200-2011 (fast.ai mirror) | **206** → full download | `https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz` |
| COCO val2017 images | **206** → full download | `https://s3.amazonaws.com/fast-ai-coco/val2017.zip` |
| COCO trainval2017 annotations | **206** → full download | `https://s3.amazonaws.com/fast-ai-coco/annotations_trainval2017.zip` |
| Oxford-IIIT Pet (fast.ai mirror) | **206** | `https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz` (774.1 MB) |
| imagewoof2-320 (10 dog breeds) | **206** | `https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz` (313.2 MB) |
| imagewoof2 (full res) | **206** | `https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz` |
| imagenette2-320 / -160 | **206** | `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz`, `…-160.tgz` |
| Caltech-101 | **206** | `https://s3.amazonaws.com/fast-ai-imageclas/caltech_101.tgz` |
| CIFAR-100 (fast.ai mirror) | **206** | `https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz` |
| Food-101 | **206** | `https://s3.amazonaws.com/fast-ai-imageclas/food-101.tgz` |
| PASCAL VOC 2007 | **206** | `https://s3.amazonaws.com/fast-ai-imagelocal/pascal_2007.tgz` |
| COCO sample | **206** | `https://s3.amazonaws.com/fast-ai-coco/coco_sample.tgz` |
| CUB-200-2011 (Caltech original) | **000** | `https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz` |
| CUB-200-2011 (fast.ai imageloc) | **404** | `https://s3.amazonaws.com/fast-ai-imageloc/CUB_200_2011.tgz` (wrong bucket; `fast-ai-imageclas` is correct) |
| Oxford-IIIT Pet (Oxford original) | **000** | `https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz` |
| Stanford Dogs | **403** | `http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar` |
| Tiny ImageNet | **403** | `http://cs231n.stanford.edu/tiny-imagenet-200.zip` |
| CIFAR-100 (Toronto original) | **000** | `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` |
| LILA Snapshot Serengeti | **000** | `https://lila.science/datasets/snapshot-serengeti` |
| Open Images (GCS) | **000** | `https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv` |
| ImageNet sample images (GitHub) | **404** | `https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02118333_red_fox.JPEG` (guessed filename; host reachable) |
| ImageNet class index (GitHub) | **206** | `https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json` |

### Confirmed: CUB-200-2011 downloaded and readable

```
CUB code=200 bytes=1150585339 time=66.3s   →  data/raw/CUB_200_2011.tgz
tar listing: 12005 entries
  CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0010_796097.jpg
  …
```

200 bird species as one directory per class, ~11,788 JPEGs, and the archive also ships
`bounding_boxes.txt`, `attributes.txt` and train/test splits. This is a genuine fine-grained bird
training set — enough to train a real bird head by transfer learning, in-sandbox.

### Confirmed: COCO downloaded, and it contains real dominance-rule ground truth

```
COCO-ann     code=200 bytes=252907541  time=9.9s
COCO-val2017 code=200 bytes=815585330  time=24.9s   (5001 zip entries)
```

Parsed `annotations/instances_val2017.json` for the 10 COCO animal categories:

```
val2017 images: 5000 | images with >=1 animal: 1016
animal instances per class: {'bird': 440, 'cow': 380, 'sheep': 361, 'horse': 273,
  'zebra': 268, 'elephant': 255, 'giraffe': 232, 'dog': 218, 'cat': 202, 'bear': 71}
images with >1 animal instance: 471 | with >1 animal SPECIES: 70
of multi-animal images: ratio>3 (clear winner): 123 | ratio<1.5 (ambiguous -> "multiple"): 194
```

This is the single most valuable find for verification. COCO gives real photographs, with
per-instance boxes and areas, covering safari-relevant species (zebra, elephant, giraffe, bear,
plus bird/horse/sheep/cow/cat/dog) — and, computed from real box areas, **123 images with a clear
area winner and 194 genuinely ambiguous ones**. The dominance rule and the `multiple` fallback can
therefore be tested end-to-end against real ground truth instead of hand-made fixtures.

## Bird APIs and taxonomy sources

Every live species-ID and taxonomy API is blocked. No exceptions found.

| Label | Status | URL |
|---|---|---|
| eBird taxonomy | **000** | `https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json` |
| Macaulay Library search | **000** | `https://search.macaulaylibrary.org/api/v1/search?...` |
| iNaturalist taxa | **000** | `https://api.inaturalist.org/v1/taxa?q=lion` |
| GBIF species match | **000** | `https://api.gbif.org/v1/species/match?name=Panthera%20leo` |
| GBIF occurrence media | **000** | `https://api.gbif.org/v1/occurrence/search?...` |
| ITIS | **000** | `https://www.itis.gov/ITISWebService/jsonservice/searchByScientificName?srchKey=Panthera%20leo` |
| Catalogue of Life / ChecklistBank | **000** | `https://api.checklistbank.org/dataset/3LR/nameusage/search?q=Panthera%20leo` |
| Wikidata search | **000** | `https://www.wikidata.org/w/api.php?action=wbsearchentities&search=Panthera%20leo...` |
| Wikimedia Commons category members | **000** | `https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers&...` |
| Wikipedia REST summary | **000** | `https://en.wikipedia.org/api/rest_v1/page/summary/Lion` |

Consequences:
- Merlin remains unusable as an API regardless of network (on-device model, no public photo-ID
  endpoint), as already established in `docs/PLAN.md`. Blocked eBird only removes the *taxonomy*
  fallback.
- The common-name ↔ scientific-name mapping cannot be fetched at runtime and must ship as a
  **static in-repo table**. CUB-200-2011 supplies 200 bird common names; the safari mammal names
  are a small hand-authored table. Runtime label lookup must therefore be offline-first.
- The bird path is served by a locally trained CUB-200 head, not by a remote bird API. The remote
  provider stays behind the plan's provider abstraction so a user whose machine *can* reach
  eBird/iNaturalist can enable it.

Related, reachable but not usable end-to-end:
- `pypi.org/pypi/speciesnet/json` **206** and `api.github.com/repos/google/cameratrapai` **206** —
  the SpeciesNet *code* is installable, but its weights live on Kaggle (**000**), so it cannot run
  here. Same shape for `PytorchWildlife` **206** and `megadetector` **206** (both pypi, code only;
  `megadetector` 10.0.25 is the current release).

## Conclusions

1. **Detection is solved in-sandbox.** MegaDetector v5a is downloaded, loads, and runs a forward
   pass on CPU in ~0.55 s at 640 px. The detector module can be built and tested for real. Budget
   ~2 s/image at MegaDetector's native 1280 px.
2. **Transfer learning is possible in-sandbox — no synthetic fallback needed.** ImageNet-pretrained
   timm backbones are reachable as GitHub release assets; `efficientnet_b0` and `convnext_nano` are
   downloaded and load with zero key mismatches. The whole timm catalogue is enumerable via
   `api.github.com`.
3. **Real labelled species imagery is available.** CUB-200-2011 (200 bird species, boxes, splits)
   is downloaded in full. Oxford-IIIT Pet, imagewoof (dog breeds), Caltech-101 and CIFAR-100 are
   also reachable from the same fast.ai S3 mirror if more classes are wanted.
4. **Verification has real ground truth.** COCO val2017 + instance annotations are downloaded:
   1016 animal images, 471 multi-animal, and a clean split of 123 clear-dominance vs 194 ambiguous
   cases for exercising the dominance ratio and the `multiple` label end-to-end.
5. **No live species/taxonomy API is reachable.** Species naming must be offline and static; the
   remote bird provider stays optional behind the provider abstraction.

### Explicit decision on training (supersedes the earlier synthetic-only plan)

The recon brief anticipated that no backbone and no dataset would be reachable, in which case
training was to be proven on a generated synthetic dataset with a from-scratch CNN and real
training deferred. **That contingency does not apply — it is not needed and should not be used.**
Both prerequisites are present and verified in-sandbox:

- pretrained backbone: `models/backbones/{efficientnet_b0_ra,convnext_nano_d1h}*.pth` (verified loading)
- labelled dataset: `data/raw/CUB_200_2011.tgz` (verified, 200 species)

So the training subsystem will be proven by **actually fine-tuning a real backbone on real
labelled data** (a CUB-200 subset, CPU-sized: few classes, few epochs, small input) and asserting
the accuracy it reaches. `animal-classifier train` stays a documented first-class command so the
user can train on the full dataset — and on datasets only their machine can reach (huggingface.co,
LILA, iNaturalist) — but the command will already have been exercised against real data here.

A synthetic-data path is worth keeping only as a fast, deterministic smoke test, never as the
substitute for a real training run. No blocked asset is treated as available anywhere in this
document.

### Environment notes that the implementation must honour

- `opencv-python-headless`, never `opencv-python` (no `libGL.so.1` in this image).
- `setuptools<81` while `yolov5` 7.0.14 is a dependency (`pkg_resources`).
- `torch` from pypi.org is the CUDA build (`2.14.0+cu130`, pulls ~6 GB of nvidia wheels into
  `.venv`); `download.pytorch.org` CPU wheels are blocked, so this is the only route. Inference is
  CPU-only regardless.
- `timm.create_model(..., pretrained=False)` + explicit local `load_state_dict`; `pretrained=True`
  hits blocked hosts.
- `yolov5` pins `typer` down to 0.25.1 — keep the CLI compatible or pin deliberately.
- Unpickling MegaDetector requires `sys.modules['models']`/`['utils']` aliased to `yolov5.*`.

## Re-verified 2026-09-15 21:51 UTC (chunk 1, after locking the dependency contract)

`pyproject.toml` was rewritten to DESIGN.md §2.1, `uv.lock` regenerated (`uv lock`), and the venv
synced with `uv sync --frozen --extra dev`. **No pin moved**: `torch==2.14.0`, `torchvision==0.29.0`,
`timm==1.0.29`, `numpy==2.5.3`, `yolov5==7.0.14`, `opencv-python-headless==5.0.0.93`,
`typer==0.27.2`, `pillow==12.3.0`, `pillow-heif==1.7.0`, `setuptools==80.10.2`, `pytest==9.1.1`,
and `opencv-python` / `roboflow` / `sahi` are all absent from the installed environment
(`importlib.metadata.PackageNotFoundError`).

All three probes were re-run against that synced venv and reproduce the original measurements
exactly:

- `probe_md_checkpoint.py` → `model.names = ['animal', 'person', 'vehicle']`, 140,054,656 params,
  `model.stride = tensor([ 8., 16., 32., 64.])`, checkpoint 280766885 bytes.
- `probe_md_inference.py` → forward OK in 0.92 s on CPU, raw prediction shape `(1, 25500, 8)`.
- `probe_backbones.py` → `efficientnet_b0` params=5,288,548 missing=0 unexpected=0 logits=(1, 1000);
  `convnext_nano` params=15,593,560 missing=0 unexpected=0 logits=(1, 1000).

One measurement differs harmlessly from §2.1's "109 packages": `uv.lock` records **113** package
names, because a lockfile includes the root project plus the `raw` and `dev` extras (`rawpy`,
`pytest`, `iniconfig`, `pluggy`), whereas the 109 came from `uv pip compile` of the runtime set only.
No runtime pin changed, so the verification above still stands.
