# OffLeaf

**Measuring and reducing background reliance in plant leaf disease classifiers.**

Leaf disease classifiers trained on PlantVillage reach very high lab accuracy and
then collapse on field photographs. A major cause is *shortcut learning*: the
model keys on the uniform lab background rather than on the lesion. This repo
measures that reliance and tests interventions that reduce it.

Plain PyTorch, no Lightning. See [CLAUDE.md](CLAUDE.md) for the full spec,
conventions and phase plan.

---

## Setup

Requires Python 3.10+ and, for training, an NVIDIA GPU.

```bash
py -3.11 -m venv .venv
.venv/Scripts/activate
python -m pip install --upgrade pip
```

Install PyTorch from the CUDA wheel index **first** — the default PyPI wheels on
Windows are CPU-only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then the rest. On Windows, install `stringzilla` first or the transaction aborts
(see the note in `requirements.txt`):

```bash
pip install "stringzilla==5.1.1"
pip install -r requirements.txt
```

Verify the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data

Datasets are **not** committed. Download them into `data/raw/`.

PlantVillage (needs `color/` **and** `segmented/` — the `bgremoval` regime trains
on `segmented/`):

```bash
kaggle datasets download abdallahalidev/plantvillage-dataset -p data/raw/_pv_full --unzip
```

Then arrange as `data/raw/plantvillage/{color,segmented}/<class_name>/*.jpg`.

PlantDoc:

```bash
git clone https://github.com/pratikkayal/PlantDoc-Dataset data/raw/plantdoc
```

> On Windows, PlantDoc's checkout partly fails: 87 upstream filenames contain
> characters illegal on Windows/exFAT (`?`, `&`, `=` from scraped URLs) and 4
> exceed the 260-character path limit. The renames applied here are recorded in
> `data/raw/plantdoc/_renamed_files.csv`. All 2581 files are present.

SAM ViT-B checkpoint:

```bash
curl -L -o weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Expected SHA-256: `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`

### No dataset to hand?

Generate a clearly-marked synthetic stand-in so the pipeline can be run end to
end. **Never report a metric computed on it.**

```bash
python data/make_synthetic.py --out data/raw/_synthetic --classes 3 --per-class 20
```

## Running

Splits are generated once and then always loaded from disk, so every regime at a
given seed sees identical train/val/test membership:

```bash
python data/splits.py --config configs/splits.yaml
```

Train one run. Every script takes `--config`:

```bash
python train/train.py --config configs/E0_resnet50_seed0.yaml
```

Smoke-test a config without a full run:

```bash
python train/train.py --config configs/E0_resnet50_seed0.yaml --epochs 2 --limit_batches 20
```

Each run writes `runs/<exp_id>/<seed>/` containing `config.yaml`, `metrics.json`,
`checkpoint.pt` and `log.csv`. Command-line overrides are written into the saved
`config.yaml`, so a run stays reproducible from its own output directory.

## Status

Phase 1 of 5. Implemented: `data/` (class map, splits, dataset), `models/`,
`train/` **baseline regime only**, `configs/E0`.

The `cam_penalty`, `copypaste` and `bgremoval` regimes raise `NotImplementedError`
by design — they depend on masks from Phase 2. See CLAUDE.md section 7 for the
phase plan and section 9 for open questions.
