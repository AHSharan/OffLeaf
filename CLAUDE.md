# OffLeaf — CLAUDE.md

Context file for future Claude Code sessions. Keep this updated after every phase.

## 1. What this project is

**Measuring and reducing background reliance in plant leaf disease classifiers**
(lab-to-field generalization).

Leaf disease classifiers trained on PlantVillage reach very high lab accuracy but
collapse on field photographs. A major cause is *shortcut learning*: the model keys
on the uniform lab background and imaging setup rather than on the lesion. This repo
measures that reliance and tests interventions that reduce it.

Three students, 8 weeks. **Optimize for correctness, reproducibility, and simple
interfaces between modules — not cleverness.** Plain PyTorch, no Lightning.

Naming: the repo and Python package are `offleaf`. In code the penalty is
`offleaf_loss` and the metrics are `offleaf_mass` and `offlesion_mass`.
(The spec was originally drafted using the name `plantshortcut`; that name is retired.)

GitHub: https://github.com/AHSharan/OffLeaf

## 2. Machine and environment

Measured on the dev machine, 2026-09-15:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | **6.0 GiB** (6441926656 bytes) |
| Compute capability | sm_86 (Ampere), 30 SMs |
| Driver | 596.08 (supports up to CUDA 13.2) |
| Python | **3.11.4** (`py -3.11`) |
| Torch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19) |
| Repo root | `F:\OffLeaf` (external USB SSD, PiBOX INSPIRE, 1.9 TB, exFAT) |

Verified working: `torch.cuda.is_available() -> True`, real matmul on device OK.

Note: `total_memory // 2**30` prints `5`, because the card reports 5.9995 GiB and
floor division rounds down. The card is a 6 GB card. Do not "fix" this.

Activate the venv:

```bash
.venv/Scripts/activate        # bash
.venv\Scripts\Activate.ps1    # PowerShell
```

### 6 GB VRAM consequences — read before training

6 GB is below what several spec defaults assume. Known pressure points:

- **CAM penalty** uses `autograd.grad(..., create_graph=True)`, i.e. double
  backward. This roughly doubles activation memory. Expect to reduce
  `batch_size` for `regime: cam_penalty` relative to `baseline`. Batch size is
  per-config, so record the value actually used in `config.yaml`.
- **SAM ViT-B** at 1024px is close to the limit. `masks/leaf_masks.py` must
  support the `--mobile_sam` flag (spec section 3 requires it below 8 GB VRAM).
- **SegFormer-B0 at 512px** (S0) will need a small batch size plus AMP.
- Always use AMP. Always `torch.cuda.empty_cache()` between sweep runs.

Batch sizes must never be hardcoded — they come from the config, so a run on a
bigger GPU reproduces by changing the yaml only.

## 3. Repo layout

```
OffLeaf/
  configs/          yaml per run (E0..E5, S0)
  data/             loaders, class_map.csv, splits/
  masks/            SAM leaf masks - lesion segmenter - pseudo-labels - verification - CVAT import
  counterfactual/   background-swap set builder
  models/           timm backbones with layer-3 feature hook
  train/            single training loop, regimes
  explain/          saliency methods -> normalized heatmaps
  eval/             all metrics, evaluate.py
  release/          HF push, model card template, gradio demo
  scripts/          one-line entry points
  tests/            pytest
  README.md  CLAUDE.md  requirements.txt  pyproject.toml
```

Untracked, on disk only: `data/raw/`, `data/masks/`, `data/counterfactual/`,
`runs/`, `weights/`, `.venv/`.

## 4. Conventions

- Python 3.10+, type hints everywhere.
- `argparse` + yaml configs. **Every script accepts `--config`.**
- All randomness seeded from `config.seed`.
- Every run writes `runs/<exp_id>/<seed>/` containing
  `config.yaml`, `metrics.json`, `checkpoint.pt`, `log.csv`.
- CSV logging by default; W&B behind a flag.
- **No notebooks in the repo.**
- **No hardcoded paths outside configs.**
- Never commit anything under `data/raw`, `data/masks`, `runs/`, `weights/`.
- Never silence exceptions.
- **Never compute a reported metric on pseudo masks.** See section 6.

## 5. Datasets and paths

| Dataset | Path | Role |
|---|---|---|
| PlantVillage | `data/raw/plantvillage/{color,segmented}/<class>/*.jpg` | train (38 classes) |
| PlantDoc | `data/raw/plantdoc/{train,test}/<class>/*.jpg` | field, test-only |
| Tomato-Village | `data/raw/tomato_village/<class>/*.jpg` | field, test-only |
| Field-PlantVillage | `data/raw/field_plantvillage/<class>/*.jpg` | field, test-only |
| PlantSeg | `data/raw/plantseg/` | lesion masks (native format) |
| PlantDoc-Seg (opt) | `data/raw/plantdoc_seg/` | masks aligned to PlantDoc |
| LDSD (opt backup) | `data/raw/ldsd/` | backup |

Field datasets are **test-only**, except the E5 few-shot arm.

`data/class_map.csv` columns: `canonical_class, crop, disease, plantvillage_name,
plantdoc_name, tomato_village_name, field_pv_name, plantseg_name` (empty where no
counterpart). Loaders **warn** for any raw folder not in the map.

`data/splits.py` writes stratified 80/10/10 PlantVillage splits per seed as CSV
path lists. **Experiments always load splits from disk**, never re-split.

`data/dataset.py` exposes one class:
`LeafDataset(paths, labels, leaf_mask_dir=None, lesion_mask_dir=None, transform=None)`
returning `image, label, leaf_mask, lesion_mask` (zeros when a mask is absent).
Albumentations transforms, masks passed jointly so every geometric op applies
identically to image and masks. Also `get_transforms(split, copypaste=False,
copypaste_p=0.5, bg_bank=None)`.

### PlantDoc state (already downloaded)

- 28 train classes / 2344 images; 27 test classes / 236 images.
- **`Tomato two spotted spider mites leaf` has only 2 train images and 0 test
  images.** Decide explicitly whether to drop it; do not let it silently break
  stratified splitting.
- **99 files were renamed on checkout.** The upstream repo contains filenames
  that are illegal on Windows/exFAT (scraped URL query strings containing
  `?`, `&`, `=`) and 4 whose paths exceeded the 260-char limit. They were
  extracted from git by blob SHA and renamed. The mapping is recorded in
  `data/raw/plantdoc/_renamed_files.csv` (`original_path, sanitized_path`).
  All 2581 files in `HEAD` are present — nothing was dropped.

## 6. Lesion masks: the human/pseudo split is a correctness rule

Two directories, never mixed:

- `data/masks/lesion_human/<dataset>/...png` — human-drawn or human-verified.
  **Reported metrics use only these.**
- `data/masks/lesion_pseudo/<dataset>/...png` — model-generated.
  **Training may use these.**

The `relevance_mass` metric must *raise* when pointed at `lesion_pseudo/` unless
`--allow_pseudo` is passed, and must then label the output clearly as pseudo-based.
This is enforced by `tests/test_metrics_refuse_pseudo.py`.

## 7. Phase plan

| Phase | Contents | Status |
|---|---|---|
| 0 | Environment, datasets, SAM checkpoint, git | **done** |
| 1 | layout, requirements, `data/`, `models/`, `train/` baseline, `configs/E0` | in progress |
| 2 | `masks/` (leaf, import, segmenter, pseudo-label, verify), `counterfactual/` | not started |
| 3 | `cam_penalty`, `copypaste`, `bgremoval` regimes; all section 10 tests | not started |
| 4 | `explain/`, `eval/` metrics, `noyan_test.py` | not started |
| 5 | configs E1-E5, `run_all.sh`, `release/` | not started |

Phase 1 ends with a **2-epoch E0 ResNet-50 seed-0 smoke test**, output shown.

After each phase: update this file and `README.md` with exact commands, and list
any assumption a human should verify.

## 8. Machine-specific setup quirks

These are real and will bite a new contributor on this machine.

1. **The repo lives on an exFAT external drive.** exFAT records no ownership, so
   git refuses the repo until an exception is added:
   ```bash
   git config --global --add safe.directory F:/OffLeaf
   git config --global --add safe.directory F:/OffLeaf/data/raw/plantdoc
   ```
   `core.fileMode=false` is set locally for the same reason.
2. **exFAT has no journaling.** Eject the drive properly; push to GitHub often.
   Datasets are re-downloadable and checkpoints are re-trainable, but commit
   history is not — treat GitHub as the source of truth.
3. **Pin the drive letter** (`diskmgmt.msc`). The venv hardcodes absolute paths;
   if the drive mounts as something other than `F:` the venv breaks. If that
   happens, delete `.venv` and rebuild from `requirements.txt` — nothing else
   is lost.
4. **Kaggle auth** reads `~/.kaggle/access_token` (a bare token file, not
   `kaggle.json`). The username is introspected from the token. Never print,
   copy, or commit it.
5. Python 3.12 is registered with the `py` launcher on this machine but the
   executable is missing. Use `py -3.11`.
6. pip's temp dir is redirected to `F:\.pip-tmp` for large wheels, because `C:`
   has very little free space. Set `TEMP`/`TMP` before big installs.

## 9. Open decisions — need a human answer

- [ ] **Plan A vs Plan B for lesion masks.**
      Plan A = train a segmenter on PlantSeg and pseudo-label PlantVillage tomato.
      Plan B = hand-annotate in CVAT.
      Blocks all of `masks/` beyond `leaf_masks.py`. **Unanswered.**
- [ ] Whether to drop `Tomato two spotted spider mites leaf` (2 images, section 5).
- [ ] The spec's literal tree makes `data`, `train`, `eval`, `models` top-level
      importable names, which is collision-prone if the package is ever
      `pip install`-ed. Currently scripts are run from the repo root, so this is
      fine. Flagging it rather than silently restructuring.

## 10. Hard rules — do not violate

Do **not**: add metrics or regimes beyond the spec; use notebooks; hardcode paths
outside configs; silence exceptions; compute any reported metric on pseudo masks;
or claim a test passes without running it.
