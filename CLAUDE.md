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
| System RAM | **16 GiB** |
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

### 16 GB system RAM: the DataLoader is the memory problem, not the GPU

This bit twice during setup, and both failure modes are misleading.

On Windows, DataLoader workers are **spawned**, not forked, so each one imports
torch/scipy/cv2 afresh and commits roughly **2.3 GB**. Two symptoms, both of
which look like something else:

```
RuntimeError: bad allocation                     # num_workers: 4
SystemError: error return without exception set  # during a worker's import
```

Neither is a CUDA error. A real GPU OOM says **"CUDA out of memory"**. Both of
the above are *host* allocation failures, so **lower `num_workers` before you
touch `batch_size`** — the GPU is not the problem. Measured during the E0 smoke
test: batch 32 ResNet-50 at 224px uses only ~2.1 GB of the 6 GB VRAM.

The binding constraint is Windows **commit charge**, not physical RAM. Check it
with:

```powershell
(Get-Counter '\Memory\Committed Bytes').CounterSamples[0].CookedValue / 1MB
(Get-Counter '\Memory\Commit Limit').CounterSamples[0].CookedValue / 1MB
```

At the failure this machine was at **99%** (55,084 / 55,474 MB).

**Crashed runs leak.** A killed training run leaves its main process and its
spawned workers behind, still holding commit (12.9 GB across four processes in
one instance here) and sometimes GPU memory. They do not clean themselves up.
After any crash, check and clear them before re-running:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object ProcessId, @{n='CommitMB';e={[math]::Round($_.PageFileUsage/1KB)}}, CommandLine
```

Config guidance for this machine:

- `num_workers: 2` for training, and `val_num_workers: 0` (the default).
  Validation must not spawn a second set of workers while training workers are
  alive — that doubling is what exhausted commit.
- `persistent_workers: false` (the default) so train workers are torn down
  between epochs.
- `--num_workers 0` is the safe fallback; it spawns nothing and always runs.
- Closing Chrome frees a meaningful amount (24 processes during setup).

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
| PlantVillage | `data/raw/plantvillage/{color,grayscale,segmented}/<class>/*.jpg` | train (38 classes) |
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

### PlantVillage state (already downloaded)

Source: **`abdallahalidev/plantvillage-dataset`** on Kaggle (~2.2 GB zipped).

```bash
kaggle datasets download abdallahalidev/plantvillage-dataset -p data/raw/_pv_full --unzip
```

Then the three variant folders were moved up to `data/raw/plantvillage/`.

- `color/`, `grayscale/`, `segmented/` — 38 classes each, 54,305 images each
  (`segmented/` has one extra file, 54,306; harmless, not yet identified).
- All 38 folder names match `class_map.csv` exactly — `data/splits.py` ran with
  **no** unmapped-folder warnings, which is the check that the map is right.
- Splits: 43,444 train / 5,430 val / 5,431 test per seed.

**Do not substitute `mohitsingh1804/plantvillage`.** It was tried first and has
no `segmented/` variant at all — only colour images pre-split into `train/val`.
The `bgremoval` regime (E1c, E3b) trains on `segmented/`, so that dataset makes
two of the specified experiments impossible. The copy was deleted to avoid two
competing sources of truth.

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
7. **`stringzilla` breaks `pip install albumentations` on Windows.**
   albumentations 2.x depends on it; version 5.1.2 ships no cp311 win_amd64
   wheel, so pip tries to compile it and the **entire transaction aborts** —
   meaning nothing in the command gets installed, not just albumentations.
   Install the pinned version first:
   ```bash
   pip install "stringzilla==5.1.1"
   pip install -r requirements.txt
   ```
8. Disk space is tight on the internal drives (C: ~14 GB, D: ~6 GB free). Keep
   large artifacts on `F:`. 14 GB of stale pip cache was purged during setup;
   `pip cache purge` is the first thing to try if C: fills again.

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
