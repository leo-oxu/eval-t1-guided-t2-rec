# Data guide for model training

How to get the preprocessed M4Raw data and what is inside each file. Written for whoever builds the dataloader and trains the models (Weeks 3–4).

The files are produced by `notebooks/02_preprocess.ipynb`. Nothing here needs the raw M4Raw download.

## 1. Getting the data

### Full data: mount the shared Drive folder

Leo owns the shared `t1t2/` folder on Google Drive. To use it from your own account:

1. Open the share link, then **Organise → Add shortcut → My Drive**. Keep the name `t1t2`.
2. In Colab, mount Drive. The folder appears at `/content/drive/MyDrive/t1t2`.

Reading through a shortcut uses none of your own Drive storage. Only files you create inside the folder (for example checkpoints under `runs/`) count against your quota.

### Mini copy: for testing a pipeline

`t1t2_mini/` is a small copy with **exactly the same layout and file format**: 3 train, 1 val and 1 test subject, about 1 GB. It also exists as one file, `t1t2_mini.tar`, which is quick to copy to the Colab disk or to download to a laptop.

### The cell to paste at the top of a training notebook

`USE_MINI` is the only line that changes between the mini copy and the full data.

```python
USE_MINI = True                             # False = full data

import os, shutil, tarfile
from google.colab import drive
drive.mount("/content/drive")
SHARED = "/content/drive/MyDrive/t1t2"      # the shared project folder (your shortcut to it)

if USE_MINI:
    if not os.path.isdir("/content/t1t2_mini"):
        shutil.copy(f"{SHARED}/t1t2_mini.tar", "/content/t1t2_mini.tar")
        with tarfile.open("/content/t1t2_mini.tar") as tar:
            tar.extractall("/content")
    DATA_ROOT = "/content/t1t2_mini"
else:
    DATA_ROOT = SHARED
print("DATA_ROOT =", DATA_ROOT, "->", sorted(os.listdir(f"{DATA_ROOT}/processed")))
```

For real training, reading every batch straight from Drive is slow. Copy the splits you need to the Colab disk once per session and point `DATA_ROOT` at the copy:

```python
for split in ("train", "val"):
    shutil.copytree(f"{SHARED}/processed/{split}", f"/content/t1t2_full/processed/{split}", dirs_exist_ok=True)
shutil.copytree(f"{SHARED}/meta", "/content/t1t2_full/meta", dirs_exist_ok=True)
DATA_ROOT = "/content/t1t2_full"
```

## 2. Folder layout

The same under the full folder and under `t1t2_mini/`:

```
<DATA_ROOT>/
├── processed/
│   ├── train/<scan_id>.h5      128 subjects, about 21 GB
│   ├── val/<scan_id>.h5         30 subjects, about 5 GB
│   └── test/<scan_id>.h5        25 subjects, about 8 GB
└── meta/
    ├── splits.csv               scan_id, split, n_t1, n_t2, usable  (the authoritative subject list)
    ├── masks.npz                reference copy of the masks (you can ignore it, see section 5)
    └── audit_summary.json       headline numbers from the data audit
```

One file per subject, about 167 MB for train and val, and about twice that for test. The full folder also holds `raw_zips/`, `runs/`, `metrics/` and `figures/`, which the dataloader does not need.

## 3. What is in a subject file

Sizes: `S` = 18 slices, `C` = 4 coils, `P` = T2 repetitions (**3 in train and val, 6 in test**), `A` = 195 acquired phase-encode lines. Images are 256 × 256, ordered `[frequency encode, phase encode]`.

| Key | Shape | Dtype | Content |
| --- | --- | --- | --- |
| `t2_kspace` | `[P, S, C, 256, A]` | complex64 | Fully sampled T2 k-space of every repetition, **acquired lines only** |
| `pe_acquired` | `[256]` | bool | Which of the 256 phase-encode lines those `A` columns are (lines 30–224) |
| `t2_maps_f16` | `[P, S, C, 256, 256, 2]` | float16 | ESPIRiT coil maps, real and imaginary parts in the last axis. Estimated from the 12 ACS lines of that repetition only |
| `t2_rss` | `[P, S, 256, 256]` | float32 | Fully sampled root-sum-of-squares image of each repetition; build targets from these |
| `t1_avg` | `[S, 256, 256]` | float32 | T1 guide: magnitude mean of the first 3 T1 repetitions. **Not aligned** |
| `t1_rep1` | `[S, 256, 256]` | float32 | T1 guide from one repetition (experiment E5). Not aligned |
| `head_mask` | `[S, 256, 256]` | bool | Foreground mask for the loss and metrics. It covers the whole head, scalp included; it is not a skull-stripped brain mask |
| `shift_full` | `[2]` | float32 | T1→T2 shift in pixels `(FE, PE)`, estimated with the fully sampled T2. **Training only** |
| `shift_acs` | `[P, 2]` | float32 | The same shift estimated from each repetition's 12 ACS lines only. **Use for val and test** |
| `shift_per_slice` | `[S, 2]` | float32 | Per-slice version of `shift_full`, for diagnostics |

File attributes: `scan_id`, `split`, `n_acs` (12), `t1_reps` (3), `code_version`, `espirit`.

Things to know:

- **Conventions.** Centred, orthonormal FFT (`t1t2.fft.fft2c` / `ifft2c`, identical to fastMRI). The k-space centre is phase-encode line 128. Coil maps satisfy Σ|S|² = 1 in every pixel, so the magnitude of a coil-combined image is directly comparable to an RSS image.
- **Normalisation: none is applied.** Everything is stored in the scanner's original units; RSS image values reach roughly 100 and vary between subjects. Normalise in the dataloader. To avoid leakage, compute the scale from the undersampled input only, for example the 99th percentile of the zero-filled RSS image of each slice, and divide both the k-space and the target by it. Normalise T1 by its own 99th percentile.
- **Targets are magnitude images.** The repetitions are not phase-stable, so references are averaged in magnitude (RSS), not as complex data. Agreed rule: input = repetition `k`, target = mean of `t2_rss` over the other repetitions, rotating `k` for 3× training data.
- **Test subjects have 6 repetitions.** How many of them form the test reference is not decided yet; the files keep all 6.
- **T1 must be shifted before use.** Apply `t1t2.align.apply_shift(t1, shift)`, or let `read_subject` do it.

## 4. Loading

`t1t2.preprocess.read_subject` returns full-size arrays for one repetition: it puts the 195 acquired columns back into 256, converts the maps to complex and applies the T1 shift.

```python
import numpy as np
from t1t2 import masks
from t1t2.preprocess import read_subject

rep = 0
d = read_subject(f"{DATA_ROOT}/processed/train/2022061001.h5", rep=rep,
                 slices=[8, 9],        # optional; a sorted list of slice indices, default all
                 aligned="full")       # "full" for training, "acs" for val/test, None for unaligned T1

d["kspace"]       # [n, 4, 256, 256] complex64   fully sampled k-space of this repetition
d["maps"]         # [n, 4, 256, 256] complex64
d["t2_rss"]       # [P, n, 256, 256] float32     all repetitions
d["t1"]           # [n, 256, 256] float32        averaged T1, already shifted
d["t1_rep1"]      # [n, 256, 256] float32        single-repetition T1, already shifted
d["head_mask"]    # [n, 256, 256] bool
d["pe_acquired"]  # [256] bool
d["shift"]        # [2] the shift that was applied

mask = masks.equispaced_mask(d["pe_acquired"], R=6)             # [256] bool
kspace_us = d["kspace"] * mask                                  # undersampled input
others = [r for r in range(d["t2_rss"].shape[0]) if r != rep]
target = d["t2_rss"][others].mean(axis=0)                       # [n, 256, 256]
```

## 5. Undersampling masks: generate them on the fly

Masks are 1D along phase encoding (the last axis) and are built by `t1t2/masks.py` from `pe_acquired`, so there is nothing to load.

**R is the true net acceleration** (the fastMRI convention): a mask samples `round(195 / R)` lines in total, of which 12 are the fully sampled centre (ACS, lines 122–133) and the rest are outer lines. Lines that were never acquired are never sampled.

| R | Total lines | ACS | Outer | Realised net R |
| --- | --- | --- | --- | --- |
| 4 | 49 | 12 | 37 | 3.98 |
| 6 | 32 | 12 | 20 | 6.09 |
| 8 | 24 | 12 | 12 | 8.12 |
| 10 | 20 | 12 | 8 | 9.75 |

```python
masks.equispaced_mask(pe_acquired, R=6)            # main mask type: deterministic, same every call
masks.random_mask(pe_acquired, R=6, seed=0)        # second mask type: same line count, random outer lines
masks.acs_mask(256)                                # the 12 centre lines
```

Keep the default `n_acs=12`. The stored coil maps and `shift_acs` were computed from exactly those 12 lines, so a mask with a different centre would not match them.

## 6. Subject lists

`meta/splits.csv` under `DATA_ROOT` is the authoritative list: use the rows with `usable == True`. The split is the official M4Raw one (the same that TGVN released), and the motion-corrupted subjects are excluded.

The full ID lists are also in this repo, one scan ID per line:

| Split | Subjects | File |
| --- | --- | --- |
| train | 128 | [`splits/train.txt`](splits/train.txt) |
| val | 30 | [`splits/val.txt`](splits/val.txt) |
| test | 25 | [`splits/test.txt`](splits/test.txt) |

Mini copy: the first 3 train, 1 val and 1 test subjects in sorted order (expected `2022061001`, `2022061002`, `2022061003`; `2022061203`; `2022101101`). Its own `meta/splits.csv` lists exactly what it contains.

Do not tune anything on the test subjects, and use only the undersampled input or its ACS for anything computed at test time (alignment, normalisation, coil maps).
