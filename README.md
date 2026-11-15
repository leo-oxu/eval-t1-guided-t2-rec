# eval-t1-guided-t2-rec

Does a fully sampled T1 scan help reconstruct a fast, noisy, undersampled T2 scan at low field (0.3 T, M4Raw)? See [the project plan](T1-Assisted%20T2%20Reconstruction%20Project%20Plan.md).

## How the project is organised

Everything runs in Google Colab. The notebooks in `notebooks/` are the only things you run; the reusable code they share lives in the `t1t2/` package, so every notebook uses the same forward model, masks, alignment and metrics.

| # | Notebook | What it does | Runtime | Status |
| --- | --- | --- | --- | --- |
| 01 | `01_data_audit` | Download M4Raw, check zero-padded lines, FFT convention, split, T1–T2 shift | CPU | ready |
| 02 | `02_preprocess` | Coil maps, masks, references, T1 guide, alignment → one file per subject | GPU | ready |
| 03 | `03_classical_baselines` | Zero-filled and FISTA-TV, metric code | GPU | to do |
| 04 | `04_train` | One training notebook for every learned model, driven by a config cell | GPU | to do |
| 05 | `05_evaluate_main` | Test-set runs for E1 and E2 | GPU | to do |
| 06 | `06_robustness` | E3–E6: misaligned, wrong, blank or noisy T1 | GPU | to do |
| 07 | `07_figures_stats` | Wilcoxon tests, all figures and tables | CPU | to do |

Notebooks never pass data to each other directly. Each reads and writes files in one Drive folder, so any of them can be re-run after a disconnect.

## Running a notebook

1. Open it in Colab from GitHub: `https://colab.research.google.com/github/leo-oxu/eval-t1-guided-t2-rec/blob/main/notebooks/<name>.ipynb`
2. Run the setup cell. It mounts Drive, pulls the latest code and makes `t1t2` importable.
3. Leave `SMOKE = True` for a first trial (3 validation subjects, outputs under `smoke/`), then set it to `False` for the real run.

If you edit a file in `t1t2/` inside Colab, commit and push it before the session ends, or the change is lost.

## Drive layout

```
MyDrive/t1t2/
├── raw_zips/      M4Raw zips cached from Zenodo (about 21 GB; can be deleted after notebook 02)
├── meta/          manifest.csv, splits.csv, alignment.csv, masks.npz, audit_summary.json
├── processed/     <split>/<scan_id>.h5, one per subject (about 35 GB in total)
├── runs/          <run_name>/ checkpoints and logs
├── metrics/       per-slice metric CSVs
├── figures/
├── t1t2_mini/     small copy of processed/ and meta/ (3 train, 1 val, 1 test subject) for testing a pipeline
├── t1t2_mini.tar  the same, as one file (about 1 GB)
└── smoke/         the same folders, for SMOKE = True runs
```

Leo owns the shared `t1t2/` folder; Jackie mounts it through a shortcut in his own My Drive, which uses none of his storage.

**Training on this data:** see [t1t2/README.md](t1t2/README.md) for the file format, the subject lists, the mini copy and the cell that sets `DATA_ROOT`.

## Tests

`python -m pytest tests -q` checks the data pipeline on a small synthetic dataset in the M4Raw file format (`tests/synth.py`).
