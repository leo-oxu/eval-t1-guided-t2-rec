"""Tiny synthetic stand-in for M4Raw, in the same on-disk format.

Used by the tests and for dry-running the notebooks without the 20 GB
download. Not real anatomy: a Shepp-Logan phantom with T2-like and T1-like
contrast, 4 birdcage coils, 195 of 256 PE lines acquired, a known T1-T2
shift and independent noise per repetition.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from t1t2.align import apply_shift
from t1t2.fft import fft2c, kspace_to_rss

TRUE_SHIFT = (1.5, 0.3)   # what must be applied to T1 to register it to T2 (FE, PE)
N, N_COILS, N_ACQ = 256, 4, 195


def _phantoms():
    import sigpy as sp
    t2 = np.abs(sp.shepp_logan((N, N))).astype(np.float32)
    support = t2 > 0
    t1 = np.where(support, 1.1 - 0.9 * t2 / t2.max(), 0).astype(np.float32)  # inverted contrast
    return t2 / t2.max(), t1 / t1.max()


def make_dataset(raw_dir, subjects=None, n_slices=3, noise=0.02, seed=0):
    """Write raw_dir/<split>/<scan_id>_<T1|T2><rep>.h5. Returns the true maps."""
    import sigpy.mri as mr
    subjects = subjects or {"train": ["2022010101", "2022010102"], "val": ["2022020101"],
                            "test": ["2022030101"]}
    rng = np.random.default_rng(seed)
    t2, t1 = _phantoms()
    t1 = apply_shift(t1, [-s for s in TRUE_SHIFT])           # T1 sits displaced from T2
    yy, xx = np.mgrid[:N, :N] / N - 0.5
    maps = mr.birdcage_maps((N_COILS, N, N)).astype(np.complex64)
    pe_acq = np.zeros(N, bool)
    start = (N - N_ACQ) // 2
    pe_acq[start:start + N_ACQ] = True

    for split, ids in subjects.items():
        (Path(raw_dir) / split).mkdir(parents=True, exist_ok=True)
        n_rep = 6 if split == "test" else 3
        for scan_id in ids:
            for contrast, img in (("T1", t1), ("T2", t2)):
                phase = np.exp(1j * 2.0 * (xx + 0.5 * yy))    # smooth image phase
                for rep in range(1, n_rep + 1):
                    vol = np.stack([img * (1 - 0.1 * s) for s in range(n_slices)])
                    k = fft2c(maps[None] * (vol * phase)[:, None])
                    k = k + noise * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape))
                    k = (k * pe_acq).astype(np.complex64)
                    path = Path(raw_dir) / split / f"{scan_id}_{contrast}{rep:02d}.h5"
                    with h5py.File(path, "w") as f:
                        f["kspace"] = k
                        rec = kspace_to_rss(k)
                        f["reconstruction_rss"] = rec
                        f.attrs["acquisition"] = "AX" + contrast
                        f.attrs["max"] = float(rec.max())
                        f.attrs["patient_id"] = scan_id
            (Path(raw_dir) / split / ".extracted").touch()
    return maps, pe_acq
