"""Turn one subject's raw M4Raw files into one compact h5 file.

Layout of ``processed/<split>/<scan_id>.h5`` (S slices, C coils, R_t2
repetitions, A acquired PE lines):

    t2_kspace      [R_t2, S, C, 256, A] complex64  acquired PE lines only
    pe_acquired    [256] bool                      where those lines sit
    t2_maps_f16    [R_t2, S, C, 256, 256, 2] f16   ESPIRiT maps from each rep's ACS
    t2_rss         [R_t2, S, 256, 256] float32     fully sampled RSS of each rep
    t1_avg         [S, 256, 256] float32           T1, mean of the first 3 reps
    t1_rep1        [S, 256, 256] float32           T1, single repetition (E5)
    head_mask      [S, 256, 256] bool                foreground incl. scalp
    shift_full     [2]        T1->T2 shift from fully sampled T2 (training)
    shift_acs      [R_t2, 2]  T1->T2 shift from each rep's ACS only (test time)
    shift_per_slice[S, 2]     per-slice version of shift_full (diagnostics)

T1 is stored unregistered; apply ``align.apply_shift(t1, shift)`` when
loading. References are built at load time from ``t2_rss`` (magnitude) or
``t2_kspace`` (complex), e.g. the mean of the repetitions other than the input.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

from . import __version__
from .align import apply_shift, lowres_pe, volume_shift
from .coils import espirit_maps, from_f16, to_f16
from .fft import kspace_to_rss
from .m4raw import acquired_pe, load_kspace
from .masks import N_ACS, acs_mask

T1_REPS = 3  # test subjects have 6 T1 reps; use 3 everywhere so the guide SNR matches


def head_mask(vol: np.ndarray, level: float = 0.1, corner: int = 24) -> np.ndarray:
    """Foreground (whole head, scalp included) mask per slice from a magnitude
    volume [S, H, W]. This is not a skull-stripped brain mask.

    The threshold sits level of the way from the background noise floor
    (measured in the four image corners) to the 99th percentile, on a smoothed
    volume. Otsu was tried first and cut out dark frontal tissue on real T2.
    """
    smooth = ndimage.gaussian_filter(vol.astype(np.float32), sigma=(0, 2, 2))
    c = corner
    floor = np.mean([smooth[:, :c, :c], smooth[:, :c, -c:], smooth[:, -c:, :c], smooth[:, -c:, -c:]])
    fg = smooth > floor + level * (np.percentile(smooth, 99) - floor)
    out = np.zeros_like(fg)
    for i, sl in enumerate(fg):
        lab, n = ndimage.label(sl)
        if n == 0:
            continue
        sizes = ndimage.sum(sl, lab, index=np.arange(1, n + 1))
        keep = lab == (1 + int(np.argmax(sizes)))
        out[i] = ndimage.binary_fill_holes(ndimage.binary_closing(keep, iterations=3))
    return out


def process_subject(files: dict[str, list[str]], out_path, scan_id: str, split: str,
                    n_acs: int = N_ACS, device: int = -1, espirit_kwargs: dict | None = None,
                    overwrite: bool = False) -> dict:
    """Process one subject and write its h5 file. Returns a small summary row.

    ``files`` is ``{'T1': [paths by rep], 'T2': [paths by rep]}``.
    The file is written to local disk first and then moved, so a session that
    dies mid-write never leaves a half-written file on Drive.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return dict(scan_id=scan_id, split=split, status="skipped")
    t0 = time.time()
    espirit_kwargs = espirit_kwargs or {}

    # ---- T2: k-space of every repetition --------------------------------
    t2_k = np.stack([load_kspace(p) for p in files["T2"]])        # [R, S, C, FE, PE]
    n_rep, n_slice, _, _, n_pe = t2_k.shape
    pe_acq = acquired_pe(t2_k)
    acs = acs_mask(n_pe, n_acs)
    if not pe_acq[acs].all():
        raise ValueError(f"{scan_id}: ACS lines are not all acquired")
    t2_rss = kspace_to_rss(t2_k)                                   # [R, S, FE, PE]
    t2_mean = t2_rss.mean(axis=0)

    # ---- T1 guide images --------------------------------------------------
    t1_rss = np.stack([kspace_to_rss(load_kspace(p)) for p in files["T1"][:T1_REPS]])
    t1_avg, t1_rep1 = t1_rss.mean(axis=0), t1_rss[0]
    if t1_avg.shape != t2_mean.shape:
        raise ValueError(f"{scan_id}: T1 {t1_avg.shape} and T2 {t2_mean.shape} differ")

    # ---- alignment --------------------------------------------------------
    shift_full, shift_per_slice = volume_shift(t2_mean, t1_avg)
    t1_low = lowres_pe(t1_avg, acs)
    shift_acs = np.stack([volume_shift(kspace_to_rss(t2_k[r] * acs), t1_low)[0]
                          for r in range(n_rep)])

    # ---- coil maps from each repetition's ACS -----------------------------
    maps = np.zeros(t2_k.shape[:3] + (t2_k.shape[3], n_pe, 2), dtype=np.float16)
    for r in range(n_rep):
        for s in range(n_slice):
            maps[r, s] = to_f16(espirit_maps(t2_k[r, s] * acs, n_acs=n_acs,
                                             device=device, **espirit_kwargs))

    mask = head_mask(t2_mean)

    # ---- write (local temp file, then move) -------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    with h5py.File(tmp, "w") as f:
        f["t2_kspace"] = t2_k[..., pe_acq]
        f["pe_acquired"] = pe_acq
        f["t2_maps_f16"] = maps
        f["t2_rss"] = t2_rss
        f["t1_avg"] = t1_avg
        f["t1_rep1"] = t1_rep1
        f.create_dataset("head_mask", data=mask, compression="gzip")
        f["shift_full"] = shift_full
        f["shift_acs"] = shift_acs
        f["shift_per_slice"] = shift_per_slice
        f.attrs.update(scan_id=scan_id, split=split, n_acs=n_acs, t1_reps=len(t1_rss),
                       code_version=__version__,
                       espirit=str(sorted(espirit_kwargs.items())))
    shutil.move(tmp, out_path)

    return dict(scan_id=scan_id, split=split, status="done", n_t2_reps=n_rep,
                n_slices=n_slice, n_pe_acquired=int(pe_acq.sum()),
                shift_full_fe=float(shift_full[0]), shift_full_pe=float(shift_full[1]),
                shift_acs_fe=float(shift_acs[:, 0].mean()),
                shift_acs_pe=float(shift_acs[:, 1].mean()),
                size_mb=out_path.stat().st_size / 1e6, seconds=time.time() - t0)


def read_subject(path, rep: int = 0, slices=None, aligned: str | None = "full") -> dict:
    """Load one repetition of a processed subject as full-size arrays.

    aligned : 'full' (training shift), 'acs' (test-time shift) or None (raw T1).
    Returns kspace [S, C, 256, 256], maps [S, C, 256, 256], t2_rss [R, S, ...],
    t1 [S, ...], head_mask, pe_acquired and the shift that was applied.
    """
    sl = slice(None) if slices is None else slices
    with h5py.File(path, "r") as f:
        pe_acq = f["pe_acquired"][()]
        k_small = f["t2_kspace"][rep, sl]
        kspace = np.zeros(k_small.shape[:-1] + (len(pe_acq),), dtype=np.complex64)
        kspace[..., pe_acq] = k_small
        shift = {"full": f["shift_full"][()], "acs": f["shift_acs"][rep],
                 None: np.zeros(2, np.float32)}[aligned]
        return dict(kspace=kspace, maps=from_f16(f["t2_maps_f16"][rep, sl]),
                    t2_rss=f["t2_rss"][:, sl], t1=apply_shift(f["t1_avg"][sl], shift),
                    t1_rep1=apply_shift(f["t1_rep1"][sl], shift),
                    head_mask=f["head_mask"][sl], pe_acquired=pe_acq, shift=shift,
                    attrs=dict(f.attrs))
