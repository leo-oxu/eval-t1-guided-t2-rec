"""T1 -> T2 alignment by in-plane translation.

M4Raw has a systematic T1-T2 shift of about 1.5 px along frequency encoding
(B0 drift between the first and last scan), so a per-subject sub-pixel
translation is the model used here. Rotation is not corrected; it is tested
as a perturbation in the robustness experiments.

T1 and T2 have different (partly inverted) contrast, so shifts are estimated
on edge maps, which both contrasts share.

All shifts are ``(d_FE, d_PE)`` in pixels: what to apply to the moving image
(T1) so that it lines up with the reference (T2).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from .fft import fft2c, ifft2c


def edge_map(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    e = ndimage.gaussian_gradient_magnitude(img.astype(np.float32), sigma)
    return e / (e.max() + 1e-12)


def estimate_shift(ref: np.ndarray, mov: np.ndarray, upsample: int = 20) -> np.ndarray:
    """Shift to apply to ``mov`` to register it to ``ref`` (2D images)."""
    shift, _, _ = phase_cross_correlation(edge_map(ref), edge_map(mov),
                                          upsample_factor=upsample, normalization=None)
    return np.asarray(shift, dtype=np.float32)


def volume_shift(ref_vol: np.ndarray, mov_vol: np.ndarray, upsample: int = 20):
    """Per-slice shifts and their median (robust to a few bad slices).

    Returns (median_shift [2], per_slice_shifts [n_slices, 2]).
    """
    per_slice = np.stack([estimate_shift(r, m, upsample) for r, m in zip(ref_vol, mov_vol)])
    return np.median(per_slice, axis=0).astype(np.float32), per_slice


def apply_shift(img: np.ndarray, shift) -> np.ndarray:
    """Sub-pixel translation of a real image (last two axes) via the Fourier
    shift theorem. No interpolation blur; wraps around at the borders, which
    is harmless for the air background around the head."""
    shift = np.asarray(shift, dtype=np.float64)
    n0, n1 = img.shape[-2:]
    f0 = np.fft.fftfreq(n0)[:, None]
    f1 = np.fft.fftfreq(n1)[None, :]
    ramp = np.exp(-2j * np.pi * (f0 * shift[0] + f1 * shift[1]))
    out = np.fft.ifftn(np.fft.fftn(img, axes=(-2, -1)) * ramp, axes=(-2, -1))
    return out.real.astype(np.float32)


def lowres_pe(img: np.ndarray, pe_keep: np.ndarray) -> np.ndarray:
    """Blur a real image exactly like keeping only the PE lines in ``pe_keep``
    (e.g. the ACS). Used to bring T1 to the resolution of the T2 ACS image
    before estimating the test-time shift."""
    return np.abs(ifft2c(fft2c(img.astype(np.complex64)) * pe_keep)).astype(np.float32)
