"""Centred, orthonormal 2D FFTs (NumPy).

Same convention as fastMRI's ``fft2c`` / ``ifft2c`` and as SigPy, which is
what the M4Raw ``reconstruction_rss`` images were made with. Notebook 01
checks this against the stored images.
"""

from __future__ import annotations

import numpy as np

AXES = (-2, -1)


def fft2c(x: np.ndarray) -> np.ndarray:
    """Image -> k-space over the last two axes."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=AXES), axes=AXES, norm="ortho"), axes=AXES)


def ifft2c(k: np.ndarray) -> np.ndarray:
    """k-space -> image over the last two axes."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k, axes=AXES), axes=AXES, norm="ortho"), axes=AXES)


def rss(x: np.ndarray, axis: int) -> np.ndarray:
    """Root-sum-of-squares over the coil axis."""
    return np.sqrt((np.abs(x) ** 2).sum(axis=axis))


def kspace_to_rss(kspace: np.ndarray, coil_axis: int = -3) -> np.ndarray:
    """[..., coil, FE, PE] k-space -> [..., FE, PE] magnitude RSS image."""
    return rss(ifft2c(kspace), axis=coil_axis).astype(np.float32)
