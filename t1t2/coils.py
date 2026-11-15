"""Coil sensitivity maps with ESPIRiT (SigPy).

To avoid test-time leakage the maps are estimated from the ACS region of the
single undersampled repetition that is the network input, never from the
fully sampled data. The ACS is the same for every acceleration factor, so
one set of maps per repetition serves all R.
"""

from __future__ import annotations

import numpy as np

from .masks import N_ACS


def espirit_maps(kspace: np.ndarray, n_acs: int = N_ACS, kernel_width: int = 4,
                 thresh: float = 0.02, crop: float = 0.0, device: int = -1) -> np.ndarray:
    """ESPIRiT maps for one slice.

    kspace : [coil, FE, PE] complex. Only its central ``n_acs`` x ``n_acs``
             block is used, so passing fully sampled k-space leaks nothing.
    crop   : 0 keeps maps everywhere (sum_c |S_c|^2 = 1 on the whole image),
             which keeps |x| comparable to an RSS reference in every pixel.
    kernel_width : 4 was closest to a 32-line reference calibration when only
             12 ACS lines are available (checked on real validation subjects).
    device : -1 for CPU, 0 for the first GPU (needs CuPy; much faster).
    """
    import sigpy as sp
    import sigpy.mri as mr

    dev = sp.Device(device)
    maps = mr.app.EspiritCalib(
        sp.to_device(np.ascontiguousarray(kspace), dev), calib_width=n_acs,
        thresh=thresh, kernel_width=kernel_width, crop=crop, device=dev,
        show_pbar=False).run()
    return sp.to_device(maps, sp.cpu_device).astype(np.complex64)


def to_f16(maps: np.ndarray) -> np.ndarray:
    """complex [...] -> float16 [..., 2]. Maps are smooth with |S| <= 1, so
    half precision is enough and halves the storage on Drive."""
    return np.stack([maps.real, maps.imag], axis=-1).astype(np.float16)


def from_f16(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)


def coil_combine(coil_images: np.ndarray, maps: np.ndarray, coil_axis: int = -3) -> np.ndarray:
    """Sensitivity-weighted combination: sum_c conj(S_c) * x_c."""
    return (np.conj(maps) * coil_images).sum(axis=coil_axis)
