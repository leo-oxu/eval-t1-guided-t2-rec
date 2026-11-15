"""1D Cartesian undersampling masks along phase encoding.

M4Raw stores 256 PE lines but only 195 are acquired, so masks are defined on
the acquired lines only: a line that was never acquired is never "sampled".

R is the true net acceleration, as in fastMRI: a mask at acceleration R
samples round(acquired / R) lines in total, ACS included. With 195 acquired
lines that is 49 / 32 / 24 / 20 lines at R = 4 / 6 / 8 / 10.
"""

from __future__ import annotations

import numpy as np

N_ACS = 12
R_VALUES = (4, 6, 8, 10)


def n_lines(pe_acquired: np.ndarray, R: float) -> int:
    """Total sampled lines (ACS included) at net acceleration R."""
    return int(round(float(pe_acquired.sum()) / R))


def acs_mask(n_pe: int, n_acs: int = N_ACS, dc: int | None = None) -> np.ndarray:
    """Fully sampled centre: ``n_acs`` lines centred on the DC line."""
    dc = n_pe // 2 if dc is None else dc
    m = np.zeros(n_pe, dtype=bool)
    m[dc - n_acs // 2: dc + (n_acs + 1) // 2] = True
    return m


def _outer_budget(pe_acquired, R, n_acs, dc):
    """ACS mask, the acquired lines outside it, and how many of them to sample."""
    acs = acs_mask(len(pe_acquired), n_acs, dc) & pe_acquired
    free = np.flatnonzero(pe_acquired & ~acs)
    n_outer = n_lines(pe_acquired, R) - int(acs.sum())
    if not 0 < n_outer <= len(free):
        raise ValueError(f"R={R} with {n_acs} ACS lines leaves {n_outer} outer lines")
    return acs, free, n_outer


def equispaced_mask(pe_acquired: np.ndarray, R: float, n_acs: int = N_ACS,
                    dc: int | None = None) -> np.ndarray:
    """ACS plus evenly spaced outer lines, round(acquired / R) lines in total.

    The outer lines are spread evenly over the acquired lines outside the
    ACS, so the spacing is uniform on each side of the centre.
    """
    m, free, n_outer = _outer_budget(pe_acquired, R, n_acs, dc)
    m[free[((np.arange(n_outer) + 0.5) * len(free) / n_outer).astype(int)]] = True
    return m


def random_mask(pe_acquired: np.ndarray, R: float, n_acs: int = N_ACS,
                seed: int = 0, dc: int | None = None) -> np.ndarray:
    """ACS plus uniformly random outer lines; same line count as equispaced."""
    m, free, n_outer = _outer_budget(pe_acquired, R, n_acs, dc)
    m[np.random.default_rng(seed).choice(free, size=n_outer, replace=False)] = True
    return m


def net_acceleration(mask: np.ndarray, pe_acquired: np.ndarray) -> float:
    return float(pe_acquired.sum() / mask.sum())


def make_all_masks(pe_acquired: np.ndarray, r_values=R_VALUES, n_acs: int = N_ACS,
                   random_seeds=(0,), dc: int | None = None) -> dict[str, np.ndarray]:
    """All masks used in the project, keyed like ``equi_R4`` / ``rand_R4_s0``."""
    out = {"acquired": pe_acquired.copy(), "acs": acs_mask(len(pe_acquired), n_acs, dc)}
    for R in r_values:
        out[f"equi_R{R}"] = equispaced_mask(pe_acquired, R, n_acs, dc)
        for s in random_seeds:
            out[f"rand_R{R}_s{s}"] = random_mask(pe_acquired, R, n_acs, s, dc)
    return out
