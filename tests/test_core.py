"""Checks of the data pipeline on synthetic M4Raw-like data.

Run from the repo root:  python -m pytest tests -q
"""

import numpy as np
import pytest

from t1t2 import m4raw, masks
from t1t2.align import apply_shift, estimate_shift
from t1t2.coils import coil_combine, espirit_maps, from_f16, to_f16
from t1t2.fft import fft2c, ifft2c
from t1t2.preprocess import process_subject, read_subject

from tests.synth import N_ACQ, TRUE_SHIFT, make_dataset


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    make_dataset(d, subjects={"train": ["2022010101"], "test": ["2022030101"]}, n_slices=2)
    return d


def test_fft_roundtrip_and_parseval():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 8, 10)) + 1j * rng.standard_normal((3, 8, 10))
    assert np.allclose(ifft2c(fft2c(x)), x)
    assert np.isclose(np.linalg.norm(fft2c(x)), np.linalg.norm(x))


def test_masks():
    pe = np.zeros(256, bool)
    pe[30:225] = True
    for R in masks.R_VALUES:
        eq = masks.equispaced_mask(pe, R)
        rd = masks.random_mask(pe, R, seed=0)
        acs = masks.acs_mask(256)
        assert acs.sum() == masks.N_ACS and eq[acs].all() and rd[acs].all()
        assert not eq[~pe].any() and not rd[~pe].any()
        assert eq.sum() == rd.sum() == round(195 / R)           # R is the net acceleration
        assert abs(masks.net_acceleration(eq, pe) - R) < 0.3
        gaps = np.diff(np.flatnonzero(eq & ~acs))               # outer lines evenly spaced
        assert np.sort(gaps)[-2] - gaps.min() <= 1              # (one larger gap spans the ACS)
        assert (rd == masks.random_mask(pe, R, seed=0)).all()   # reproducible


def test_shift_estimation_across_contrast():
    rng = np.random.default_rng(0)
    from tests.synth import _phantoms
    t2, t1 = _phantoms()
    t1_moved = apply_shift(t1, (-1.5, -0.3)) + 0.02 * rng.standard_normal(t1.shape)
    est = estimate_shift(t2 + 0.02 * rng.standard_normal(t2.shape), t1_moved)
    assert np.abs(est - np.array([1.5, 0.3])).max() < 0.25


def test_index_and_acquired_lines(raw):
    idx = m4raw.index_files(raw)
    assert set(idx.split) == {"train", "test"}
    assert (idx.groupby(["split", "contrast"]).size() == [6, 6, 3, 3]).all()
    k = m4raw.load_kspace(idx.path[0])
    assert k.shape == (2, 4, 256, 256) and m4raw.acquired_pe(k).sum() == N_ACQ
    assert int(np.argmax(m4raw.pe_energy(k))) == 128


def test_espirit_maps_explain_the_data(raw):
    idx = m4raw.index_files(raw)
    k = m4raw.load_kspace(idx[idx.contrast == "T2"].path.iloc[0])[0]
    maps = espirit_maps(k)
    assert np.allclose((np.abs(maps) ** 2).sum(0), 1, atol=1e-3)
    assert np.abs(from_f16(to_f16(maps)) - maps).max() < 1e-3
    # project the coil images onto the maps and back: little signal should be lost
    coil_imgs = ifft2c(k)
    resid = coil_imgs - maps * coil_combine(coil_imgs, maps)
    obj = np.abs(coil_imgs).sum(0) > 0.3 * np.abs(coil_imgs).sum(0).max()
    assert np.linalg.norm(resid[:, obj]) / np.linalg.norm(coil_imgs[:, obj]) < 0.15


@pytest.mark.parametrize("split,n_rep", [("train", 3), ("test", 6)])
def test_process_and_read_subject(raw, tmp_path, split, n_rep):
    idx = m4raw.index_files(raw)
    scan_id = idx[idx.split == split].scan_id.iloc[0]
    out = tmp_path / f"{scan_id}.h5"
    row = process_subject(m4raw.subject_files(idx, scan_id), out, scan_id, split)
    assert row["status"] == "done" and row["n_t2_reps"] == n_rep
    assert abs(row["shift_full_fe"] - TRUE_SHIFT[0]) < 0.25
    assert abs(row["shift_full_pe"] - TRUE_SHIFT[1]) < 0.25
    assert abs(row["shift_acs_fe"] - TRUE_SHIFT[0]) < 0.5      # ACS-only: FE still resolved
    assert process_subject({}, out, scan_id, split)["status"] == "skipped"

    d = read_subject(out, rep=1)
    raw_k = m4raw.load_kspace(idx[(idx.scan_id == scan_id) & (idx.contrast == "T2")
                                  & (idx.rep == 2)].path.iloc[0])
    assert np.array_equal(d["kspace"], raw_k)                  # lossless re-embedding
    assert d["maps"].shape == raw_k.shape and d["t2_rss"].shape[0] == n_rep
    assert 0.05 < d["head_mask"].mean() < 0.8
    assert d["attrs"]["t1_reps"] == 3
