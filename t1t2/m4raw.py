"""M4Raw download, file discovery and loading.

Facts this module relies on (M4Raw v1.6, Zenodo record 8056074):

* files are named ``<scan_id>_<contrast><rep>.h5``, e.g. ``2022061003_T201.h5``
* ``kspace`` is ``[slice, coil, FE, PE]`` complex64, typically (18, 4, 256, 256)
* ``reconstruction_rss`` is ``[slice, FE, PE]`` float32
* only ~195 of the 256 phase-encode lines are acquired; the rest are zero
* train / val have 3 repetitions of T1 and T2 per subject, test has 6
* the motion-corrupted subjects are in a separate zip that we never download
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .paths import Paths

ZENODO_URL = "https://zenodo.org/api/records/8056074/files/{name}/content"

# split -> (zip name on Zenodo, md5, size in GB)
ZIPS = {
    "train": ("M4RawV1.5_multicoil_train.zip", "b26afbb2acca8c2dea91b94735b9a94a", 12.7),
    "val": ("M4RawV1.5_multicoil_val.zip", "4768c897750b903c98b95dcd79284f7b", 3.0),
    "test": ("M4Raw_multicoil_test.zip", "fb2ff6b684493f122cad4a5f84cba82e", 5.0),
}

CONTRASTS = ("T1", "T2")
FNAME_RE = re.compile(r"(?P<scan_id>\d{10})_(?P<contrast>T1|T2)(?P<rep>\d{2})\.h5$")


# --------------------------------------------------------------------------
# download + extract
# --------------------------------------------------------------------------
def _md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch_split(split: str, paths: Paths, keep_zip_on_drive: bool = True,
                verify_md5: bool = True) -> Path:
    """Make ``paths.raw/<split>/`` contain the T1 and T2 h5 files of one split.

    Order of preference: already extracted on this VM -> zip cached on Drive ->
    download from Zenodo. Only T1/T2 files are extracted (FLAIR is skipped).
    Safe to call again: finished work is skipped.
    """
    name, md5, size_gb = ZIPS[split]
    out = paths.raw / split
    marker = out / ".extracted"
    if marker.exists():
        print(f"[{split}] already extracted: {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)

    local_zip = paths.downloads / name
    drive_zip = paths.zips / name
    if drive_zip.exists():
        print(f"[{split}] copying zip from Drive ({size_gb:.1f} GB)...")
        shutil.copy(drive_zip, local_zip)
    else:
        print(f"[{split}] downloading from Zenodo ({size_gb:.1f} GB)...")
        subprocess.run(["wget", "-c", "-nv", "-O", str(local_zip),
                        ZENODO_URL.format(name=name)], check=True)
        if verify_md5:
            got = _md5(local_zip)
            if got != md5:
                local_zip.unlink()
                raise IOError(f"{name}: md5 mismatch ({got} != {md5}); "
                              "the partial file was deleted, run again")
        if keep_zip_on_drive:
            print(f"[{split}] saving zip to Drive for later sessions...")
            shutil.copy(local_zip, drive_zip)

    n = 0
    with zipfile.ZipFile(local_zip) as zf:
        for member in zf.namelist():
            if not FNAME_RE.search(member):
                continue
            with zf.open(member) as src, open(out / Path(member).name, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 24)
            n += 1
    if n == 0:
        raise IOError(f"{name}: no files matching <scan_id>_T1xx.h5 / _T2xx.h5 found")
    local_zip.unlink()
    marker.touch()
    print(f"[{split}] extracted {n} files to {out}")
    return out


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def index_files(raw_dir: Path) -> pd.DataFrame:
    """One row per h5 file under ``raw_dir/<split>/``."""
    rows = []
    for p in sorted(Path(raw_dir).rglob("*.h5")):
        m = FNAME_RE.search(p.name)
        if m:
            rows.append(dict(split=p.parent.name, scan_id=m["scan_id"],
                             contrast=m["contrast"], rep=int(m["rep"]), path=str(p)))
    return pd.DataFrame(rows, columns=["split", "scan_id", "contrast", "rep", "path"])


def subject_files(index: pd.DataFrame, scan_id: str) -> dict[str, list[str]]:
    """{'T1': [paths sorted by rep], 'T2': [...]} for one subject."""
    sub = index[index.scan_id == scan_id].sort_values("rep")
    return {c: sub[sub.contrast == c].path.tolist() for c in CONTRASTS}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_kspace(path, slices=None) -> np.ndarray:
    """[slice, coil, FE, PE] complex64. ``slices`` may be an int, slice or list."""
    with h5py.File(path, "r") as f:
        k = f["kspace"][()] if slices is None else f["kspace"][slices]
    return np.asarray(k, dtype=np.complex64)


def load_rss(path) -> np.ndarray:
    """The stored reconstruction, [slice, FE, PE] float32."""
    with h5py.File(path, "r") as f:
        return np.asarray(f["reconstruction_rss"][()], dtype=np.float32)


def load_attrs(path) -> dict:
    with h5py.File(path, "r") as f:
        out = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        out["kspace_shape"] = tuple(f["kspace"].shape)
    return out


def acquired_pe(kspace: np.ndarray) -> np.ndarray:
    """Boolean vector over the PE axis (last): True where the line holds data."""
    other = tuple(range(kspace.ndim - 1))
    return np.abs(kspace).sum(axis=other) > 0


def pe_energy(kspace: np.ndarray) -> np.ndarray:
    """Mean |k| per phase-encode line; its peak is the k-space centre line."""
    other = tuple(range(kspace.ndim - 1))
    return np.abs(kspace).mean(axis=other)
