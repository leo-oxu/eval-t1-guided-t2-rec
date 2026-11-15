"""Where everything lives.

Two roots:

* ``drive``  - persistent (Google Drive). Everything a later notebook needs.
* ``local``  - the Colab VM disk. Fast but wiped when the session ends.
               Holds the unzipped raw M4Raw files and scratch copies.

With ``smoke=True`` every output goes under ``<drive>/smoke/`` instead, so a
quick trial run can never overwrite real results. The zip cache is shared.

Both can be overridden with the environment variables ``T1T2_DRIVE_ROOT`` and
``T1T2_LOCAL_ROOT`` (used for local testing outside Colab).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/t1t2"
DEFAULT_LOCAL_ROOT = "/content/t1t2_local"


@dataclass(frozen=True)
class Paths:
    drive: Path
    local: Path
    smoke: bool = False

    @property
    def out(self) -> Path:
        return self.drive / "smoke" if self.smoke else self.drive

    # --- ephemeral -------------------------------------------------------
    @property
    def raw(self) -> Path:
        """Unzipped M4Raw h5 files: raw/<split>/<scan_id>_<T1|T2><rep>.h5"""
        return self.local / "raw"

    @property
    def downloads(self) -> Path:
        return self.local / "downloads"

    # --- persistent ------------------------------------------------------
    @property
    def zips(self) -> Path:
        """Optional copy of the Zenodo zips, so later sessions skip Zenodo."""
        return self.drive / "raw_zips"

    @property
    def meta(self) -> Path:
        """manifest.csv, splits.csv, alignment.csv, masks.npz, audit summary."""
        return self.out / "meta"

    @property
    def processed(self) -> Path:
        """processed/<split>/<scan_id>.h5 written by notebook 02."""
        return self.out / "processed"

    @property
    def runs(self) -> Path:
        """runs/<run_name>/ checkpoints and logs written by notebook 04."""
        return self.out / "runs"

    @property
    def metrics(self) -> Path:
        """Per-slice metric CSVs written by notebooks 03, 05, 06."""
        return self.out / "metrics"

    @property
    def figures(self) -> Path:
        return self.out / "figures"

    def make(self) -> "Paths":
        for p in (self.raw, self.downloads, self.zips, self.meta, self.processed,
                  self.runs, self.metrics, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(drive_root: str | os.PathLike | None = None,
              local_root: str | os.PathLike | None = None, smoke: bool = False) -> Paths:
    drive = drive_root or os.environ.get("T1T2_DRIVE_ROOT", DEFAULT_DRIVE_ROOT)
    local = local_root or os.environ.get("T1T2_LOCAL_ROOT", DEFAULT_LOCAL_ROOT)
    return Paths(Path(drive), Path(local), smoke).make()
