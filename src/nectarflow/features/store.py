"""Feature store: versioned Parquet files in a Hugging Face dataset repo.

Every upload is a commit on the Hub, so each write creates a new dataset revision. That
revision is the data version a model is trained on.

Layout (relative to the data directory and the repo root):
    labels/labels.parquet                       daily labels, rewritten every run
    climatology/climatology.parquet              per day of year, 2019-2023
    features/issue_date=YYYY-MM-DD.parquet       ten rows per issue date, written once
    raw/openmeteo/run=YYYY-MM-DD.parquet         archived hourly run, per location
    raw/hivewatch/poll_date=YYYY-MM-DD.parquet   last 30 days of the weight series as polled
    quality/n_scales.parquet                     scale count per poll date (from 15.10.2026)
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

log = logging.getLogger(__name__)


def features_path(issue_date: date) -> str:
    return f"features/issue_date={issue_date.isoformat()}.parquet"


def run_path(issue_date: date) -> str:
    return f"raw/openmeteo/run={issue_date.isoformat()}.parquet"


def poll_path(poll_date: date) -> str:
    return f"raw/hivewatch/poll_date={poll_date.isoformat()}.parquet"


LABELS_PATH = "labels/labels.parquet"
CLIMATOLOGY_PATH = "climatology/climatology.parquet"
N_SCALES_PATH = "quality/n_scales.parquet"


class FeatureStore:
    """Writes locally first; ``push`` uploads the written files in one Hub commit."""

    def __init__(self, data_dir: Path, repo_id: str | None = None, token: str | None = None):
        self.data_dir = data_dir
        self.repo_id = repo_id
        self.api = HfApi(token=token) if repo_id else None
        self.token = token
        self._pending: list[str] = []
        self._remote_files: set[str] | None = None

    @property
    def remote(self) -> bool:
        return self.api is not None

    def write(self, rel_path: str, df: pd.DataFrame) -> None:
        path = self.data_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self._pending.append(rel_path)

    def exists(self, rel_path: str) -> bool:
        if (self.data_dir / rel_path).exists():
            return True
        if not self.remote:
            return False
        if self._remote_files is None:
            try:
                self._remote_files = set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
            except RepositoryNotFoundError:
                self._remote_files = set()
        return rel_path in self._remote_files

    def read(self, rel_path: str) -> pd.DataFrame | None:
        """Read a file, from the Hub if it is not available locally."""
        path = self.data_dir / rel_path
        if not path.exists() and self.remote:
            try:
                hf_hub_download(
                    self.repo_id, rel_path, repo_type="dataset", token=self.token, local_dir=self.data_dir
                )
            except EntryNotFoundError, RepositoryNotFoundError:
                return None
        return pd.read_parquet(path) if path.exists() else None

    def push(self, message: str) -> str | None:
        """Upload all pending files in one commit. Returns the commit id (None if nothing pushed)."""
        files = sorted(set(self._pending))
        self._pending.clear()
        if not files:
            return None
        if not self.remote:
            log.info("local mode: %d files written to %s, nothing uploaded", len(files), self.data_dir)
            return None
        self.api.create_repo(self.repo_id, repo_type="dataset", exist_ok=True)
        info = self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type="dataset",
            folder_path=self.data_dir,
            allow_patterns=files,
            commit_message=message,
        )
        log.info("pushed %d files to %s@%s", len(files), self.repo_id, info.oid[:8])
        return info.oid
