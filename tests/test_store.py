"""Feature store against a fake Hugging Face API (no network, no token)."""

from types import SimpleNamespace

import pandas as pd

from nectarflow.features import store as fs


class FakeHfApi:
    def __init__(self, token=None):
        self.remote_files = {"features/issue_date=2024-03-14.parquet"}
        self.commits = []
        self.created = []

    def list_repo_files(self, repo_id, repo_type):
        return sorted(self.remote_files)

    def create_repo(self, repo_id, repo_type, exist_ok):
        self.created.append((repo_id, repo_type, exist_ok))

    def upload_folder(self, repo_id, repo_type, folder_path, allow_patterns, commit_message):
        self.commits.append((commit_message, list(allow_patterns)))
        self.remote_files |= set(allow_patterns)
        return SimpleNamespace(oid=f"{len(self.commits):040d}")


def make_store(tmp_path, monkeypatch) -> fs.FeatureStore:
    monkeypatch.setattr(fs, "HfApi", FakeHfApi)
    return fs.FeatureStore(tmp_path, repo_id="user/nectarflow-features", token="hf_test")


def test_exists_sees_files_that_are_only_on_the_hub(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    assert store.exists("features/issue_date=2024-03-14.parquet")
    assert not store.exists("features/issue_date=2024-03-15.parquet")


def test_push_uploads_exactly_the_written_files_in_one_commit(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    for i in range(5):
        store.write(f"features/issue_date=2024-04-0{i + 1}.parquet", pd.DataFrame({"h": [1]}))

    assert store.push("backfill") is not None
    assert len(store.api.commits) == 1
    message, uploaded = store.api.commits[0]
    assert message == "backfill"
    assert len(uploaded) == 6 and fs.CARD_PATH in uploaded  # 5 files + dataset card, once
    assert store.push("nothing left") is None  # pending list is cleared

    store.write(fs.LABELS_PATH, pd.DataFrame({"label": [0.1]}))
    store.push("second")
    assert store.api.created == [("user/nectarflow-features", "dataset", True)]  # repo created once
    assert store.api.commits[1][1] == [fs.LABELS_PATH]


def test_local_mode_never_touches_the_hub(tmp_path):
    store = fs.FeatureStore(tmp_path)
    store.write(fs.LABELS_PATH, pd.DataFrame({"label": [0.1]}))
    assert store.push("local") is None
    assert (tmp_path / fs.LABELS_PATH).exists()
