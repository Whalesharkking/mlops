"""Model registry: every trained model is registered, only the gate moves the champion alias.

``WandbRegistry`` is the real registry (W&B Registry, collection REGISTRY_COLLECTION).
``LocalRegistry`` keeps the same contract in a folder, for tests and runs without a W&B
account. A model is a folder with ``model.txt`` (LightGBM booster) and ``model.json``
(metadata: feature columns, targets, training period, data revision, Git commit, scores).
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from nectarflow.common.config import CHAMPION_ALIAS, REGISTRY_COLLECTION

MODEL_FILE = "model.txt"
META_FILE = "model.json"


@dataclass(frozen=True)
class RegisteredModel:
    version: str
    path: Path
    meta: dict[str, Any]


def read_model_dir(path: Path, version: str) -> RegisteredModel:
    return RegisteredModel(version, path, json.loads((path / META_FILE).read_text()))


class Registry(Protocol):
    def champion(self) -> RegisteredModel | None: ...

    def register(self, model_dir: Path, promote: bool) -> str: ...


class LocalRegistry:
    """``<root>/v<N>/`` per model and ``<root>/champion`` holding the champion's version."""

    def __init__(self, root: Path):
        self.root = root

    def champion(self) -> RegisteredModel | None:
        pointer = self.root / CHAMPION_ALIAS
        if not pointer.exists():
            return None
        version = pointer.read_text().strip()
        return read_model_dir(self.root / version, version)

    def register(self, model_dir: Path, promote: bool) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        version = f"v{sum(1 for p in self.root.glob('v*') if p.is_dir())}"
        shutil.copytree(model_dir, self.root / version)
        if promote:
            (self.root / CHAMPION_ALIAS).write_text(version)
        return version


class WandbRegistry:
    """Models are logged as artifacts of the run and linked into the W&B Registry."""

    def __init__(self, run, cache_dir: Path, registry: str = "model"):
        self.run = run
        self.cache_dir = cache_dir
        self.target = f"wandb-registry-{registry}/{REGISTRY_COLLECTION}"

    def champion(self) -> RegisteredModel | None:
        import wandb

        try:
            artifact = wandb.Api().artifact(f"{self.target}:{CHAMPION_ALIAS}")
        except wandb.errors.CommError as err:
            # wandb turns every failure into CommError. Only "not found" means there is no
            # champion yet, a network or auth error must not look like an empty registry.
            if "not found" in str(err):
                return None
            raise
        path = Path(artifact.download(root=str(self.cache_dir / "champion")))
        return read_model_dir(path, artifact.version)

    def register(self, model_dir: Path, promote: bool) -> str:
        import wandb

        meta = json.loads((model_dir / META_FILE).read_text())
        artifact = wandb.Artifact("nectarflow-lgbm", type="model", metadata=meta)
        artifact.add_dir(str(model_dir))
        self.run.log_artifact(artifact)
        artifact.wait()
        # Every model is registered. W&B keeps an alias on one version per collection,
        # so linking with "champion" moves it away from the old champion.
        linked = self.run.link_artifact(artifact, self.target, aliases=[CHAMPION_ALIAS] if promote else [])
        return getattr(linked, "version", None) or artifact.version
