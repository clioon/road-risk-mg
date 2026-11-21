import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import psutil
from filelock import FileLock
from pydantic import BaseModel, ConfigDict

from road_accidents.config import PipelineError

if TYPE_CHECKING:
    from mlflow.entities import Run


class MemorySampler:
    def __init__(self) -> None:
        self.finished = threading.Event()
        self.peak_bytes = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        process = psutil.Process()
        while not self.finished.is_set():
            total = 0
            for child in [process, *process.children(recursive=True)]:
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, total)
            self.finished.wait(0.05)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.finished.set()
        self.thread.join()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exception: Any) -> None:
        self.stop()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any, *, exclude_keys: frozenset[str] = frozenset()) -> str:
    if exclude_keys:
        if not isinstance(value, dict):
            raise ValueError("Excluded fields require a mapping.")
        value = {key: item for key, item in value.items() if key not in exclude_keys}
    return "sha256:" + hash_bytes(canonical_bytes(value))


def append_event(path: Path, event: dict[str, Any]) -> None:
    payload = canonical_bytes(event) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"), path.open("ab") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def capture_code_state(root: Path) -> dict[str, Any]:
    root = root.resolve()
    archived_state = root / ".road-risk-code.json"
    if archived_state.is_file():
        state: dict[str, Any] = json.loads(archived_state.read_bytes())
        if canonical_hash(state["inventory"]) != state["code_hash"]:
            raise PipelineError("Archived code inventory is invalid.", 5, "code_changed")
        for entry in state["inventory"]:
            path = root / entry["path"]
            if (
                not path.resolve().is_relative_to(root)
                or path.is_symlink()
                or not path.is_file()
                or hash_file(path) != entry["sha256"]
            ):
                raise PipelineError("Archived code content is invalid.", 5, "code_changed")
        return state
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if listing.returncode:
        raise PipelineError(
            "Code provenance requires a local Git repository.", 6, "code_provenance_missing"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    inventory = []
    for name in sorted(set(listing.stdout.decode("utf-8").split("\0")) - {""}):
        path = root / name
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise PipelineError("Code snapshot contains an unsafe path.")
        inventory.append({"path": name, "size": path.stat().st_size, "sha256": hash_file(path)})
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout),
        "code_hash": canonical_hash(inventory),
        "inventory": inventory,
    }


def runtime_environment() -> dict[str, Any]:
    return {
        "python": list(sys.version_info[:3]),
        "packages": dict(
            sorted(
                (distribution.metadata["Name"].lower().replace("_", "-"), distribution.version)
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            )
        ),
    }


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    owner_type: Literal["training", "bundle", "evaluation", "report"]
    owner_id: str
    run_id: str | None = None
    artifact_type: Literal["model", "predictions", "metrics", "table", "figure", "manifest", "log"]
    uri: str
    sha256: str
    created_at: datetime


def record_artifact(
    path: Path,
    root: Path,
    *,
    owner_type: Literal["training", "bundle", "evaluation", "report"],
    owner_id: str,
    artifact_type: Literal["model", "predictions", "metrics", "table", "figure", "manifest", "log"],
    run_id: str | None = None,
) -> ArtifactRecord:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root.resolve()):
        raise PipelineError("Artifact path must remain inside the work directory.")
    digest = hash_file(resolved)
    return ArtifactRecord(
        artifact_id="sha256:" + digest,
        owner_type=owner_type,
        owner_id=owner_id,
        run_id=run_id,
        artifact_type=artifact_type,
        uri=resolved.relative_to(root.resolve()).as_posix(),
        sha256=digest,
        created_at=datetime.now(UTC),
    )


class RunTracker:
    def __init__(self, root: Path) -> None:
        from mlflow import MlflowClient

        self.root = root.resolve()
        artifacts = self.root / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        locks = self.root / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(locks / "mlflow.lock"))
        with self.lock:
            self.client = MlflowClient(tracking_uri=f"sqlite:///{artifacts.as_posix()}/mlflow.db")
            experiment = self.client.get_experiment_by_name("road-risk")
            self.experiment_id = (
                experiment.experiment_id
                if experiment
                else self.client.create_experiment(
                    "road-risk", artifact_location=(artifacts / "mlflow").as_uri()
                )
            )

    @contextmanager
    def run(
        self, parameters: dict[str, str], *, tags: dict[str, str] | None = None
    ) -> Iterator[str]:
        with self.lock:
            run = self.client.create_run(self.experiment_id, tags=tags)
            identifier = run.info.run_id
        try:
            with self.lock:
                for name, value in parameters.items():
                    self.client.log_param(identifier, name, value)
            yield identifier
        except BaseException:
            with self.lock:
                self.client.set_terminated(identifier, status="FAILED")
            raise
        else:
            with self.lock:
                self.client.set_terminated(identifier, status="FINISHED")

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        if not all(math.isfinite(value) for value in metrics.values()):
            raise PipelineError("Only finite metrics may be logged.", 6, "invalid_metric")
        with self.lock:
            if self.client.get_run(run_id).info.status != "RUNNING":
                raise PipelineError("A completed run cannot be updated.", 6, "closed_run")
            for name, value in metrics.items():
                self.client.log_metric(run_id, name, value)

    def log_artifact(self, run_id: str, path: Path) -> None:
        if not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise PipelineError("Tracked artifact must be contained in the work directory.")
        with self.lock:
            if self.client.get_run(run_id).info.status != "RUNNING":
                raise PipelineError("A completed run cannot be updated.", 6, "closed_run")
            self.client.log_artifact(run_id, str(path.resolve()))

    def get_run(self, run_id: str) -> "Run":
        with self.lock:
            return self.client.get_run(run_id)
