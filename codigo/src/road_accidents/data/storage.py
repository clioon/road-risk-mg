import json
import os
import shutil
import ssl
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import geopandas as geopandas
import httpx
import pandas as pandas
from filelock import FileLock

from road_accidents import __version__
from road_accidents.config import PipelineError
from road_accidents.data.schemas import BUFFER_LIMIT, SourceSnapshot, public_url
from road_accidents.provenance import canonical_bytes, hash_bytes, hash_file


def read_limited(source: IO[bytes], limit: int = BUFFER_LIMIT) -> bytes:
    if limit <= 0:
        raise PipelineError("Buffer limit must be positive.", 2, "configuration_error")
    payload = bytearray()
    while chunk := source.read(min(65_536, limit - len(payload) + 1)):
        payload.extend(chunk)
        if len(payload) > limit:
            raise PipelineError("Source exceeds the memory buffer limit.", 4, "buffer_limit")
    return bytes(payload)


def download_bytes(
    url: str,
    *,
    max_buffer_bytes: int = BUFFER_LIMIT,
    client: httpx.Client | None = None,
    data: dict[str, str] | None = None,
) -> bytes:
    if client is None:
        with httpx.Client(
            timeout=60, verify=ssl.create_default_context(), follow_redirects=False
        ) as connection:
            return download_bytes(
                url, max_buffer_bytes=max_buffer_bytes, client=connection, data=data
            )
    try:
        public_url(url)
        if max_buffer_bytes <= 0:
            raise PipelineError("Buffer limit must be positive.", 2, "configuration_error")
        for _attempt in range(6):
            with client.stream(
                "POST" if data is not None else "GET",
                url,
                data=data,
                headers={
                    "Accept-Encoding": "identity",
                    "User-Agent": f"road-accidents/{__version__} (private academic research)",
                },
            ) as response:
                if response.is_redirect:
                    url = public_url(urljoin(url, response.headers["location"]))
                    if urlsplit(url).hostname == "accounts.google.com":
                        raise PipelineError(
                            "Source requires manual authentication.", 3, "source_access"
                        )
                    continue
                response.raise_for_status()
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise PipelineError("Unsupported source transport encoding.", 4, "schema_error")
                declared = response.headers.get("content-length")
                expected = int(declared) if declared is not None else None
                if expected is not None and expected > max_buffer_bytes:
                    raise PipelineError(
                        "Source exceeds the memory buffer limit.", 4, "buffer_limit"
                    )
                payload = bytearray()
                for chunk in response.iter_bytes(chunk_size=65_536):
                    payload.extend(chunk)
                    if len(payload) > max_buffer_bytes:
                        raise PipelineError(
                            "Source exceeds the memory buffer limit.", 4, "buffer_limit"
                        )
                if expected is not None and len(payload) != expected:
                    raise PipelineError("Source download was incomplete.", 3, "source_incomplete")
                return bytes(payload)
        raise PipelineError("Source redirect limit reached.", 3, "source_access")
    except httpx.HTTPStatusError as error:
        raise PipelineError(
            f"Source server returned HTTP {error.response.status_code}.", 3, "source_access"
        ) from None
    except httpx.TimeoutException:
        raise PipelineError("Source request timed out.", 3, "source_access") from None
    except (httpx.HTTPError, OSError, ValueError, KeyError):
        raise PipelineError("Source acquisition failed.", 3, "source_access") from None


def publish_snapshot(root: Path, snapshot: SourceSnapshot, payload: bytes) -> SourceSnapshot:
    if snapshot.sha256 != hash_bytes(payload) or snapshot.privacy_status != "passed":
        raise PipelineError("Source approval and content identity are required.")
    raw = safe_path(root, snapshot.raw_path)
    if raw.parent.exists():
        if not raw.is_file() or hash_file(raw) != snapshot.sha256:
            raise PipelineError("Existing source content failed integrity verification.")
    else:
        try:
            with atomic_directory(root, raw.parent.relative_to(root.resolve())) as staging:
                with (staging / raw.name).open("xb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
        except FileExistsError:
            if not raw.is_file() or hash_file(raw) != snapshot.sha256:
                raise PipelineError(
                    "Concurrent source publication failed integrity verification."
                ) from None
    manifest = safe_path(root, snapshot.manifest_path)
    with FileLock(str(root.resolve() / ".locks" / "source-manifests.lock")):
        if manifest.exists():
            previous = SourceSnapshot.model_validate(read_json(manifest))
            if (
                previous.source_manifest_id != snapshot.source_manifest_id
                or previous.snapshot_id != snapshot.snapshot_id
            ):
                raise PipelineError("Existing source manifest failed integrity verification.")
            return previous
        write_json(manifest, snapshot.model_dump(mode="json"))
    return snapshot


def safe_path(root: Path, relative: str | Path) -> Path:
    text = str(relative).replace("\\", "/")
    portable = PurePosixPath(text)
    if portable.is_absolute() or PureWindowsPath(text).drive or ".." in portable.parts:
        raise PipelineError("Path must be relative and contained in the work directory.")
    resolved_root = root.resolve()
    destination = (resolved_root / portable).resolve()
    if destination == resolved_root or not destination.is_relative_to(resolved_root):
        raise PipelineError("Path must identify a child of the work directory.")
    return destination


@contextmanager
def atomic_directory(root: Path, relative: str | Path) -> Iterator[Path]:
    """Publish approved or derived content; never buffer uninspected sources here."""
    destination = safe_path(root, relative)
    locks = root.resolve() / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    identifier = hash_bytes(str(destination.relative_to(root.resolve())).encode())
    with FileLock(str(locks / f"{identifier}.lock")):
        if destination.exists():
            raise FileExistsError("Immutable output already exists.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination.parent))
        try:
            yield staging
            if any(path.is_symlink() for path in staging.rglob("*")):
                raise PipelineError("Published content cannot contain symbolic links.")
            os.rename(staging, destination)
        except BaseException:
            failures = root.resolve() / ".failures"
            failures.mkdir(parents=True, exist_ok=True)
            write_json(
                failures / f"{uuid4().hex}.json",
                {
                    "status": "failed",
                    "code": "publication_failed",
                    "target_hash": identifier,
                },
            )
            shutil.rmtree(staging, ignore_errors=True)
            raise


def write_json(path: Path, content: Any) -> None:
    payload = canonical_bytes(content) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def read_json(path: Path) -> Any:
    with path.open("rb") as source:
        payload = source.read(16_777_217)
    if len(payload) > 16_777_216:
        raise PipelineError("Manifest exceeds the supported size limit.")
    try:
        return json.loads(payload)
    except (ValueError, UnicodeError):
        raise PipelineError("Invalid JSON manifest.") from None


def write_parquet(path: Path, frame: pandas.DataFrame) -> None:
    if path.exists():
        raise FileExistsError("Immutable table already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_parquet(path: Path) -> pandas.DataFrame:
    return pandas.read_parquet(path)


def write_geoparquet(path: Path, frame: geopandas.GeoDataFrame) -> None:
    if frame.crs is None:
        raise PipelineError("Geometry requires an explicit CRS.")
    write_parquet(path, frame)


def read_geoparquet(path: Path) -> geopandas.GeoDataFrame:
    return geopandas.read_parquet(path)
