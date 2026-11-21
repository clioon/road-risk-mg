import io
import re
import ssl
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from road_accidents.config import PipelineError
from road_accidents.data.osm import (
    OVERPASS_URL,
    extract_mg_boundary,
    historical_tiles,
    ingest_osm,
    inspect_osm,
    osm_boundary_query,
    osm_tile_query,
)
from road_accidents.data.schemas import (
    BUFFER_LIMIT,
    OSM_EFFECTIVE_AT,
    PRIVACY_POLICY,
    SourceSnapshot,
    public_url,
)
from road_accidents.data.storage import (
    atomic_directory,
    download_bytes,
    read_json,
    read_limited,
    safe_path,
    write_json,
)
from road_accidents.provenance import canonical_bytes, canonical_hash, hash_bytes


def respect_service_capacity(client: httpx.Client, endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.hostname != "overpass-api.de":
        return
    status_url = urlunsplit((parsed.scheme, parsed.netloc, "/api/status", "", ""))
    for attempt in range(2):
        status = download_bytes(status_url, client=client, max_buffer_bytes=65536).decode("utf-8")
        available = re.search(r"(?m)^(\d+) slots available now\.", status)
        if available and int(available.group(1)) > 0:
            return
        delays = [
            int(value)
            for value in re.findall(r"Slot available after:[^\n]*\bin (\d+) seconds", status)
        ]
        if attempt or not delays or min(delays) > 180:
            raise PipelineError(
                "Source capacity unavailable (HTTP 429); approved progress is retained.",
                3,
                "source_access",
            )
        delay = min(delays) + 1
        print(
            f"Overpass capacity cooldown: {delay} seconds before the announced slot.",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)


def acquire_part(
    root: Path,
    query: str,
    endpoint: str,
    client: httpx.Client,
    max_buffer_bytes: int,
) -> tuple[bytes, dict[str, Any]]:
    identity = canonical_hash(
        {"query": query, "endpoint": endpoint, "privacy_policy": PRIVACY_POLICY}
    )
    relative = f"data/raw/osm/parts/{identity[7:]}"
    directory = safe_path(root, relative)
    if directory.exists():
        manifest = read_json(directory / "part.json")
        with (directory / "response.osm").open("rb") as stream:
            payload = read_limited(stream, max_buffer_bytes)
        if (
            manifest.get("part_id") != identity
            or manifest.get("sha256") != hash_bytes(payload)
            or manifest.get("query") != query
            or manifest.get("endpoint") != endpoint
            or manifest.get("byte_count") != len(payload)
        ):
            raise PipelineError("Cached historical OSM part failed integrity verification.")
        inspect_osm(payload, OSM_EFFECTIVE_AT, max_buffer_bytes, require_network=False)
        return payload, manifest
    for attempt in range(2):
        try:
            respect_service_capacity(client, endpoint)
            payload = download_bytes(
                endpoint, client=client, max_buffer_bytes=max_buffer_bytes, data={"data": query}
            )
            inspect_osm(payload, OSM_EFFECTIVE_AT, max_buffer_bytes, require_network=False)
            break
        except PipelineError as error:
            if not attempt and error.code == "source_access" and "HTTP 429" in str(error):
                print(
                    "Overpass rate limit: respecting a 60-second cooldown before one retry.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(60)
                continue
            if (
                attempt
                or error.code not in {"source_access", "source_incomplete"}
                or "429" in str(error)
            ):
                raise
            print(
                "Historical part did not complete; retrying once without changing its query.",
                file=sys.stderr,
                flush=True,
            )
    manifest = {
        "part_id": identity,
        "status": "privacy_approved_part_not_complete_network",
        "privacy_policy_version": PRIVACY_POLICY,
        "endpoint": endpoint,
        "query": query,
        "effective_at": "2018-12-31T23:59:59Z",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "sha256": hash_bytes(payload),
        "byte_count": len(payload),
    }
    with atomic_directory(root, relative) as staging:
        (staging / "response.osm").write_bytes(payload)
        write_json(staging / "part.json", manifest)
    return payload, manifest


def fetch_osm_tiled(
    root: Path,
    *,
    effective_at: str,
    client: httpx.Client | None = None,
    max_buffer_bytes: int = BUFFER_LIMIT,
    output: str = "data/raw/osm",
    endpoint: str = OVERPASS_URL,
) -> SourceSnapshot:
    try:
        public_url(endpoint)
        effective = datetime.fromisoformat(effective_at.replace("Z", "+00:00"))
    except ValueError:
        raise PipelineError(
            "Invalid historical OSM configuration.", 2, "configuration_error"
        ) from None
    if effective != OSM_EFFECTIVE_AT:
        raise PipelineError(
            "The fixed historical OSM instant is required.", 5, "temporal_integrity"
        )
    root = root.resolve()
    safe_path(root, output)
    if client is None:
        with httpx.Client(
            timeout=90, verify=ssl.create_default_context(), follow_redirects=False
        ) as connection:
            return fetch_osm_tiled(
                root,
                effective_at=effective_at,
                client=connection,
                max_buffer_bytes=max_buffer_bytes,
                output=output,
                endpoint=endpoint,
            )
    boundary_query = osm_boundary_query()
    boundary_bytes, boundary_manifest = acquire_part(
        root, boundary_query, endpoint, client, max_buffer_bytes
    )
    boundary = inspect_osm(boundary_bytes, effective, max_buffer_bytes, require_network=False)
    tiles = historical_tiles(boundary)
    contents = [boundary_bytes]
    parts = [
        {
            "name": "parts/0000.osm",
            "query": boundary_query,
            "bounds": None,
            "sha256": boundary_manifest["sha256"],
            "byte_count": len(boundary_bytes),
            "retrieved_at": boundary_manifest["retrieved_at"],
        }
    ]
    total_bytes = len(boundary_bytes)
    pending = list(tiles)
    unresolved: list[dict[str, Any]] = []
    while pending:
        bounds = pending.pop(0)
        print(
            f"OSM historical tiles: {len(parts) - 1} complete, {len(pending) + 1} pending; {bounds}",
            file=sys.stderr,
            flush=True,
        )
        query = osm_tile_query(bounds)
        split_id = canonical_hash({"query": query, "endpoint": endpoint})[7:]
        split_path = safe_path(root, f"data/raw/osm/subdivisions/{split_id}.json")
        south, west, north, east = bounds
        middle_latitude, middle_longitude = (south + north) / 2, (west + east) / 2
        children = [
            (south, west, middle_latitude, middle_longitude),
            (south, middle_longitude, middle_latitude, east),
            (middle_latitude, west, north, middle_longitude),
            (middle_latitude, middle_longitude, north, east),
        ]
        split_record = {"bounds": bounds, "children": children, "endpoint": endpoint}
        if split_path.exists():
            if canonical_hash(read_json(split_path)) != canonical_hash(split_record):
                raise PipelineError("Historical subdivision plan failed integrity verification.")
            pending[0:0] = children
            continue
        try:
            payload, manifest = acquire_part(root, query, endpoint, client, max_buffer_bytes)
        except PipelineError as error:
            if error.code not in {"source_access", "source_incomplete"} or "429" in str(error):
                raise
            if north - south <= 0.25 or east - west <= 0.25:
                unresolved.append(
                    {"bounds": bounds, "error_code": error.code, "message": str(error)}
                )
                print(
                    "Historical cell remains unavailable; retaining it as pending and processing other cells.",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            write_json(split_path, split_record)
            pending[0:0] = children
            print(
                "Subdividing the failed tile into four complete spatial cells.",
                file=sys.stderr,
                flush=True,
            )
            continue
        total_bytes += len(payload)
        if total_bytes > max_buffer_bytes:
            raise PipelineError(
                "OSM collection exceeds the memory buffer limit.", 4, "buffer_limit"
            )
        contents.append(payload)
        parts.append(
            {
                "name": f"parts/{len(parts):04d}.osm",
                "query": query,
                "bounds": bounds,
                "sha256": manifest["sha256"],
                "byte_count": len(payload),
                "retrieved_at": manifest["retrieved_at"],
            }
        )
    if unresolved:
        receipt = {
            "status": "incomplete_historical_collection",
            "endpoint": endpoint,
            "effective_at": "2018-12-31T23:59:59Z",
            "base_tile_count": len(tiles),
            "approved_cells": len(parts) - 1,
            "unresolved_cells": unresolved,
            "complete_network_published": False,
        }
        identifier = canonical_hash(receipt)[7:]
        receipt_path = safe_path(root, f"artifacts/verification/osm-acquisition-{identifier}.json")
        if not receipt_path.exists():
            write_json(receipt_path, receipt)
        raise PipelineError(
            f"Historical collection has {len(unresolved)} unavailable cells; approved progress is retained.",
            3,
            "source_incomplete",
        )
    collection = {
        "format": "osm-tiles-1",
        "effective_at": "2018-12-31T23:59:59Z",
        "endpoint": endpoint,
        "parts": parts,
    }
    manifest_bytes = canonical_bytes(collection)
    if total_bytes + len(manifest_bytes) > max_buffer_bytes:
        raise PipelineError("OSM collection exceeds the memory buffer limit.", 4, "buffer_limit")
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in [
            ("collection.json", manifest_bytes),
            *[(part["name"], content) for part, content in zip(parts, contents, strict=True)],
        ]:
            member = zipfile.ZipInfo(name, date_time=(2018, 12, 31, 23, 59, 58))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, content)
    print(
        f"All {len(parts) - 1} historical tiles acquired; verifying the complete collection.",
        file=sys.stderr,
        flush=True,
    )
    snapshot = ingest_osm(
        packed.getvalue(),
        root,
        effective_at=effective_at,
        source_url=endpoint,
        max_buffer_bytes=max_buffer_bytes,
        output=output,
    )
    boundary_document = extract_mg_boundary(snapshot, root)
    boundary_path = safe_path(root, "data/raw/osm/mg-boundary.geojson")
    if not boundary_path.exists():
        write_json(boundary_path, boundary_document)
    elif canonical_hash(read_json(boundary_path)) != canonical_hash(boundary_document):
        raise PipelineError("Existing MG boundary belongs to another source version.")
    return snapshot
