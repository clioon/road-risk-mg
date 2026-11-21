import csv
import io
import math
import re
import unicodedata
import zipfile
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit

import geopandas as gpd
import httpx
import pandas as pd
from shapely.geometry import Point

from road_accidents.config import PipelineError
from road_accidents.data.schemas import (
    BUFFER_LIMIT,
    PRIVACY_POLICY,
    Profile,
    SourceSnapshot,
    public_url,
)
from road_accidents.data.storage import download_bytes, publish_snapshot, read_limited, safe_path
from road_accidents.provenance import canonical_hash, hash_bytes

REQUIRED_COLUMNS = {"id", "data_inversa", "horario", "uf", "latitude", "longitude"}
ALLOWED_COLUMNS = REQUIRED_COLUMNS | {
    "dia_semana",
    "br",
    "km",
    "municipio",
    "causa_acidente",
    "tipo_acidente",
    "classificacao_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "condicao_meteorologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "feridos",
    "veiculos",
    "regional",
    "delegacia",
    "uop",
}
PERSONAL_VALUE = re.compile(
    r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b(?:cpf|cnh|telefone|e-mail|nome_completo)\s*[:= ]"
    r"|[\w.+-]+@[\w.-]+\.[a-z]{2,}|\b[A-Z]{3}[- ]?\d[A-Z0-9]\d{2}\b",
    re.IGNORECASE,
)
NUMERIC_COLUMNS = {
    "id",
    "br",
    "km",
    "latitude",
    "longitude",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "feridos",
    "veiculos",
}


def unpack_csv(payload: bytes, limit: int = BUFFER_LIMIT) -> bytes:
    if limit <= 0 or len(payload) > limit:
        raise PipelineError("Source exceeds the memory buffer limit.", 4, "buffer_limit")
    if not payload.startswith(b"PK"):
        return payload
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) != 1:
                raise PipelineError("A PRF archive must contain exactly one CSV.")
            member = members[0]
            path = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                path.is_absolute()
                or ".." in path.parts
                or PureWindowsPath(member.filename).drive
                or path.suffix.lower() != ".csv"
                or member.is_dir()
                or member.flag_bits & 1
            ):
                raise PipelineError("Unsafe or unsupported source archive.")
            if member.file_size > limit:
                raise PipelineError("Source exceeds the memory buffer limit.", 4, "buffer_limit")
            with archive.open(member) as source:
                return read_limited(source, limit)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError):
        raise PipelineError("Invalid source archive.") from None


def inspect_prf(
    payload: bytes, year: int, limit: int = BUFFER_LIMIT
) -> tuple[list[dict[str, str]], str]:
    if year not in range(2019, 2027):
        raise PipelineError("Unsupported source year.")
    decoded = unpack_csv(payload, limit)
    if decoded.lstrip().lower().startswith((b"<html", b"<!doctype")):
        raise PipelineError(
            "Source returned a web page instead of occurrence data.", 3, "source_access"
        )
    try:
        try:
            text, encoding = decoded.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError:
            text, encoding = decoded.decode("cp1252"), "cp1252"
        reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=";", strict=True)
        headers = reader.fieldnames or []
        if (
            len(headers) != len(set(headers))
            or not REQUIRED_COLUMNS <= set(headers)
            or not set(headers) <= ALLOWED_COLUMNS
        ):
            raise PipelineError("Unapproved source schema.", 4, "privacy_schema_rejected")
        records: list[dict[str, str]] = []
        observed_years: set[int] = set()
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise PipelineError("Malformed source record.")
            for column, value in row.items():
                if PERSONAL_VALUE.search(value) or (
                    column not in NUMERIC_COLUMNS and re.search(r"\b\d{11}\b", value)
                ):
                    raise PipelineError(
                        "Source rejected by privacy screening.", 4, "privacy_rejected"
                    )
                if (
                    column in NUMERIC_COLUMNS
                    and value.strip()
                    and not re.fullmatch(
                        r"[-+\d., ]+|[Nn][Aa][Nn]|[Nn]/[Aa]"
                        r"|[-+]?\d+(?:[.,]\d+)?[eE][-+]?\d+",
                        value,
                    )
                ):
                    raise PipelineError(
                        "Unapproved numeric field content.", 4, "privacy_schema_rejected"
                    )
            occurred = parse_date(row["data_inversa"])
            if occurred:
                observed_years.add(occurred.year)
            records.append(row)
        if not records or observed_years != {year}:
            raise PipelineError("Source year or occurrence granularity does not match.")
        return records, encoding
    except (UnicodeError, csv.Error):
        raise PipelineError("Source encoding or CSV schema is invalid.") from None


def parse_date(value: str) -> date | None:
    for pattern in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            continue
    return None


def ingest_prf(
    payload: bytes,
    root: Path,
    *,
    year: int,
    source_url: str,
    max_buffer_bytes: int = BUFFER_LIMIT,
    profile: Profile = "confirmatory",
    dry_run: bool = False,
    output: str = "data/raw/prf",
) -> SourceSnapshot:
    try:
        public_url(source_url)
    except ValueError:
        raise PipelineError(
            "A public source URL without credentials is required.", 2, "configuration_error"
        ) from None
    records, encoding = inspect_prf(payload, year, max_buffer_bytes)
    digest = hash_bytes(payload)
    raw_path = (
        safe_path(root, Path(output) / digest / "source.bin").relative_to(root.resolve()).as_posix()
    )
    identity = {
        "snapshot_id": "sha256:" + digest,
        "source_url": source_url,
        "nominal_year": year,
        "profile": profile,
        "privacy_policy_version": PRIVACY_POLICY,
        "schema_version": "3.0.0",
        "raw_path": raw_path,
    }
    manifest_id = canonical_hash(identity)
    observed = [occurred for row in records if (occurred := parse_date(row["data_inversa"]))]
    snapshot = SourceSnapshot(
        snapshot_id="sha256:" + digest,
        source_url=source_url,
        nominal_year=year,
        profile=profile,
        source_manifest_id=manifest_id,
        source_type="prf_occurrence_csv",
        granularity="occurrence",
        retrieved_at=datetime.now(UTC),
        sha256=digest,
        byte_count=len(payload),
        media_type="text/csv",
        license="PRF public occurrence data",
        observed_min_date=min(observed),
        observed_max_date=max(observed),
        raw_path=raw_path,
        manifest_path=f"data/manifests/sources/{manifest_id[7:]}.json",
        metadata={
            "encoding": encoding,
            "delimiter": ";",
            "record_count": len(records),
            "max_buffer_bytes": max_buffer_bytes,
        },
    )
    return snapshot if dry_run else publish_snapshot(root, snapshot, payload)


def import_prf(
    path: Path,
    root: Path,
    *,
    year: int,
    source_url: str,
    max_buffer_bytes: int = BUFFER_LIMIT,
    profile: Profile = "confirmatory",
    dry_run: bool = False,
    output: str = "data/raw/prf",
) -> SourceSnapshot:
    try:
        with path.open("rb") as source:
            payload = read_limited(source, max_buffer_bytes)
    except OSError:
        raise PipelineError("Local source is unavailable.", 3, "source_access") from None
    return ingest_prf(
        payload,
        root,
        year=year,
        source_url=source_url,
        max_buffer_bytes=max_buffer_bytes,
        profile=profile,
        dry_run=dry_run,
        output=output,
    )


class OccurrenceCatalog(HTMLParser):
    def __init__(self, year: int) -> None:
        super().__init__()
        self.year = year
        self.text: list[str] = []
        self.links: list[str] = []
        self.matches: list[str] = []
        self.in_row = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.in_row, self.text, self.links = True, [], []
        if tag == "a" and self.in_row:
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_data(self, data: str) -> None:
        if self.in_row:
            self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "tr":
            return
        text = (
            unicodedata.normalize("NFKD", " ".join(self.text))
            .encode("ascii", "ignore")
            .decode()
            .lower()
        )
        if f"acidentes {self.year}" in " ".join(text.split()) and "por ocorrencia" in text:
            self.matches.extend(self.links)
        self.in_row = False


def fetch_prf_source(
    url: str,
    root: Path,
    *,
    year: int,
    max_buffer_bytes: int = BUFFER_LIMIT,
    client: httpx.Client | None = None,
    profile: Profile = "confirmatory",
    catalog: bool = False,
    dry_run: bool = False,
    output: str = "data/raw/prf",
) -> SourceSnapshot:
    if catalog:
        page = download_bytes(url, max_buffer_bytes=max_buffer_bytes, client=client)
        parser = OccurrenceCatalog(year)
        parser.feed(page.decode("utf-8"))
        matches = sorted(set(parser.matches))
        if len(matches) != 1:
            raise PipelineError(
                "Official catalog does not identify a unique occurrence source.",
                3,
                "source_catalog",
            )
        url = urljoin(url, matches[0])
    try:
        parsed = urlsplit(public_url(url))
    except ValueError:
        raise PipelineError(
            "A public source URL without credentials is required.", 2, "configuration_error"
        ) from None
    download_url = url
    drive_file = re.fullmatch(r"/file/d/([A-Za-z0-9_-]+)/view/?", parsed.path)
    if parsed.netloc == "drive.google.com" and drive_file:
        download_url = "https://drive.google.com/uc?" + urlencode(
            {"export": "download", "id": drive_file.group(1)}
        )
    payload = download_bytes(download_url, max_buffer_bytes=max_buffer_bytes, client=client)
    return ingest_prf(
        payload,
        root,
        year=year,
        source_url=url,
        max_buffer_bytes=max_buffer_bytes,
        profile=profile,
        dry_run=dry_run,
        output=output,
    )


def numeric_value(value: str) -> float | None:
    try:
        number = float(value.strip().replace(",", "."))
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def curate_prf(
    snapshot: SourceSnapshot, root: Path, boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    if (
        snapshot.source_type != "prf_occurrence_csv"
        or snapshot.nominal_year is None
        or snapshot.status != "validated"
    ):
        raise PipelineError("Curated data requires an approved occurrence snapshot.")
    with safe_path(root, snapshot.raw_path).open("rb") as source:
        payload = read_limited(source, int(snapshot.metadata.get("max_buffer_bytes", BUFFER_LIMIT)))
    if hash_bytes(payload) != snapshot.sha256:
        raise PipelineError("PRF source checksum does not match its manifest.")
    rows, _encoding = inspect_prf(
        payload, snapshot.nominal_year, int(snapshot.metadata.get("max_buffer_bytes", BUFFER_LIMIT))
    )
    if boundary.crs is None or boundary.empty or not boundary.geometry.is_valid.all():
        raise PipelineError("A valid, versioned MG boundary is required.")
    state_geometry = boundary.to_crs(4326).geometry.union_all()
    seen: dict[str, str] = {}
    curated = []
    for ordinal, row in enumerate(rows, start=1):
        record_id = f"{snapshot.snapshot_id}:{ordinal:08d}"
        reasons: list[str] = []
        identifier = row["id"].strip()
        try:
            number = Decimal(identifier.replace(",", "."))
            if number <= 0 or number != number.to_integral_value():
                raise InvalidOperation
            identifier = str(int(number))
        except (InvalidOperation, ValueError):
            identifier = ""
            reasons.append("schema_error")
        duplicate_of = seen.get(identifier) if identifier else None
        if duplicate_of:
            reasons.append("duplicate")
        elif identifier:
            seen[identifier] = record_id
        state = row["uf"].strip().upper()
        if state != "MG":
            reasons.append("outside_scope")
        occurred: Any = pd.NaT
        occurred_date = parse_date(row["data_inversa"])
        try:
            clock = time.fromisoformat(row["horario"].strip())
            if occurred_date is not None and clock.tzinfo is None:
                occurred = pd.Timestamp(datetime.combine(occurred_date, clock)).tz_localize(
                    "America/Sao_Paulo", ambiguous="NaT", nonexistent="NaT"
                )
        except ValueError:
            pass
        if pd.isna(occurred):
            reasons.append("missing_datetime")
        latitude, longitude = numeric_value(row["latitude"]), numeric_value(row["longitude"])
        geometry = None
        if latitude is None or longitude is None:
            reasons.append("missing_coordinates")
        elif latitude == 0 or longitude == 0:
            reasons.append("zero_coordinates")
        elif not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            reasons.append("invalid_coordinates")
        else:
            geometry = Point(longitude, latitude)
            if not state_geometry.covers(geometry):
                reasons.append("outside_minas_gerais")
        highway = numeric_value(row.get("br", ""))
        curated.append(
            {
                "record_id": record_id,
                "accident_id": identifier or None,
                "source_snapshot_id": snapshot.snapshot_id,
                "occurred_at": occurred,
                "available_at": None,
                "availability_evidence": None,
                "state_code": state,
                "highway_ref": f"BR-{int(highway):03d}"
                if highway is not None and highway.is_integer()
                else None,
                "kilometer": numeric_value(row.get("km", "")),
                "latitude": latitude,
                "longitude": longitude,
                "geometry": geometry,
                "descriptive_attributes": {
                    key: value
                    for key, value in row.items()
                    if key not in NUMERIC_COLUMNS | {"data_inversa", "horario", "uf"}
                },
                "quality_status": "invalid" if reasons else "valid",
                "exclusion_reason": reasons[0] if reasons else None,
                "exclusion_reasons": reasons,
                "target_eligible": False,
                "target_exclusion_reason": reasons[0] if reasons else "unknown_coverage",
                "duplicate_of": duplicate_of,
            }
        )
    frame = gpd.GeoDataFrame(curated, geometry="geometry", crs=4326)
    frame["occurred_at"] = pd.to_datetime(frame.occurred_at, utc=True)
    frame["available_at"] = pd.to_datetime(frame.available_at, utc=True)
    return frame
