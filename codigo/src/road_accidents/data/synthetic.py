import csv
import hashlib
import io
from datetime import date, timedelta
from typing import Any

COLUMNS = (
    "id",
    "data_inversa",
    "horario",
    "uf",
    "br",
    "km",
    "municipio",
    "causa_acidente",
    "tipo_acidente",
    "latitude",
    "longitude",
)


def occurrence(year: int, identifier: int = 1, **values: str) -> dict[str, str]:
    first = date(year, 1, 1)
    monday = first + timedelta(days=(7 - first.weekday()) % 7)
    return {
        "id": str(year * 100 + identifier),
        "data_inversa": monday.isoformat(),
        "horario": "12:00:00",
        "uf": "MG",
        "br": "40",
        "km": "500,0",
        "municipio": "BELO HORIZONTE",
        "causa_acidente": "Falta de atencao",
        "tipo_acidente": "Colisao traseira",
        "latitude": "-20,0",
        "longitude": "-44,015",
    } | values


def annual_rows(year: int, *, revision: bool = False) -> list[dict[str, str]]:
    first = occurrence(year)
    rows = [
        first,
        dict(first),
        occurrence(year, 2, uf="SP"),
        occurrence(year, 3, latitude="0"),
        occurrence(year, 4, data_inversa="invalid-date"),
        occurrence(year, 5, longitude="-43,99"),
        occurrence(year, 6, data_inversa=f"{year}-07-20"),
        occurrence(year, 7, data_inversa=f"{year}-07-31"),
    ]
    if year != 2026:
        rows.append(occurrence(year, 8, data_inversa=f"{year}-12-30"))
    if revision:
        rows.append(occurrence(year, 9, data_inversa=f"{year}-07-21"))
    return rows


def csv_bytes(rows: list[dict[str, str]], *, sensitive_last: bool = False) -> bytes:
    records = [dict(row) for row in rows]
    if sensitive_last:
        records[-1]["municipio"] = "CPF 123.456.789-09"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS, delimiter=";", lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("cp1252")


def prf_bytes(year: int, *, revision: bool = False, sensitive_last: bool = False) -> bytes:
    return csv_bytes(annual_rows(year, revision=revision), sensitive_last=sensitive_last)


def coverage_document(
    year: int, snapshot_id: str, *, available_at: str | None = None
) -> dict[str, Any]:
    end = f"{year}-07-31" if year == 2026 else f"{year}-12-31"
    statement = f"Synthetic coverage only: {snapshot_id}; {year}-01-01/{end}"
    return {
        "snapshot_id": snapshot_id,
        "nominal_year": year,
        "coverage_status": "partial" if year == 2026 else "complete",
        "coverage_intervals": [{"start": f"{year}-01-01", "end": end}],
        "available_at": available_at,
        "evidence": {
            "kind": "synthetic",
            "statement": statement,
            "sha256": hashlib.sha256(statement.encode()).hexdigest(),
            "verified_by": "synthetic-fixture",
            "verified_at": "2026-09-01T00:00:00Z",
        },
    }
