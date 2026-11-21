import json
import subprocess
from pathlib import Path


def test_canonical_hash_is_order_independent_but_content_sensitive() -> None:
    from road_accidents.provenance import canonical_hash

    assert canonical_hash({"year": 2026, "state": "MG"}) == canonical_hash(
        {"state": "MG", "year": 2026}
    )
    assert canonical_hash({"year": 2026}) != canonical_hash({"year": 2025})


def test_only_explicit_volatile_fields_are_excluded() -> None:
    from road_accidents.provenance import canonical_hash

    first = {"snapshot": "bytes-id", "created_at": "old", "coverage": "unknown"}
    second = {**first, "created_at": "new"}
    excluded = frozenset({"created_at"})
    assert canonical_hash(first, exclude_keys=excluded) == canonical_hash(
        second, exclude_keys=excluded
    )
    assert canonical_hash(first) != canonical_hash(second)
    assert canonical_hash(first, exclude_keys=excluded) != canonical_hash(
        {**second, "coverage": "partial"}, exclude_keys=excluded
    )


def test_byte_hash_preserves_source_identity(tmp_path: Path) -> None:
    from road_accidents.provenance import hash_bytes, hash_file

    payload = b"id;uf\r\n1;MG\r\n"
    source = tmp_path / "source.csv"
    source.write_bytes(payload)
    assert hash_file(source) == hash_bytes(payload)
    assert hash_bytes(payload) != hash_bytes(payload.replace(b"\r\n", b"\n"))
    assert source.read_bytes() == payload


def test_chunked_table_identity_matches_canonical_whole_table() -> None:
    import pandas as pd

    from road_accidents.data.pipeline import table_digest
    from road_accidents.provenance import canonical_hash

    frame = pd.DataFrame(
        {
            "segment_id": ["segment-1"] * 10001,
            "week_start": pd.date_range("2019-01-07", periods=10001, freq="D", tz="UTC"),
            "value": [0.123456789, float("nan"), -0.0] * 3333 + [1.0, 2.0],
            "attributes": [{"missing": None, "values": [1, 2]}] * 10001,
        }
    )
    for content in (frame, frame.iloc[:0], frame.iloc[:1]):
        expected = canonical_hash(
            {
                "columns": list(content.columns),
                "rows": json.loads(
                    content.to_json(orient="records", date_format="iso", double_precision=15)
                ),
            }
        )
        assert table_digest(content) == expected


def test_append_events_preserves_previous_history(tmp_path: Path) -> None:
    from road_accidents.provenance import append_event

    log = tmp_path / "events.jsonl"
    append_event(log, {"event": "started", "owner": "run-1"})
    prefix = log.read_bytes()
    append_event(log, {"event": "failed", "owner": "run-1"})
    assert log.read_bytes().startswith(prefix)
    assert [json.loads(line)["event"] for line in log.read_text().splitlines()] == [
        "started",
        "failed",
    ]


def test_uncommitted_code_has_content_identity_without_invented_revision(tmp_path: Path) -> None:
    from road_accidents.provenance import capture_code_state

    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    source = tmp_path / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = capture_code_state(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = capture_code_state(tmp_path)
    assert first["git_revision"] is None
    assert first["worktree_dirty"] is True
    assert first["code_hash"] != second["code_hash"]
    assert second["inventory"][0]["path"] == "example.py"
