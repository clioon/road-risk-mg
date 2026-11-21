import io
import json
import zipfile
from urllib.parse import parse_qs
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx
import pytest

from road_accidents.config import PipelineError
from tests.fixtures.prf_factory import prf_bytes


@pytest.mark.parametrize("mode", ["bytes", "local", "download"])
def test_sensitive_last_record_never_reaches_managed_storage(tmp_path, capsys, mode):
    from road_accidents.data.prf import fetch_prf_source, import_prf, ingest_prf

    root = tmp_path / "managed"
    payload = prf_bytes(2026, sensitive_last=True)
    original = tmp_path / "original.csv"
    original.write_bytes(payload)
    with pytest.raises(PipelineError) as failure:
        if mode == "bytes":
            ingest_prf(payload, root, year=2026, source_url="https://example.org/prf.csv")
        elif mode == "local":
            import_prf(original, root, year=2026, source_url="https://example.org/prf.csv")
        else:
            with httpx.Client(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
            ) as client:
                fetch_prf_source("https://example.org/prf.csv", root, year=2026, client=client)
    assert failure.value.exit_code == 4
    assert "123.456.789-09" not in str(failure.value)
    assert "123.456.789-09" not in str(capsys.readouterr())
    assert not root.exists() or not list(root.rglob("*"))
    assert original.read_bytes() == payload


def test_approved_import_preserves_exact_bytes_and_original(tmp_path):
    from road_accidents.data.prf import import_prf

    original = tmp_path / "original.csv"
    original.write_bytes(prf_bytes(2026))
    root = tmp_path / "managed"
    first = import_prf(original, root, year=2026, source_url="https://example.org/prf.csv")
    second = import_prf(original, root, year=2026, source_url="https://example.org/prf.csv")
    assert first.snapshot_id == second.snapshot_id
    assert first.privacy_status == "passed"
    assert (root / first.raw_path).read_bytes() == original.read_bytes() == prf_bytes(2026)
    assert first.coverage_status == "unknown"


@pytest.mark.parametrize("catalog", [False, True])
def test_public_drive_preview_links_download_occurrence_bytes(tmp_path, catalog):
    from road_accidents.data.prf import fetch_prf_source

    source_url = "https://drive.google.com/file/d/public-file_2019/view?usp=drive_link/download"
    catalog_url = "https://example.org/prf-catalog"
    payload = prf_bytes(2019)

    def respond(request):
        if str(request.url) == catalog_url:
            return httpx.Response(
                200,
                text=(
                    "<table><tr><td>Acidentes 2019 (Agrupados por ocorr&#234;ncia)</td>"
                    f'<td><a href="{source_url}">Download</a></td></tr></table>'
                ),
            )
        if (
            request.url.host == "drive.google.com"
            and request.url.path == "/uc"
            and request.url.params.get("id") == "public-file_2019"
            and request.url.params.get("export") == "download"
        ):
            return httpx.Response(200, content=payload)
        return httpx.Response(200, text="<html>Drive preview, not CSV</html>")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        snapshot = fetch_prf_source(
            catalog_url if catalog else source_url,
            tmp_path,
            year=2019,
            client=client,
            catalog=catalog,
        )
    assert snapshot.source_url == source_url
    assert (tmp_path / snapshot.raw_path).read_bytes() == payload
    assert snapshot.privacy_status == "passed"


@pytest.mark.parametrize("identifier", ["9e+05", "1.23E+6"])
def test_numeric_scientific_notation_preserves_source_bytes(tmp_path, identifier):
    from road_accidents.data.prf import ingest_prf

    payload = (
        "id;data_inversa;horario;uf;latitude;longitude\n"
        f"{identifier};2019-01-07;12:00:00;MG;-20.0;-44.0\n"
    ).encode()
    snapshot = ingest_prf(payload, tmp_path, year=2019, source_url="https://example.org/prf.csv")
    assert snapshot.privacy_status == "passed"
    assert (tmp_path / snapshot.raw_path).read_bytes() == payload


@pytest.mark.parametrize("identifier", ["9e+05unapproved", "1.0e", "=9e+05"])
def test_numeric_fields_still_reject_free_text(tmp_path, identifier):
    from road_accidents.data.prf import ingest_prf

    payload = (
        "id;data_inversa;horario;uf;latitude;longitude\n"
        f"{identifier};2019-01-07;12:00:00;MG;-20.0;-44.0\n"
    ).encode()
    with pytest.raises(PipelineError):
        ingest_prf(
            payload, tmp_path / "managed", year=2019, source_url="https://example.org/prf.csv"
        )
    assert not (tmp_path / "managed").exists()


@pytest.mark.parametrize(
    "payload", [b"<html>Login</html>", b"id;cpf\n1;123.456.789-09\n", b"unknown;schema\n1;2\n"]
)
def test_unknown_schema_or_html_is_not_a_snapshot(tmp_path, payload):
    from road_accidents.data.prf import ingest_prf

    with pytest.raises(PipelineError):
        ingest_prf(
            payload, tmp_path / "managed", year=2026, source_url="https://example.org/prf.csv"
        )
    assert not (tmp_path / "managed").exists()


def test_buffer_limit_fails_without_spill(tmp_path):
    from road_accidents.data.prf import ingest_prf

    with pytest.raises(PipelineError, match="limit"):
        ingest_prf(
            prf_bytes(2026),
            tmp_path / "managed",
            year=2026,
            source_url="https://example.org/prf.csv",
            max_buffer_bytes=64,
        )
    assert not (tmp_path / "managed").exists()


def test_zip_path_escape_is_rejected_in_memory(tmp_path):
    from road_accidents.data.prf import ingest_prf

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.csv", prf_bytes(2026))
    with pytest.raises(PipelineError):
        ingest_prf(
            archive.getvalue(),
            tmp_path / "managed",
            year=2026,
            source_url="https://example.org/prf.zip",
        )
    assert not list(tmp_path.iterdir())


def test_wrong_year_is_rejected(tmp_path):
    from road_accidents.data.prf import ingest_prf

    with pytest.raises(PipelineError):
        ingest_prf(
            prf_bytes(2025),
            tmp_path / "managed",
            year=2026,
            source_url="https://example.org/prf.csv",
        )


@pytest.mark.parametrize("failure", ["login", "partial", "timeout"])
def test_remote_failures_do_not_publish_partial_source(tmp_path, failure):
    from road_accidents.data.prf import fetch_prf_source

    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "login":
            return httpx.Response(200, content=b"<html>Login</html>")
        return httpx.Response(200, content=prf_bytes(2026), headers={"Content-Length": "999999"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(PipelineError) as error:
            fetch_prf_source(
                "https://example.org/prf.csv", tmp_path / "managed", year=2026, client=client
            )
    assert error.value.exit_code == 3
    assert not (tmp_path / "managed").exists()


@pytest.mark.parametrize("status", [406, 429, 504, "timeout"])
def test_source_transport_failure_reports_safe_cause(tmp_path, status):
    from road_accidents.data.prf import fetch_prf_source

    def respond(request):
        if status == "timeout":
            raise httpx.ReadTimeout("private-response-marker", request=request)
        return httpx.Response(status, text="private-response-marker")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(PipelineError) as error:
            fetch_prf_source(
                "https://example.org/source.csv",
                tmp_path / "managed",
                year=2026,
                client=client,
            )
    assert error.value.exit_code == 3
    assert error.value.code == "source_access"
    assert "private-response-marker" not in str(error.value)
    assert ("timed out" if status == "timeout" else f"HTTP {status}") in str(error.value)
    assert not (tmp_path / "managed").exists()


def test_osm_accessory_road_refs_are_not_vehicle_plates(project_root):
    from road_accidents.data.osm import inspect_osm
    from road_accidents.data.schemas import OSM_EFFECTIVE_AT

    payload = (project_root / "tests/fixtures/network.osm").read_bytes()
    road = payload.replace(b'k="ref" v="BR-040"', b'k="ref" v="AMG-1108"')
    assert inspect_osm(road, OSM_EFFECTIVE_AT).tag == "osm"
    for rejected in (b"ABC-1234", b"AMG-1108 cpf: 123.456.789-09"):
        with pytest.raises(PipelineError):
            inspect_osm(road.replace(b"AMG-1108", rejected), OSM_EFFECTIVE_AT)


def test_osm_personal_metadata_rejected_before_cache(tmp_path, project_root):
    from road_accidents.data.osm import ingest_osm

    payload = (
        (project_root / "tests/fixtures/network.osm")
        .read_bytes()
        .replace(b'<node id="1"', b'<node user="synthetic-person" uid="999" id="1"')
    )
    with pytest.raises(PipelineError):
        ingest_osm(payload, tmp_path / "managed", effective_at="2018-12-31T23:59:59Z")
    assert not (tmp_path / "managed").exists()


def test_osm_missing_historical_snapshot_does_not_fallback(tmp_path):
    from road_accidents.data.osm import fetch_osm

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(PipelineError) as error:
            fetch_osm(tmp_path / "managed", effective_at="2018-12-31T23:59:59Z", client=client)
    assert error.value.exit_code == 3
    assert len(requests) == 1
    assert not (tmp_path / "managed").exists()


def test_overpass_requests_identify_research_client(tmp_path, project_root):
    from road_accidents.data.osm import fetch_osm

    payload = (project_root / "tests/fixtures/network.osm").read_bytes()

    def respond(request):
        if not request.headers.get("User-Agent", "").startswith("road-accidents/"):
            return httpx.Response(406)
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        snapshot = fetch_osm(
            tmp_path / "managed",
            effective_at="2018-12-31T23:59:59Z",
            client=client,
            dry_run=True,
        )
    assert snapshot.privacy_status == "passed"
    assert snapshot.effective_at.isoformat() == "2018-12-31T23:59:59+00:00"
    assert not (tmp_path / "managed").exists()


def test_overpass_runtime_error_is_incomplete_acquisition(tmp_path):
    from road_accidents.data.osm import fetch_osm

    payload = b'<osm version="0.6"><remark>runtime error: Query run out of memory</remark></osm>'
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    ) as client:
        with pytest.raises(PipelineError) as error:
            fetch_osm(tmp_path / "managed", effective_at="2018-12-31T23:59:59Z", client=client)
    assert error.value.code == "source_incomplete"
    assert error.value.exit_code == 3
    assert not (tmp_path / "managed").exists()


def test_historical_query_bounds_road_relation_expansion(tmp_path, project_root):
    from road_accidents.data.osm import fetch_osm

    payload = (project_root / "tests/fixtures/network.osm").read_bytes()

    def respond(request):
        query = parse_qs(request.content.decode())["data"][0]
        assert '[date:"2018-12-31T23:59:59Z"]' in query
        assert "way(r.routes)(area.area)" in query
        assert 'way.federal_candidates["highway"]' in query
        assert 'rel.route_candidates["route"="road"]' in query
        assert 'way(area.area)["highway"]' not in query
        assert "node(w)" in query
        assert ">>" not in query
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        snapshot = fetch_osm(
            tmp_path / "managed",
            effective_at="2018-12-31T23:59:59Z",
            client=client,
            dry_run=True,
        )
    assert snapshot.privacy_status == "passed"
    assert not (tmp_path / "managed").exists()


def test_approved_osm_preserves_bytes_and_loads_federal_graph(tmp_path, project_root):
    from road_accidents.data.osm import ingest_osm, load_osm_graph

    payload = (project_root / "tests/fixtures/network.osm").read_bytes()
    snapshot = ingest_osm(
        payload, tmp_path, effective_at="2018-12-31T23:59:59Z", profile="smoke_only"
    )
    graph = load_osm_graph(snapshot, tmp_path)
    assert graph.number_of_edges() > 0
    assert {attributes["osmid"] for _, _, attributes in graph.edges(data=True)} == set(
        range(101, 108)
    )
    assert (tmp_path / snapshot.raw_path).read_bytes() == payload
    assert (tmp_path / snapshot.metadata["graphml_path"]).is_file()


def test_tiled_osm_resumes_subdivision_and_rejects_missing_part(tmp_path, project_root):
    from road_accidents.data.osm import ingest_osm, load_osm_graph, osm_boundary_query
    from road_accidents.data.osm_acquisition import fetch_osm_tiled

    boundary = Element("osm", {"version": "0.6"})
    coordinates = [
        ("9001", "-19.9", "-44.9"),
        ("9002", "-19.9", "-44.1"),
        ("9003", "-19.1", "-44.1"),
        ("9004", "-19.1", "-44.9"),
    ]
    for identifier, latitude, longitude in coordinates:
        SubElement(boundary, "node", {"id": identifier, "lat": latitude, "lon": longitude})
    way = SubElement(boundary, "way", {"id": "9000"})
    for identifier in ("9001", "9002", "9003", "9004", "9001"):
        SubElement(way, "nd", {"ref": identifier})
    relation = SubElement(boundary, "relation", {"id": "315173"})
    SubElement(relation, "member", {"type": "way", "ref": "9000", "role": "outer"})
    for key, value in (("boundary", "administrative"), ("ISO3166-2", "BR-MG")):
        SubElement(relation, "tag", {"k": key, "v": value})
    fixture = (project_root / "tests/fixtures/network.osm").read_bytes()
    requests = []

    def respond(request):
        if request.url.path == "/api/status":
            return httpx.Response(200, text="2 slots available now.\n")
        query = parse_qs(request.content.decode())["data"][0]
        requests.append(query)
        if query == osm_boundary_query():
            return httpx.Response(200, content=tostring(boundary))
        if "way(-20.0000000,-45.0000000,-19.0000000,-44.0000000)" in query:
            return httpx.Response(504)
        return httpx.Response(200, content=fixture)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        first = fetch_osm_tiled(tmp_path, effective_at="2018-12-31T23:59:59Z", client=client)
        count = len(requests)
        second = fetch_osm_tiled(tmp_path, effective_at="2018-12-31T23:59:59Z", client=client)
    assert count == len(requests) == 7
    assert first.snapshot_id == second.snapshot_id
    assert len(list((tmp_path / "data/raw/osm/subdivisions").glob("*.json"))) == 1
    graph = load_osm_graph(first, tmp_path)
    assert {attributes["osmid"] for _, _, attributes in graph.edges(data=True)} == set(
        range(101, 108)
    )
    original = (tmp_path / first.raw_path).read_bytes()
    damaged = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(damaged, "w") as target:
        manifest = json.loads(source.read("collection.json"))
        assert len(manifest["parts"]) == 5
        for name in source.namelist():
            if name != "parts/0001.osm":
                target.writestr(name, source.read(name))
    with pytest.raises(PipelineError):
        ingest_osm(damaged.getvalue(), tmp_path / "rejected", effective_at="2018-12-31T23:59:59Z")
    assert not (tmp_path / "rejected").exists()


def test_tiled_acquisition_retains_successful_cells_after_an_unavailable_cell(
    tmp_path, project_root, monkeypatch
):
    from road_accidents.data import osm_acquisition
    from road_accidents.data.osm import osm_boundary_query

    fixture = (project_root / "tests/fixtures/network.osm").read_bytes()
    monkeypatch.setattr(
        osm_acquisition,
        "historical_tiles",
        lambda boundary: [(-20.0, -45.0, -19.75, -44.75), (-20.0, -44.75, -19.75, -44.5)],
    )
    queries = []

    def respond(request):
        query = parse_qs(request.content.decode())["data"][0]
        queries.append(query)
        if query != osm_boundary_query() and "-45.0000000" in query:
            return httpx.Response(504)
        return httpx.Response(200, content=fixture)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(PipelineError, match="1 unavailable cells"):
            osm_acquisition.fetch_osm_tiled(
                tmp_path,
                effective_at="2018-12-31T23:59:59Z",
                endpoint="https://example.org/overpass",
                client=client,
            )
    assert len(queries) == 4
    assert len(list((tmp_path / "data/raw/osm/parts").glob("*/part.json"))) == 2
    assert not (tmp_path / "data/manifests").exists()
    receipt = json.loads(
        next((tmp_path / "artifacts/verification").glob("osm-acquisition-*.json")).read_bytes()
    )
    assert receipt["approved_cells"] == 1
    assert len(receipt["unresolved_cells"]) == 1
    assert receipt["complete_network_published"] is False


def test_tiled_request_respects_announced_service_cooldown(tmp_path, monkeypatch):
    from road_accidents.data import osm_acquisition

    delays = []
    monkeypatch.setattr(osm_acquisition.time, "sleep", delays.append)
    calls = {"status": 0, "query": 0}

    def respond(request):
        if request.url.path == "/api/status":
            calls["status"] += 1
            text = (
                "Slot available after: 2026-09-22T13:15:00Z, in 2 seconds.\n"
                if calls["status"] == 1
                else "1 slots available now.\n"
            )
            return httpx.Response(200, text=text)
        calls["query"] += 1
        return (
            httpx.Response(429)
            if calls["query"] == 1
            else httpx.Response(200, content=b'<osm version="0.6"/>')
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        payload, manifest = osm_acquisition.acquire_part(
            tmp_path, "historical-query", "https://overpass-api.de/api/interpreter", client, 1024
        )
    assert delays == [3, 60]
    assert calls["query"] == 2
    assert payload == b'<osm version="0.6"/>'
    assert manifest["status"] == "privacy_approved_part_not_complete_network"


def test_osm_projection_keeps_shared_nodes_and_rejects_conflicts():
    from road_accidents.data.osm import inspect_osm, merge_osm_roots
    from road_accidents.data.schemas import OSM_EFFECTIVE_AT

    payload = (
        b'<osm version="0.6"><node id="1" lat="-20" lon="-44"/>'
        b'<node id="2" lat="-20.1" lon="-44"/><way id="10"><nd ref="1"/><nd ref="2"/></way>'
        b'<road_attributes id="10"><tag k="ref" v="BR-040"/><tag k="highway" v="trunk"/></road_attributes></osm>'
    )
    first = inspect_osm(payload, OSM_EFFECTIVE_AT, require_network=False)
    second = inspect_osm(payload, OSM_EFFECTIVE_AT, require_network=False)
    merged = merge_osm_roots([first, second])
    assert len(merged.findall("node")) == 2
    assert len(merged.findall("way")) == 1
    assert merged.find('way/tag[@k="ref"]').get("v") == "BR-040"
    conflict = inspect_osm(
        payload.replace(b'lat="-20"', b'lat="-21"'), OSM_EFFECTIVE_AT, require_network=False
    )
    with pytest.raises(PipelineError, match="conflicting"):
        merge_osm_roots([first, conflict])
