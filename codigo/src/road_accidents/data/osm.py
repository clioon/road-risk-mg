import io
import json
import math
import re
import ssl
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx
import networkx as nx
import osmnx as ox
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from filelock import FileLock
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import polygonize, unary_union

from road_accidents.config import PipelineError
from road_accidents.data.prf import PERSONAL_VALUE
from road_accidents.data.schemas import (
    BUFFER_LIMIT,
    OSM_EFFECTIVE_AT,
    PRIVACY_POLICY,
    Profile,
    SourceSnapshot,
    public_url,
)
from road_accidents.data.storage import (
    atomic_directory,
    download_bytes,
    publish_snapshot,
    read_limited,
    safe_path,
    write_json,
)
from road_accidents.provenance import canonical_hash, hash_bytes

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
FEDERAL_REF = re.compile(r"(?<![A-Z])BR[- ]?\d{2,3}(?!\d)", re.IGNORECASE)
ACCESSORY_REF = re.compile(r"\b(?:AMG|LMG|MGC)[- ]?\d{3,4}\b", re.IGNORECASE)
OSM_TAGS = {
    "highway",
    "ref",
    "int_ref",
    "name",
    "lanes",
    "lanes:forward",
    "lanes:backward",
    "maxspeed",
    "maxspeed:forward",
    "maxspeed:backward",
    "maxspeed:hgv",
    "maxheight",
    "oneway",
    "junction",
    "bridge",
    "tunnel",
    "layer",
    "surface",
    "width",
    "access",
    "service",
    "area",
    "toll",
    "lit",
    "shoulder",
    "smoothness",
    "network",
    "type",
    "route",
    "boundary",
    "admin_level",
    "ISO3166-2",
    "source",
    "source:geometry",
    "source:ref",
    "wikidata",
    "wikipedia",
    "note",
    "charge",
    "barrier",
    "operator",
    "source:description",
    "description",
    "source:maxspeed",
    "incorrect:name",
    "IBGE:CD_ADMINIS",
    "foot",
    "transit:lanes",
    "turn:lanes",
    "source:name",
    "history",
    "destination",
    "construction",
    "alt_name",
    "covered",
    "place",
    "website",
    "timezone",
    "short_name",
    "is_in:country",
    "IBGE:GEOCODIGO",
    "flag",
    "name:ar",
    "name:be",
    "name:bg",
    "name:de",
    "name:en",
    "name:eo",
    "name:fa",
    "name:gl",
    "name:he",
    "name:hi",
    "name:hy",
    "name:ja",
    "name:ka",
    "name:ko",
    "name:ku",
    "name:la",
    "name:lt",
    "name:lv",
    "name:mk",
    "name:mr",
    "name:os",
    "name:pl",
    "name:pt",
    "name:ru",
    "name:sr",
    "name:tg",
    "name:th",
    "name:uk",
    "name:uz",
    "name:zh",
    "source:highway_classification",
    "source:surface",
    "note:pt",
    "old_ref",
    "FIXME",
    "loc_name",
    "lanes:psv",
    "source:highway",
    "sidewalk",
    "cycleway:right",
    "destination:lanes",
    "old_name",
}
ELEMENT_ATTRIBUTES = {
    "osm": {"version", "generator", "copyright", "attribution", "license"},
    "node": {"id", "lat", "lon", "version", "timestamp", "visible"},
    "way": {"id", "version", "timestamp", "visible"},
    "relation": {"id", "version", "timestamp", "visible"},
    "road_attributes": {"id"},
    "route_attributes": {"id"},
    "nd": {"ref"},
    "member": {"type", "ref", "role"},
    "tag": {"k", "v"},
    "meta": {"osm_base", "areas"},
    "note": set(),
    "bounds": {"minlat", "minlon", "maxlat", "maxlon"},
}


def inspect_osm(
    payload: bytes,
    effective_at: datetime,
    limit: int = BUFFER_LIMIT,
    *,
    require_network: bool = True,
) -> Element:
    if effective_at != OSM_EFFECTIVE_AT:
        raise PipelineError(
            "Only the frozen historical OSM instant is permitted.", 5, "temporal_integrity"
        )
    if limit <= 0 or len(payload) > limit:
        raise PipelineError("Source exceeds the memory buffer limit.", 4, "buffer_limit")
    if payload.startswith(b"PK"):
        return inspect_osm_archive(payload, effective_at, limit)
    try:
        root: Element = ElementTree.fromstring(
            payload, forbid_dtd=True, forbid_entities=True, forbid_external=True
        )
    except (ElementTree.ParseError, DefusedXmlException, ValueError):
        raise PipelineError("OSM source is not safe XML.", 4, "privacy_schema_rejected") from None
    if root.tag != "osm" or root.attrib.get("version") != "0.6":
        raise PipelineError("Unsupported OSM source schema.")
    if root.find("remark") is not None:
        raise PipelineError(
            "OSM service returned an incomplete or failed historical response.",
            3,
            "source_incomplete",
        )
    for element in root.iter():
        if (
            element.tag not in ELEMENT_ATTRIBUTES
            or not set(element.attrib) <= ELEMENT_ATTRIBUTES[element.tag]
        ):
            raise PipelineError(
                "OSM source contains unapproved metadata.", 4, "privacy_schema_rejected"
            )
        if element.tag == "tag":
            value = element.get("v", "")
            if element.get("k") in {"ref", "name", "old_ref", "old_name", "alt_name"}:
                value = ACCESSORY_REF.sub("", value)
            if element.get("k") not in OSM_TAGS or PERSONAL_VALUE.search(value):
                raise PipelineError(
                    "OSM source rejected by privacy screening.", 4, "privacy_rejected"
                )
        if element.tag == "node":
            try:
                if not (
                    -90 <= float(element.attrib["lat"]) <= 90
                    and -180 <= float(element.attrib["lon"]) <= 180
                ):
                    raise ValueError
            except (ValueError, KeyError):
                raise PipelineError("OSM node coordinates are invalid.") from None
        if element.tag in {"node", "way", "relation"} and "timestamp" in element.attrib:
            try:
                if (
                    datetime.fromisoformat(element.attrib["timestamp"].replace("Z", "+00:00"))
                    > effective_at
                ):
                    raise PipelineError(
                        "OSM object is newer than the frozen snapshot.", 5, "temporal_integrity"
                    )
            except (TypeError, ValueError):
                raise PipelineError("OSM timestamp is invalid.") from None
    if require_network and (not root.findall("node") or not root.findall("way")):
        raise PipelineError("OSM source has no usable network.", 3, "source_incomplete")
    return root


def merge_osm_roots(roots: list[Element]) -> Element:
    elements: dict[tuple[str, str], Element] = {}
    for root in roots:
        for element in root:
            kind = {"road_attributes": "way", "route_attributes": "relation"}.get(
                element.tag, element.tag
            )
            if kind not in {"node", "way", "relation"}:
                continue
            identifier = element.get("id", "")
            if not identifier.isdigit():
                raise PipelineError("OSM element identifier is invalid.")
            key = kind, identifier
            if element.tag != kind:
                projected = Element(kind, element.attrib)
                projected.extend(element)
                element = projected
            previous = elements.get(key)
            if previous is None:
                elements[key] = element
                continue
            previous_geometry = [
                (child.tag, child.attrib) for child in previous if child.tag != "tag"
            ]
            current_geometry = [
                (child.tag, child.attrib) for child in element if child.tag != "tag"
            ]
            if previous.attrib != element.attrib or (
                previous_geometry and current_geometry and previous_geometry != current_geometry
            ):
                raise PipelineError(
                    "OSM parts contain conflicting element geometry.", 3, "source_conflict"
                )
            old_tags = {tag.get("k"): tag.get("v") for tag in previous.findall("tag")}
            new_tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            if any(old_tags[name] != new_tags[name] for name in old_tags.keys() & new_tags.keys()):
                raise PipelineError(
                    "OSM parts contain conflicting element attributes.", 3, "source_conflict"
                )
            combined = Element(kind, previous.attrib)
            combined.extend(
                child
                for child in (previous if previous_geometry else element)
                if child.tag != "tag"
            )
            for name, value in sorted((old_tags | new_tags).items()):
                if name is None or value is None:
                    raise PipelineError("OSM tag is incomplete.")
                SubElement(combined, "tag", {"k": name, "v": value})
            elements[key] = combined
    result = Element("osm", {"version": "0.6", "generator": "road-accidents-verified-collection"})
    for key in sorted(
        elements, key=lambda item: ({"node": 0, "way": 1, "relation": 2}[item[0]], int(item[1]))
    ):
        result.append(elements[key])
    node_ids = {node.get("id") for node in result.findall("node")}
    for way in result.findall("way"):
        if len(way.findall("nd")) < 2 or any(
            node.get("ref") not in node_ids for node in way.findall("nd")
        ):
            raise PipelineError(
                "OSM collection has incomplete way geometry.", 3, "source_incomplete"
            )
    return result


def historical_tiles(boundary: Element) -> list[tuple[float, float, float, float]]:
    document = osm_boundary_document(boundary, "unpublished", "confirmatory")
    geometry = shape(document["features"][0]["geometry"])
    west, south, east, north = geometry.bounds
    return [
        (float(latitude), float(longitude), float(latitude + 1), float(longitude + 1))
        for latitude in range(math.floor(south), math.ceil(north))
        for longitude in range(math.floor(west), math.ceil(east))
        if geometry.intersects(box(longitude, latitude, longitude + 1, latitude + 1))
    ]


def osm_boundary_query() -> str:
    return (
        '[out:xml][timeout:60][date:"2018-12-31T23:59:59Z"];'
        "rel(315173)->.mg;.mg out body;way(r.mg)->.border;"
        ".border out skel;node(w.border);out skel;"
    )


def inspect_osm_archive(payload: bytes, effective_at: datetime, limit: int) -> Element:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if (
                len({member.filename for member in members}) != len(members)
                or any(member.is_dir() or member.flag_bits & 1 for member in members)
                or sum(member.file_size for member in members) > limit
            ):
                raise PipelineError("OSM collection exceeds its limit or has invalid members.")
            manifest = json.loads(archive.read("collection.json"))
            if (
                manifest.get("format") != "osm-tiles-1"
                or manifest.get("effective_at") != "2018-12-31T23:59:59Z"
            ):
                raise PipelineError("OSM collection protocol is invalid.")
            public_url(manifest["endpoint"])
            parts = manifest["parts"]
            if not isinstance(parts, list) or not parts:
                raise PipelineError("OSM collection is incomplete.", 3, "source_incomplete")
            expected_names = {"collection.json"} | {
                f"parts/{index:04d}.osm" for index in range(len(parts))
            }
            if {member.filename for member in members} != expected_names:
                raise PipelineError(
                    "OSM collection inventory is incomplete.", 3, "source_incomplete"
                )
            roots = []
            actual_tiles = []
            for index, part in enumerate(parts):
                name = f"parts/{index:04d}.osm"
                content = archive.read(name)
                if (
                    part["name"] != name
                    or hash_bytes(content) != part["sha256"]
                    or len(content) != part["byte_count"]
                    or content.startswith(b"PK")
                ):
                    raise PipelineError("OSM collection part failed integrity verification.")
                if index == 0:
                    if part["query"] != osm_boundary_query() or part.get("bounds") is not None:
                        raise PipelineError(
                            "OSM boundary query is not the declared historical query."
                        )
                else:
                    bounds = tuple(float(value) for value in part["bounds"])
                    if len(bounds) != 4 or part["query"] != osm_tile_query(
                        (bounds[0], bounds[1], bounds[2], bounds[3])
                    ):
                        raise PipelineError("OSM tile query is not the declared historical query.")
                    actual_tiles.append(bounds)
                roots.append(inspect_osm(content, effective_at, limit, require_network=False))
            expected_area = unary_union(
                [
                    box(west, south, east, north)
                    for south, west, north, east in historical_tiles(roots[0])
                ]
            )
            rectangles = [
                box(west, south, east, north) for south, west, north, east in actual_tiles
            ]
            actual_area = unary_union(rectangles)
            if not actual_area.equals(expected_area) or not math.isclose(
                sum(rectangle.area for rectangle in rectangles), actual_area.area, abs_tol=1e-10
            ):
                raise PipelineError(
                    "OSM collection does not cover every historical MG tile.",
                    3,
                    "source_incomplete",
                )
            return merge_osm_roots(roots)
    except (zipfile.BadZipFile, KeyError, TypeError, ValueError, RuntimeError):
        raise PipelineError(
            "OSM collection is invalid or incomplete.", 3, "source_incomplete"
        ) from None


def federal_references(root: Element) -> dict[int, str]:
    references: dict[int, str] = {}
    for way in root.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        reference = tags.get("ref", "")
        if "highway" in tags and FEDERAL_REF.search(reference):
            references[int(way.attrib["id"])] = reference
    for relation in root.findall("relation"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in relation.findall("tag")}
        if tags.get("route") == "road" and FEDERAL_REF.search(tags.get("ref", "")):
            for member in relation.findall("member"):
                if member.get("type") == "way":
                    references[int(member.attrib["ref"])] = tags["ref"]
    return references


def load_osm_graph(snapshot: SourceSnapshot, root: Path) -> Any:
    source = safe_path(root, snapshot.raw_path)
    with source.open("rb") as stream:
        payload = read_limited(stream, int(snapshot.metadata.get("max_buffer_bytes", BUFFER_LIMIT)))
    if hash_bytes(payload) != snapshot.sha256:
        raise PipelineError("OSM source checksum does not match its manifest.")
    parsed = inspect_osm(
        payload, OSM_EFFECTIVE_AT, int(snapshot.metadata.get("max_buffer_bytes", BUFFER_LIMIT))
    )
    references = federal_references(parsed)
    if payload.startswith(b"PK"):
        merged_payload = tostring(parsed, encoding="utf-8", xml_declaration=True)
        merged_relative = f"data/interim/osm-collections/{snapshot.sha256}"
        merged_directory = safe_path(root, merged_relative)
        source = merged_directory / "network.osm"
        if not merged_directory.exists():
            with atomic_directory(root, merged_relative) as staging:
                (staging / "network.osm").write_bytes(merged_payload)
        elif hash_bytes(source.read_bytes()) != hash_bytes(merged_payload):
            raise PipelineError("Derived OSM collection failed integrity verification.")
    graph = ox.graph_from_xml(source, simplify=False, retain_all=True)
    for origin, destination, key, attributes in list(graph.edges(keys=True, data=True)):
        way_id = int(attributes["osmid"])
        if way_id not in references or "highway" not in attributes:
            graph.remove_edge(origin, destination, key)
        else:
            attributes["ref"] = references[way_id]
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if not graph.number_of_edges():
        raise PipelineError("No eligible federal road was found.")
    return graph


def ingest_osm(
    payload: bytes,
    root: Path,
    *,
    effective_at: str,
    source_url: str = OVERPASS_URL,
    max_buffer_bytes: int = BUFFER_LIMIT,
    profile: Profile = "confirmatory",
    dry_run: bool = False,
    output: str = "data/raw/osm",
) -> SourceSnapshot:
    try:
        effective = datetime.fromisoformat(effective_at.replace("Z", "+00:00"))
    except ValueError:
        raise PipelineError("Historical instant is invalid.", 2, "configuration_error") from None
    parsed = inspect_osm(payload, effective, max_buffer_bytes)
    if not federal_references(parsed):
        raise PipelineError("OSM source contains no eligible federal roads.")
    digest = hash_bytes(payload)
    source_name = "source.zip" if payload.startswith(b"PK") else "source.osm"
    raw_path = (
        safe_path(root, Path(output) / digest / source_name).relative_to(root.resolve()).as_posix()
    )
    identifier = canonical_hash(
        {
            "snapshot_id": "sha256:" + digest,
            "source_url": source_url,
            "effective_at": effective.isoformat(),
            "profile": profile,
            "privacy_policy_version": PRIVACY_POLICY,
            "schema_version": "3.0.0",
            "raw_path": raw_path,
        }
    )
    graph_path = (Path(raw_path).parent / "network.graphml").as_posix()
    snapshot = SourceSnapshot(
        snapshot_id="sha256:" + digest,
        source_manifest_id=identifier,
        profile=profile,
        source_type="osm_network",
        source_url=source_url,
        effective_at=effective,
        retrieved_at=datetime.now(UTC),
        sha256=digest,
        byte_count=len(payload),
        media_type="application/zip" if payload.startswith(b"PK") else "application/xml",
        license="ODbL-1.0",
        raw_path=raw_path,
        manifest_path=f"data/manifests/sources/{identifier[7:]}.json",
        metadata={
            "graphml_path": graph_path,
            "effective_at": effective.isoformat(),
            "state": "MG",
            "max_buffer_bytes": max_buffer_bytes,
        },
    )
    if dry_run:
        return snapshot
    snapshot = publish_snapshot(root, snapshot, payload)
    with FileLock(str(root.resolve() / ".locks" / "osm-graphml.lock")):
        destination = safe_path(root, graph_path)
        if not destination.exists():
            graph = load_osm_graph(snapshot, root)
            ox.save_graphml(graph, destination)
    return snapshot


def extract_mg_boundary(snapshot: SourceSnapshot, root: Path) -> dict[str, Any]:
    payload = safe_path(root, snapshot.raw_path).read_bytes()
    parsed = inspect_osm(payload, OSM_EFFECTIVE_AT)
    return osm_boundary_document(parsed, snapshot.snapshot_id, snapshot.profile)


def osm_boundary_document(parsed: Element, snapshot_id: str, profile: Profile) -> dict[str, Any]:
    coordinates = {
        node.attrib["id"]: (float(node.attrib["lon"]), float(node.attrib["lat"]))
        for node in parsed.findall("node")
    }
    ways = {
        way.attrib["id"]: [node.attrib["ref"] for node in way.findall("nd")]
        for way in parsed.findall("way")
    }
    for relation in parsed.findall("relation"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in relation.findall("tag")}
        if tags.get("ISO3166-2") != "BR-MG" or tags.get("boundary") != "administrative":
            continue
        outer: list[LineString] = []
        inner: list[LineString] = []
        try:
            for member in relation.findall("member"):
                if member.get("type") != "way" or member.get("role") not in {"outer", "inner", ""}:
                    continue
                line = LineString([coordinates[node] for node in ways[member.attrib["ref"]]])
                (inner if member.get("role") == "inner" else outer).append(line)
            geometry = unary_union(list(polygonize(outer)))
            if inner:
                geometry = geometry.difference(unary_union(list(polygonize(inner))))
        except (KeyError, ValueError):
            raise PipelineError("MG boundary is incomplete.", 3, "source_incomplete") from None
        if geometry.is_empty or not geometry.is_valid:
            raise PipelineError("MG boundary cannot be assembled.", 3, "source_incomplete")
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "state": "MG",
                        "source_snapshot_id": snapshot_id,
                        "profile": profile,
                        "effective_at": OSM_EFFECTIVE_AT.isoformat(),
                    },
                    "geometry": mapping(geometry),
                }
            ],
        }
    raise PipelineError("Historical MG boundary is missing.", 3, "source_incomplete")


def osm_tile_query(bounds: tuple[float, float, float, float]) -> str:
    south, west, north, east = bounds
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise PipelineError("Invalid historical query bounds.", 2, "configuration_error")
    rectangle = ",".join(format(value, ".7f") for value in bounds)
    road_tags = (
        "highway",
        "ref",
        "name",
        "lanes",
        "maxspeed",
        "oneway",
        "junction",
        "bridge",
        "tunnel",
        "layer",
        "surface",
        "width",
        "access",
        "service",
        "area",
        "toll",
        "lit",
        "shoulder",
        "smoothness",
        "lanes:forward",
        "lanes:backward",
        "maxspeed:forward",
        "maxspeed:backward",
    )
    road_projection = ",".join(f"{json.dumps(key)}=t[{json.dumps(key)}]" for key in road_tags)
    route_projection = ",".join(
        f"{json.dumps(key)}=t[{json.dumps(key)}]"
        for key in ("route", "ref", "type", "network", "name")
    )
    return (
        '[out:xml][timeout:60][date:"2018-12-31T23:59:59Z"];'
        f"way({rectangle})->.spatial;"
        'way.spatial(if:is_tag("highway") && is_tag("ref"))->.referenced;'
        "rel(bw.spatial)->.parents;"
        'rel.parents(if:t["route"] == "road" && is_tag("ref"))->.routes;'
        '(.referenced;way.spatial(r.routes)(if:is_tag("highway"));)->.roads;'
        ".roads out skel;.roads;convert road_attributes ::id=id(),"
        f"{road_projection};out;node(w.roads);out skel;"
        ".routes out skel;.routes;convert route_attributes ::id=id(),"
        f"{route_projection};out;"
    )


def fetch_osm(
    root: Path,
    *,
    effective_at: str,
    client: httpx.Client | None = None,
    max_buffer_bytes: int = BUFFER_LIMIT,
    dry_run: bool = False,
    output: str = "data/raw/osm",
    endpoint: str = OVERPASS_URL,
) -> SourceSnapshot:
    try:
        public_url(endpoint)
    except ValueError:
        raise PipelineError(
            "OSM endpoint must be a public HTTPS URL without credentials.",
            2,
            "configuration_error",
        ) from None
    if datetime.fromisoformat(effective_at.replace("Z", "+00:00")) != OSM_EFFECTIVE_AT:
        raise PipelineError(
            "Only the frozen historical OSM instant is permitted.", 5, "temporal_integrity"
        )
    if client is None:
        with httpx.Client(
            timeout=930, verify=ssl.create_default_context(), follow_redirects=False
        ) as connection:
            return fetch_osm(
                root,
                effective_at=effective_at,
                client=connection,
                max_buffer_bytes=max_buffer_bytes,
                dry_run=dry_run,
                output=output,
                endpoint=endpoint,
            )
    query = (
        '[out:xml][timeout:900][maxsize:1073741824][date:"2018-12-31T23:59:59Z"];'
        'rel["boundary"="administrative"]["ISO3166-2"="BR-MG"]->.mg;'
        ".mg map_to_area->.area;"
        'rel(area.area)["ref"~"BR[- ]?[0-9]"]->.route_candidates;'
        'rel.route_candidates["route"="road"]->.routes;('
        'way(area.area)["ref"~"BR[- ]?[0-9]"];'
        "way(r.routes)(area.area);)->.federal_candidates;"
        'way.federal_candidates["highway"]->.roads;.roads out body;'
        "(way(r.mg); - .roads;)->.border;.border out skel;"
        "(.roads;.border;);node(w);out skel;.mg out body;.routes out body;"
    )
    payload = download_bytes(
        endpoint, client=client, max_buffer_bytes=max_buffer_bytes, data={"data": query}
    )
    snapshot = ingest_osm(
        payload,
        root,
        effective_at=effective_at,
        source_url=endpoint + "?" + urlencode({"data": query}),
        max_buffer_bytes=max_buffer_bytes,
        dry_run=dry_run,
        output=output,
    )
    if not dry_run:
        boundary = extract_mg_boundary(snapshot, root)
        path = safe_path(root, "data/raw/osm/mg-boundary.geojson")
        if not path.exists():
            write_json(path, boundary)
    return snapshot
