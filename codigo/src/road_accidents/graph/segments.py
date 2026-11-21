import math
import re
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point

from road_accidents.config import PipelineError
from road_accidents.data.osm import FEDERAL_REF
from road_accidents.data.schemas import validate_table
from road_accidents.provenance import canonical_hash


def values_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def positive_attribute(value: Any) -> float | None:
    values = {str(item) for item in values_list(value) if item is not None}
    if len(values) != 1:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(km/h|mph)?\s*", values.pop())
    if not match:
        return None
    number = float(match[1]) * (1.609344 if match[2] == "mph" else 1)
    return number if number > 0 and math.isfinite(number) else None


def build_segments(graph: Any, snapshot_id: str, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if graph.graph.get("crs") is None or boundary.crs is None or boundary.empty:
        raise PipelineError("Network and MG boundary require explicit reference systems.")
    state_geometry = boundary.to_crs(4326).geometry.union_all()
    if not state_geometry.is_valid:
        raise PipelineError("MG boundary is invalid.")
    network = graph.copy()
    for origin, destination, key, attributes in network.edges(keys=True, data=True):
        attributes["source_edge_key"] = f"{origin}:{destination}:{key}"
    undirected = nx.Graph(network)
    for component in nx.connected_components(undirected):
        if len(component) > 2 and all(undirected.degree(node) == 2 for node in component):
            network.nodes[min(component)]["ring_anchor"] = True
    if not network.graph.get("simplified"):
        network = ox.simplification.simplify_graph(
            network,
            node_attrs_include=["ring_anchor"],
            edge_attrs_differ=["osmid"],
            remove_rings=False,
        )
    network_version = canonical_hash(
        {
            "snapshot_id": snapshot_id,
            "boundary_wkb": state_geometry.normalize().wkb_hex,
            "policy": "physical-segments-mg-1",
            "metric_crs": "EPSG:5880",
        }
    )
    physical: dict[str, dict[str, Any]] = {}
    for origin, destination, _key, attributes in sorted(
        network.edges(keys=True, data=True),
        key=lambda edge: (str(edge[0]), str(edge[1]), str(edge[2])),
    ):
        geometry = attributes.get("geometry")
        if geometry is None:
            geometry = LineString(
                [
                    (network.nodes[origin]["x"], network.nodes[origin]["y"]),
                    (network.nodes[destination]["x"], network.nodes[destination]["y"]),
                ]
            )
        if geometry.is_empty or not geometry.is_valid or not state_geometry.covers(geometry):
            continue
        geometry = geometry.normalize()
        endpoints = sorted((str(origin), str(destination)))
        identifier = canonical_hash({"endpoints": endpoints, "geometry": geometry.wkb_hex})
        references = sorted(
            {
                reference.upper().replace(" ", "-")
                for value in values_list(attributes.get("ref", ""))
                for reference in FEDERAL_REF.findall(str(value))
            }
        )
        if not references:
            continue
        way_ids = {int(value) for value in values_list(attributes["osmid"])}
        edge_keys = {str(value) for value in values_list(attributes["source_edge_key"])}
        if identifier not in physical:
            physical[identifier] = {
                "segment_id": identifier,
                "network_version_id": network_version,
                "from_junction_id": endpoints[0],
                "to_junction_id": endpoints[1],
                "osm_way_ids": way_ids,
                "osm_edge_keys": edge_keys,
                "geometry": geometry,
                "highway_refs": set(references),
                "road_type": str(attributes.get("highway", "unknown")),
                "lanes": positive_attribute(attributes.get("lanes")),
                "maxspeed_kph": positive_attribute(attributes.get("maxspeed")),
            }
        else:
            record = physical[identifier]
            record["osm_way_ids"].update(way_ids)
            record["osm_edge_keys"].update(edge_keys)
            record["highway_refs"].update(references)
            for field, tag in (("lanes", "lanes"), ("maxspeed_kph", "maxspeed")):
                if record[field] != positive_attribute(attributes.get(tag)):
                    record[field] = None
    if not physical:
        raise PipelineError("No federal segments wholly inside MG are available.")
    records = []
    junctions: dict[str, set[str]] = {}
    for identifier, record in sorted(physical.items()):
        for field in ("osm_way_ids", "osm_edge_keys", "highway_refs"):
            record[field] = sorted(record[field])
        for endpoint in (record["from_junction_id"], record["to_junction_id"]):
            junctions.setdefault(endpoint, set()).add(identifier)
        records.append(record)
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
    projected = frame.to_crs(5880)
    frame["length_m"] = projected.length.astype(float)
    frame["chord_length_m"] = [
        float(Point(geometry.coords[0]).distance(Point(geometry.coords[-1])))
        for geometry in projected.geometry
    ]
    frame["sinuosity"] = [
        length / chord if chord > 0 else float("nan")
        for length, chord in zip(frame.length_m, frame.chord_length_m, strict=True)
    ]
    frame["lanes"] = frame.lanes.astype(float)
    frame["maxspeed_kph"] = frame.maxspeed_kph.astype(float)
    frame["missing_attributes"] = [
        [
            field
            for field in ("lanes", "maxspeed_kph", "sinuosity")
            if not math.isfinite(float(row[field]))
        ]
        for row in frame.to_dict("records")
    ]
    frame["is_isolated"] = [
        len(junctions[row.from_junction_id] | junctions[row.to_junction_id]) == 1
        for row in frame.itertuples()
    ]
    frame.attrs["boundary_policy"] = (
        "Only wholly covered geometries; omitted crossing segments remain a reported spatial limitation."
    )
    validate_table("RoadSegment", frame)
    return frame
