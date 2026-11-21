from itertools import combinations

import geopandas as gpd
import pandas as pd

from road_accidents.config import PipelineError
from road_accidents.data.schemas import validate_table


def node_order(segments: gpd.GeoDataFrame) -> pd.DataFrame:
    if segments.empty or not segments.segment_id.is_unique:
        raise PipelineError("Graph nodes require unique, nonempty segment identities.")
    identifiers = sorted(segments.segment_id.tolist())
    return pd.DataFrame({"node_index": range(len(identifiers)), "segment_id": identifiers})


def build_adjacency(segments: gpd.GeoDataFrame) -> pd.DataFrame:
    node_order(segments)
    versions = set(segments.network_version_id)
    if len(versions) != 1:
        raise PipelineError("Segments must belong to one network version.")
    network_version = versions.pop()
    junctions: dict[str, set[str]] = {}
    for row in segments.itertuples():
        for endpoint in (str(row.from_junction_id), str(row.to_junction_id)):
            junctions.setdefault(endpoint, set()).add(str(row.segment_id))
    links = [
        {
            "network_version_id": network_version,
            "segment_id_low": low,
            "segment_id_high": high,
            "shared_junction_id": junction,
        }
        for junction, identifiers in sorted(junctions.items())
        for low, high in combinations(sorted(identifiers), 2)
    ]
    frame = pd.DataFrame(
        links,
        columns=["network_version_id", "segment_id_low", "segment_id_high", "shared_junction_id"],
    )
    validate_table("SegmentAdjacency", frame)
    return frame
