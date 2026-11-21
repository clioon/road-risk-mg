from datetime import date, timedelta

import numpy as np
import pandas as pd

from road_accidents.features.temporal import FEATURE_COLUMNS


def model_frame(year: int = 2019, weeks: int = 4) -> tuple[pd.DataFrame, np.ndarray]:
    first = date(year, 1, 1)
    first += timedelta(days=(7 - first.weekday()) % 7)
    rows = []
    labels = []
    for week in range(weeks):
        for segment in range(4):
            row = dict.fromkeys(FEATURE_COLUMNS, 0.0)
            row.update(
                segment_id=f"segment-{segment}",
                week_start=first + timedelta(weeks=week),
                length_m=float(100 + segment * 200),
                road_type="trunk" if segment % 2 else "primary",
                lanes=float(segment % 2 + 1),
                maxspeed_kph=80.0,
                sinuosity=1.1,
                calendar_year=year,
                calendar_month=1,
                calendar_iso_week=week + 1,
                accidents_lag_52w=float("nan"),
            )
            rows.append(row)
            labels.append((week + segment) % 3 == 0)
    return pd.DataFrame(rows), np.array(labels, dtype=np.int64)


def graph_structure():
    nodes = pd.DataFrame(
        {"node_index": range(4), "segment_id": [f"segment-{index}" for index in range(4)]}
    )
    adjacency = pd.DataFrame(
        {
            "segment_id_low": ["segment-0", "segment-1"],
            "segment_id_high": ["segment-1", "segment-2"],
        }
    )
    return nodes, adjacency
