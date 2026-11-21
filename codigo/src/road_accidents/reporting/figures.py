from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from road_accidents.evaluation.metrics import METRICS
from road_accidents.experiments.schemas import FAMILIES, EvaluationRun


def render_figures(evaluations: list[EvaluationRun], directory: Path) -> list[str]:
    figures = []
    colors = ["#197278", "#c44e52"]
    for metric in METRICS:
        for cohort in ("full_period", "matched_season"):
            figure, axes = plt.subplots(figsize=(10, 4.5), layout="constrained")
            positions = np.arange(len(FAMILIES), dtype=float)
            for index, evaluation in enumerate(evaluations):
                values, deviations = [], []
                for family in FAMILIES:
                    aggregate = (
                        evaluation.aggregates[family + ":full"].get(cohort, {}).get(metric, {})
                    )
                    values.append(
                        aggregate.get("mean") if aggregate.get("mean") is not None else np.nan
                    )
                    deviations.append(aggregate.get("std") or 0.0)
                axes.bar(
                    positions + (index - 0.5) * 0.35,
                    values,
                    0.35,
                    yerr=deviations,
                    color=colors[index],
                    label=(
                        f"{evaluation.split} ({'partial' if evaluation.period['partial'] else 'complete'}; "
                        f"coverage through {evaluation.period['coverage_cutoff']})"
                    ),
                    capsize=3,
                )
                for position, value in zip(positions + (index - 0.5) * 0.35, values, strict=True):
                    if not np.isfinite(value):
                        axes.text(position, 0.03, "N/A", ha="center", fontsize=8)
            axes.set_xticks(positions, FAMILIES)
            axes.set_ylim(0, 1.05)
            axes.set_ylabel(metric.upper())
            axes.set_title(f"{metric.upper()} | {cohort} | {evaluations[0].profile}")
            axes.legend()
            axes.spines[["top", "right"]].set_visible(False)
            for extension in ("png", "pdf"):
                name = f"{metric}-{cohort}.{extension}"
                metadata = (
                    {"CreationDate": None, "ModDate": None}
                    if extension == "pdf"
                    else {"Software": "road-risk"}
                )
                figure.savefig(directory / name, dpi=160, metadata=metadata)
                figures.append(name)
            plt.close(figure)
    return figures
