import csv
import io
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from road_accidents.config import PipelineError
from road_accidents.data.pipeline import dataset_directory, load_dataset
from road_accidents.data.storage import atomic_directory, read_json, safe_path, write_json
from road_accidents.evaluation.metrics import METRICS
from road_accidents.evaluation.runner import load_evaluation
from road_accidents.experiments.freeze import load_bundle
from road_accidents.experiments.schemas import EvaluationRun, ExperimentRun
from road_accidents.experiments.training import load_run
from road_accidents.provenance import canonical_hash, hash_file
from road_accidents.reporting.figures import render_figures


def verify_report(root: Path, manifest: str) -> dict[str, Any]:
    report: dict[str, Any] = read_json(safe_path(root, manifest))
    if (
        report.get("schema_version") != "3.0.0"
        or report.get("status") != "completed"
        or report.get("manifest") != manifest
    ):
        raise PipelineError("A completed report manifest is required.", 7, "invalid_report")
    for relative, digest in report["artifact_hashes"].items():
        path = safe_path(root, relative)
        if not path.is_file() or hash_file(path) != digest:
            raise PipelineError(
                "Report artifact integrity verification failed.", 7, "report_changed"
            )
    return report


def report_rows(
    evaluations: list[EvaluationRun],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates, individual = [], []
    for evaluation in evaluations:
        for model_variant, cohorts in sorted(evaluation.aggregates.items()):
            family, variant = model_variant.split(":")
            for cohort, metrics in cohorts.items():
                for metric, result in metrics.items():
                    aggregates.append(
                        {
                            "split": evaluation.split,
                            "model": family,
                            "feature_set": variant,
                            "cohort": cohort,
                            "metric": metric,
                            "mean": result["mean"],
                            "sample_std": result["std"],
                            "valid_seed_count": result["valid_seed_count"],
                            "undefined_reason": result["undefined_reason"],
                            "expected_seeds": str(result["expected_seeds"]),
                            "evaluation_id": evaluation.evaluation_id,
                            "bundle_id": evaluation.bundle_id,
                            "dataset_version_id": evaluation.dataset_version_id,
                        }
                    )
        for run_id, cohorts in evaluation.results.items():
            for cohort, metrics in cohorts.items():
                for metric, result in metrics.items():
                    individual.append(
                        {
                            "split": evaluation.split,
                            "run_id": run_id,
                            "cohort": cohort,
                            "metric": metric,
                            **result,
                            "evaluation_id": evaluation.evaluation_id,
                            "dataset_version_id": evaluation.dataset_version_id,
                            "predictions": evaluation.predictions[run_id],
                        }
                    )
    return aggregates, individual


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No comparable observations.\n"
    columns = list(rows[0])
    output = io.StringIO()
    output.write("| " + " | ".join(columns) + " |\n")
    output.write("| " + " | ".join("---" for _column in columns) + " |\n")
    for row in rows:
        output.write(
            "| "
            + " | ".join(
                "undefined"
                if row[column] is None
                else str(row[column]).replace("|", "\\|").replace("\n", " ")
                for column in columns
            )
            + " |\n"
        )
    return output.getvalue()


def benchmark_rows(
    runs: list[ExperimentRun], evaluations: list[EvaluationRun]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    training = [
        {
            "run_id": run.run_id,
            "model": run.model_family,
            "feature_set": run.feature_set,
            "seed": run.seed,
            "decision_threshold": run.decision_threshold,
            **run.benchmark,
        }
        for run in runs
        if run.benchmark
    ]
    lookup = {run.run_id: run for run in runs}
    inference = [
        {
            "split": evaluation.split,
            "run_id": run_id,
            "model": lookup[run_id].model_family,
            "feature_set": lookup[run_id].feature_set,
            "seed": lookup[run_id].seed,
            **benchmark,
        }
        for evaluation in evaluations
        for run_id, benchmark in evaluation.inference_benchmarks.items()
    ]
    summary = []
    for family, variant in sorted({(row["model"], row["feature_set"]) for row in training}):
        selected = [
            row for row in training if (row["model"], row["feature_set"]) == (family, variant)
        ]
        times = [row["fit_seconds"] for row in selected]
        row = {
            "model": family,
            "feature_set": variant,
            "runs": len(selected),
            "fit_mean_seconds": fmean(times),
            "fit_std_seconds": stdev(times) if len(times) > 1 else None,
            "preprocessing_mean_seconds": fmean(item["preprocessing_seconds"] for item in selected),
            "validation_prediction_mean_seconds": fmean(
                item["validation_prediction_seconds"] for item in selected
            ),
            "max_sampled_rss_mib": max(item["sampled_peak_rss_mib"] for item in selected),
            "model_mean_mib": fmean(item["model_bytes"] for item in selected) / 1024**2,
        }
        for split in ("test_2025", "test_2026"):
            values = [
                item["prediction_seconds"]
                for item in inference
                if (item["model"], item["feature_set"], item["split"]) == (family, variant, split)
            ]
            row[split + "_prediction_mean_seconds"] = fmean(values) if values else None
        summary.append(row)
    return training, inference, summary


def metric_summary(evaluation: EvaluationRun, cohort: str) -> list[dict[str, Any]]:
    rows = []
    for key, cohorts in sorted(evaluation.aggregates.items()):
        if not key.endswith(":full") or cohort not in cohorts:
            continue
        row: dict[str, Any] = {"split": evaluation.split, "model": key.split(":")[0]}
        for metric in METRICS:
            result = cohorts[cohort][metric]
            row[metric] = (
                "undefined"
                if result["mean"] is None
                else f"{result['mean']:.6f} +/- {result['std']:.6f}"
                if result["std"] is not None
                else f"{result['mean']:.6f} (one seed)"
            )
        rows.append(row)
    return rows


def generate_report(
    root: Path, main_evaluation_id: str, temporal_evaluation_id: str, output: str
) -> dict[str, Any]:
    root = root.resolve()
    main = load_evaluation(root, main_evaluation_id)
    temporal = load_evaluation(root, temporal_evaluation_id)
    if (
        main.split != "test_2025"
        or temporal.split != "test_2026"
        or (main.bundle_id, main.dataset_version_id, main.profile)
        != (temporal.bundle_id, temporal.dataset_version_id, temporal.profile)
    ):
        raise PipelineError(
            "Report requires compatible, separate main and temporal evaluations.",
            7,
            "incompatible_evaluations",
        )
    bundle = load_bundle(root, main.bundle_id)
    version = load_dataset(root, main.dataset_version_id)
    coverage = read_json(dataset_directory(root, version.dataset_version_id) / "coverage.json")
    assumed_sources = [
        {
            "nominal_year": entry["nominal_year"],
            "snapshot_id": entry["snapshot_id"],
            "intervals": entry["coverage_intervals"],
            "authorization": entry["evidence"]["document_path"],
        }
        for entry in coverage["sources"]
        if (entry.get("evidence") or {}).get("kind") == "assumption"
    ]
    identifier = canonical_hash(
        {
            "main": hash_file(root / main.manifest_path),
            "temporal": hash_file(root / temporal.manifest_path),
            "bundle_id": bundle.bundle_id,
        }
    )
    destination = safe_path(root, output)
    prefix = destination.relative_to(root).as_posix()
    if destination.exists():
        report = verify_report(root, prefix + "/manifest.json")
        if report["report_id"] != identifier:
            raise PipelineError(
                "Immutable report output already belongs to other evaluations.",
                7,
                "report_conflict",
            )
        return report
    evaluations = [main, temporal]
    runs = [load_run(root, run_id) for run_id in bundle.selected_run_ids]
    training_costs, inference_costs, computational_summary = benchmark_rows(runs, evaluations)
    directory = dataset_directory(root, version.dataset_version_id)
    network = read_json(directory / "network/manifest.json")
    populations = [
        {
            "split": split,
            "start": partition["start"],
            "end_exclusive": partition["end_exclusive"],
            "weeks": partition["week_count"],
            "segments": network["segment_count"],
            "observations": partition["row_count"],
            **{
                key: value
                for key, value in read_json(
                    directory / f"segment_week/{split}/target-audit.json"
                ).items()
                if key != "split"
            },
        }
        for split, partition in version.partition_manifests.items()
    ]
    validation_rows = [
        {
            "model": run.model_family,
            "feature_set": run.feature_set,
            "seed": run.seed,
            "threshold": run.decision_threshold,
            **{metric: run.validation_metrics[metric]["value"] for metric in METRICS},
            "run_id": run.run_id,
        }
        for run in runs
    ]
    preparation_receipts = {
        name: read_json(root / "artifacts/verification" / name)
        for name in (
            "data-acceptance-assumed-20260922.json",
            "development-pilot-20260922.json",
            "computational-budget-20260922.json",
            "reproduction-access.json",
        )
        if (root / "artifacts/verification" / name).is_file()
    }
    preparation_benchmarks = [
        {
            "stage": name,
            "elapsed_seconds": receipt["elapsed_seconds"],
            "sampled_peak_rss_mib": receipt.get("sampled_peak_rss_mib"),
            "interpretation": "preparation_or_development_pilot_not_final_test_metric",
        }
        for name, receipt in preparation_receipts.items()
        if "elapsed_seconds" in receipt
    ]
    ablations = []
    for family in ("graphsage", "gatv2"):
        full_metrics = main.aggregates[family + ":full"]["full_period"]
        for variant in ("without_road", "without_temporal", "without_topology"):
            ablated = main.aggregates.get(family + ":" + variant, {}).get("full_period")
            if ablated is None:
                continue
            for metric in METRICS:
                complete, reduced = full_metrics[metric], ablated[metric]
                differences = [
                    complete["values"][str(seed)] - reduced["values"][str(seed)]
                    for seed in bundle.seeds
                    if complete["values"][str(seed)] is not None
                    and reduced["values"][str(seed)] is not None
                ]
                ablations.append(
                    {
                        "model": family,
                        "ablation": variant,
                        "metric": metric,
                        "full_mean": complete["mean"],
                        "ablation_mean": reduced["mean"],
                        "paired_delta_mean": fmean(differences)
                        if len(differences) == len(bundle.seeds)
                        else None,
                        "paired_delta_std": stdev(differences)
                        if len(differences) == len(bundle.seeds) and len(differences) > 1
                        else None,
                        "interpretation": "full_minus_ablation_not_causal_or_significance_test",
                    }
                )
    aggregates, individual = report_rows(evaluations)
    deltas = []
    for evaluation in evaluations:
        baseline = evaluation.aggregates[bundle.baseline_reference + ":full"]
        for family in ("graphsage", "gatv2"):
            for cohort, metrics in evaluation.aggregates[family + ":full"].items():
                for metric in METRICS:
                    value, reference = metrics[metric]["mean"], baseline[cohort][metric]["mean"]
                    deltas.append(
                        {
                            "split": evaluation.split,
                            "model": family,
                            "reference": bundle.baseline_reference,
                            "cohort": cohort,
                            "metric": metric,
                            "delta": value - reference
                            if value is not None and reference is not None
                            else None,
                            "evaluation_id": evaluation.evaluation_id,
                            "interpretation": "difference_not_statistical_equivalence",
                        }
                    )
    with atomic_directory(root, prefix) as staging:
        write_csv(staging / "metrics.csv", aggregates)
        write_csv(staging / "per-run.csv", individual)
        write_csv(staging / "deltas.csv", deltas)
        write_csv(staging / "training-costs.csv", training_costs)
        write_csv(staging / "inference-costs.csv", inference_costs)
        write_csv(staging / "computational-summary.csv", computational_summary)
        write_csv(staging / "populations.csv", populations)
        write_csv(staging / "validation.csv", validation_rows)
        write_csv(staging / "ablations.csv", ablations)
        write_csv(staging / "preparation-costs.csv", preparation_benchmarks)
        write_json(staging / "execution-context.json", preparation_receipts)
        write_json(
            staging / "model-configurations.json",
            [
                {
                    "run_id": run.run_id,
                    "model": run.model_family,
                    "feature_set": run.feature_set,
                    "seed": run.seed,
                    "parameters": run.parameters,
                    "feature_columns": run.feature_columns,
                }
                for run in runs
            ],
        )
        write_json(
            staging / "data-audit.json",
            {"network": network, "row_counts": version.row_counts, "coverage": coverage},
        )
        periods = {evaluation.split: evaluation.period for evaluation in evaluations}
        write_json(staging / "periods.json", periods)
        period_rows = [
            {
                key: evaluation.period[key]
                for key in (
                    "split",
                    "coverage_basis",
                    "start",
                    "end_exclusive",
                    "coverage_cutoff",
                    "partial",
                    "week_count",
                    "segment_count",
                    "observation_count",
                    "event_count",
                    "prevalence",
                )
            }
            for evaluation in evaluations
        ]
        write_csv(staging / "periods.csv", period_rows)
        figure_names = render_figures(evaluations, staging)
        limitations = [
            "Static pre-2019 OSM network; later roads may be absent.",
            "Segments crossing the MG polygon boundary are omitted and must be audited before real acceptance.",
            "Unknown prior publication yields missing history, not zero accidents.",
            "Between-seed dispersion is not a confidence interval or an equivalence test.",
            "Partial-year observations are not annualized.",
            "Budgets and candidates are fixed before test access; no convergence guarantee or exhaustive hyperparameter optimization.",
            "No traffic exposure denominator: predictions concern recorded occurrence, not a causal or exposure-adjusted accident rate.",
            "Ablation differences are descriptive under the declared training budget; they do not establish causal feature importance.",
        ]
        if main.profile == "smoke_only":
            limitations.insert(0, "Synthetic one-seed validation; not a scientific result.")
        if assumed_sources:
            limitations.insert(
                0,
                "Coverage is assumed by explicit owner authorization, not independently verified. Missing records or months may create false negative labels and bias prevalence and predictive metrics; their impact is not asserted to be negligible.",
            )
        conclusions = []
        for evaluation in evaluations:
            values = {
                key.split(":")[0]: cohorts["full_period"]["auprc"]["mean"]
                for key, cohorts in evaluation.aggregates.items()
                if key.endswith(":full") and cohorts["full_period"]["auprc"]["mean"] is not None
            }
            if values:
                best = max(values, key=lambda family: values[family])
                conclusions.append(
                    f"{evaluation.split}: maior AP media observada = {best}, {values[best]:.6f}. "
                    "Este ranking e descritivo; nao altera a selecao ou os limiares congelados."
                )
        topology_rows = [
            row
            for row in ablations
            if row["ablation"] == "without_topology" and row["metric"] == "auprc"
        ]
        for row in topology_rows:
            delta = row["paired_delta_mean"]
            if delta is not None:
                conclusions.append(
                    f"{row['model']}: AP completa menos AP sem topologia = {delta * 100:.3f} pontos percentuais em 2025. "
                    "Valor positivo favorece o modelo completo neste recorte, sem teste de significancia."
                )
        report_text = (
            "# Relatorio Final: Acidentes Rodoviarios em Minas Gerais\n\n"
            + f"Profile: {main.profile}\n\nBundle: {bundle.bundle_id}\n\n"
            + (
                "**Conditional benchmark under owner-authorized, unverified temporal coverage.**\n\n"
                if assumed_sources
                else ""
            )
            + "## 1. Sintese dos Resultados\n\n"
            + "\n\n".join(conclusions)
            + f"\n\nReferencia escolhida somente em 2024: {bundle.baseline_reference}. "
            + "Resultados condicionados aos dados e ao protocolo registrados, sem uso operacional validado.\n\n"
            + "## 2. Pergunta e Protocolo\n\n"
            + "Prever pelo menos uma ocorrencia PRF em cada segmento rodoviario e semana completa. "
            + "Treino: 2019-2023; validacao: 2024; testes independentes: 2025 e 2026 parcial. "
            + "O alvo binario nao estima gravidade ou contagem. Pesos, transformadores e limiares sao os mesmos nos dois testes; nenhum fit ocorre na avaliacao.\n\n"
            + f"Seeds: {bundle.seeds}. Runs congelados: {len(runs)}. "
            + "AP significa average_precision_score, e nao integracao trapezoidal de uma curva PR. "
            + "Limiar maximiza F1 em 2024; empate escolhe o maior limiar. AUROC, F1, precisao e recall complementam AP. "
            + "Valores sao medias e desvios-padrao amostrais entre seeds, nao intervalos de confianca.\n\n"
            + "## 3. Fontes, Curadoria e Populacoes\n\n"
            + f"Linhas nacionais inspecionadas: {version.row_counts['approved_source_rows']}. "
            + f"Registros validos apos escopo e associacao: {version.row_counts['valid_records']}. "
            + f"Ocorrencias elegiveis como alvos: {version.row_counts['target_eligible']}. "
            + "Linhas fora de MG permanecem na auditoria, mas nao nos alvos. "
            + "Zeros representam ausencia de registro sob a cobertura declarada, nao prova de ausencia real de acidentes.\n\n"
            + markdown_table(populations)
            + "\nAuditoria completa: [data-audit.json](data-audit.json). Populacoes: [populations.csv](populations.csv).\n\n"
            + "## 4. Rede e Integracao Espacial\n\n"
            + f"OSM historico: {network['effective_at']}. Segmentos fisicos: {network['segment_count']}; "
            + f"conexoes nao direcionadas: {network['adjacency_count']}; juncoes: {network['junction_count']}. "
            + "Cada segmento e um no do grafo de linhas. Conexoes exigem uma juncao OSM compartilhada; cruzamentos geometricos nao criam atalhos. "
            + "Associacao em EPSG:5880, distancia maxima inclusiva de 500 m e desempate por ID canonico. "
            + "Nos isolados e geometrias paralelas distintas sao preservados. Geometrias que cruzam o limite de MG sao omitidas.\n\n"
            + "## 5. Atributos e Disponibilidade\n\n"
            + "As cinco familias compartilham atributos de via e calendario e o mesmo procedimento de pre-processamento ajustado somente em treino. "
            + "Somente GraphSAGE e GATv2 recebem adjacencia. IDs, coordenadas de identificacao e centralidades nao sao preditores. "
            + "Historicos de 1/4/13/52 semanas exigem cobertura e publicacao anterior a previsao. "
            + (
                "Neste experimento todas as fontes anuais usam cobertura assumida sem publicacao historica comprovada: as contagens defasadas ficam ausentes, com indicadores de falta. A ablacao temporal mede essencialmente calendario, nao historicos observados.\n\n"
                if len(assumed_sources) == len(coverage["sources"])
                else "Ausencia de evidencia gera valores ausentes, nunca zeros historicos inventados.\n\n"
            )
            + "## 6. Orcamento e Selecao em Desenvolvimento\n\n"
            + "Configuracoes e colunas por run: [model-configurations.json](model-configurations.json). "
            + "Metricas e limiares por seed em 2024: [validation.csv](validation.csv). "
            + "O benchmark real usa um candidato por familia/variante e orcamento limitado definido por piloto computacional; "
            + f"nao demonstra otimizacao exaustiva ou convergencia. Todas as observacoes e as {len(bundle.seeds)} seeds declaradas sao mantidas.\n\n"
            + "## 7. Periodos Efetivamente Avaliados\n\n"
            + markdown_table(period_rows)
            + "\nSemanas que cruzam particoes e semanas incompletas ficam fora dos alvos. "
            + "2026 nao e anualizado. A prevalencia e a referencia de AP para uma ordenacao aleatoria.\n\n"
            + "## 8. Benchmark Preditivo de 2025\n\n"
            + markdown_table(metric_summary(main, "full_period"))
            + "\n## 9. Teste Temporal de 2026\n\n"
            + markdown_table(metric_summary(temporal, "full_period"))
            + "\n## 10. Comparacao Sazonal\n\n"
            + "Intersecao das semanas ISO e segmentos dos dois testes definida antes dos scores. "
            + "Reduz a diferenca de calendario, sem controlar alteracoes no trafego, registro ou rede.\n\n"
            + markdown_table(
                metric_summary(main, "matched_season") + metric_summary(temporal, "matched_season")
            )
            + "\n## 11. Ablacoes e Contribuicao da Topologia\n\n"
            + "Diferencas emparelhadas por seed, modelo completo menos variante, sobre a mesma populacao de 2025. "
            + "without_road remove via; without_temporal remove calendario/historico; without_topology remove mensagens de vizinhos e preserva o proprio no. "
            + "Em 2026 sao avaliados apenas os modelos completos, conforme pre-declarado.\n\n"
            + markdown_table(
                [
                    {
                        key: row[key]
                        for key in (
                            "model",
                            "ablation",
                            "metric",
                            "full_mean",
                            "ablation_mean",
                            "paired_delta_mean",
                            "paired_delta_std",
                        )
                    }
                    for row in ablations
                ]
            )
            + "\nDiferencas para a referencia tabular: [deltas.csv](deltas.csv).\n\n"
            + "## 12. Benchmark Computacional\n\n"
            + "Tempos em segundos; memoria em MiB; execucao em CPU com um thread de modelo. "
            + "RSS amostrado a cada 50 ms inclui o processo e seus filhos, dados e bibliotecas; nao e memoria incremental exclusiva do modelo. "
            + "Inferencia de teste inclui verificacao/carregamento de pesos e pre-processamento; inferencia de validacao usa pesos e matrizes ja carregados. "
            + "Carga concorrente da maquina e caches afetam os tempos.\n\n"
            + markdown_table(computational_summary)
            + "\nDetalhes por run: [training-costs.csv](training-costs.csv) e [inference-costs.csv](inference-costs.csv).\n\n"
            + markdown_table(preparation_benchmarks)
            + "\nRecibos de preparacao, piloto e decisao anterior aos testes: [execution-context.json](execution-context.json).\n\n"
            + "## 13. Figuras e Tabelas Completas\n\n"
            + "\n".join(
                f"![{Path(name).stem}]({name})\n"
                for name in figure_names
                if "auprc" in name and name.endswith(".png")
            )
            + "\nMetricas completas e motivos de indefinicao: [metrics.csv](metrics.csv). "
            + "Resultados individuais: [per-run.csv](per-run.csv). "
            + "\n\n"
            + "\n".join(f"- [{name}]({name})" for name in figure_names)
            + "\n\n## 14. Limites da Interpretacao\n\n"
            + "\n".join("- " + item for item in limitations)
            + "\n\n## 15. Rastreabilidade e Reproducao\n\n"
            + f"Dataset: {version.dataset_version_id}\n\nDesenvolvimento: {bundle.development_digest}\n\n"
            + f"Codigo: {bundle.code_hash}\n\nLock: {bundle.environment_lock_hash}\n\n"
            + "Fontes, evidencias, codigo, ambiente, pre-processadores, pesos, predicoes e tabelas sao verificados por hash. "
            + "O pacote e criado depois deste relatorio. A reproducao full reconstroi os dados e executa os pesos arquivados "
            + "em processo isolado, sem rede e sem novo treino, com tolerancia maxima de um ponto percentual. "
            + "A conclusao de reproducao depende do recibo posterior; este relatorio por si so nao declara a reproducao concluida.\n"
        )
        (staging / "report.md").write_text(report_text, encoding="utf-8")
        hashes = {
            f"{prefix}/{path.name}": hash_file(path) for path in staging.iterdir() if path.is_file()
        }
        report = {
            "schema_version": "3.0.0",
            "report_id": identifier,
            "status": "completed",
            "profile": main.profile,
            "coverage_basis": "owner_assumption" if assumed_sources else "recorded_evidence",
            "coverage_assumptions": assumed_sources,
            "bundle_id": bundle.bundle_id,
            "dataset_version_id": version.dataset_version_id,
            "evaluation_ids": [main_evaluation_id, temporal_evaluation_id],
            "periods": periods,
            "tables": {
                "metrics": prefix + "/metrics.csv",
                "per_run": prefix + "/per-run.csv",
                "deltas": prefix + "/deltas.csv",
                "periods": prefix + "/periods.csv",
                "training_costs": prefix + "/training-costs.csv",
                "inference_costs": prefix + "/inference-costs.csv",
                "computational_summary": prefix + "/computational-summary.csv",
                "populations": prefix + "/populations.csv",
                "validation": prefix + "/validation.csv",
                "ablations": prefix + "/ablations.csv",
                "preparation_costs": prefix + "/preparation-costs.csv",
            },
            "figures": [prefix + "/" + name for name in figure_names],
            "limitations": limitations,
            "baseline_reference": bundle.baseline_reference,
            "dataset_manifest": version.manifest_path,
            "dataset_directory": dataset_directory(root, version.dataset_version_id)
            .relative_to(root)
            .as_posix(),
            "manifest": prefix + "/manifest.json",
            "artifact_hashes": hashes,
        }
        write_json(staging / "manifest.json", report)
    return report
