import argparse
import csv
import json
import platform
import ssl
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from road_accidents.config import PipelineError, load_config
from road_accidents.data.prf import inspect_prf, numeric_value, parse_date
from road_accidents.data.schemas import SourceCatalog
from road_accidents.data.storage import atomic_directory, read_json, safe_path, write_json
from road_accidents.provenance import capture_code_state, hash_file
from road_accidents.reporting.tables import markdown_table, write_csv


def source_diagnostics(root: Path, catalog_path: str, output: str) -> dict[str, Any]:
    from road_accidents.pipeline import MemorySampler

    root = root.resolve()
    catalog_file = safe_path(root, catalog_path)
    catalog = SourceCatalog.model_validate(read_json(catalog_file))
    if len(catalog.sources) != 8 or {source.nominal_year for source in catalog.sources} != set(
        range(2019, 2027)
    ):
        raise PipelineError("Diagnostics require the eight selected annual PRF sources.")
    code = capture_code_state(Path(__file__).resolve().parents[3])
    started = time.perf_counter()
    annual: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for source in sorted(catalog.sources, key=lambda item: item.nominal_year or 0):
        if source.source_type != "prf_occurrence_csv" or source.nominal_year is None:
            raise PipelineError("Diagnostics require approved PRF occurrence sources.")
        raw_path = safe_path(root, source.raw_path)
        if hash_file(raw_path) != source.sha256:
            raise PipelineError("Selected PRF snapshot failed integrity verification.")
        memory = MemorySampler()
        memory.start()
        measured_at = time.perf_counter()
        try:
            rows, encoding = inspect_prf(raw_path.read_bytes(), source.nominal_year)
            mg = [row for row in rows if row["uf"].strip().upper() == "MG"]
            dates = [parse_date(row["data_inversa"]) for row in mg]
            known_dates = [value for value in dates if value is not None]
            dates_by_month = Counter(value.strftime("%Y-%m") for value in known_dates)
            coordinates = [
                (numeric_value(row["latitude"]), numeric_value(row["longitude"])) for row in mg
            ]
            missing_coordinates = sum(
                latitude is None or longitude is None for latitude, longitude in coordinates
            )
            zero_coordinates = sum(
                latitude is not None and longitude is not None and (latitude == 0 or longitude == 0)
                for latitude, longitude in coordinates
            )
            out_of_range = sum(
                latitude is not None
                and longitude is not None
                and not (-90 <= latitude <= 90 and -180 <= longitude <= 180)
                for latitude, longitude in coordinates
            )
            road_refs = [numeric_value(row.get("br", "")) for row in mg]
            annual.append(
                {
                    "year": source.nominal_year,
                    "national_rows": len(rows),
                    "mg_rows": len(mg),
                    "other_state_rows": len(rows) - len(mg),
                    "mg_observed_start": min(known_dates).isoformat() if known_dates else None,
                    "mg_observed_end": max(known_dates).isoformat() if known_dates else None,
                    "mg_missing_date": len(dates) - len(known_dates),
                    "mg_missing_coordinates": missing_coordinates,
                    "mg_zero_coordinates": zero_coordinates,
                    "mg_out_of_range_coordinates": out_of_range,
                    "mg_missing_br": sum(value is None for value in road_refs),
                    "raw_bytes": source.byte_count,
                    "encoding": encoding,
                    "coverage_status": source.coverage_status,
                    "inspection_seconds": time.perf_counter() - measured_at,
                }
            )
            monthly.extend(
                {"year": source.nominal_year, "month": month, "mg_rows": count}
                for month, count in sorted(dates_by_month.items())
            )
            identities.append(
                {
                    "year": source.nominal_year,
                    "snapshot_id": source.snapshot_id,
                    "source_url": source.source_url,
                    "raw_path": source.raw_path,
                    "sha256": source.sha256,
                    "source_manifest": source.manifest_path,
                }
            )
            del rows, mg, dates, known_dates, coordinates, road_refs
        finally:
            memory.stop()
        annual[-1]["process_peak_rss_mib"] = memory.peak_bytes / 1024**2
        print(
            f"PRF {source.nominal_year}: inspected {annual[-1]['national_rows']} rows.", flush=True
        )
    destination = safe_path(root, output)
    summary = {
        "schema_version": "3.0.0",
        "status": "completed_source_diagnostics",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "source_quality_not_model_benchmark",
        "profile": catalog.sources[0].profile,
        "catalog": catalog_path,
        "catalog_sha256": hash_file(catalog_file),
        "code_hash": code["code_hash"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "national_rows": sum(row["national_rows"] for row in annual),
        "mg_rows": sum(row["mg_rows"] for row in annual),
        "elapsed_seconds": time.perf_counter() - started,
        "sources": identities,
        "annual": annual,
        "model_training_performed": False,
        "coverage_inferred": False,
        "original_sources_modified": False,
    }
    overview = [
        {
            key: row[key]
            for key in (
                "year",
                "national_rows",
                "mg_rows",
                "mg_observed_start",
                "mg_observed_end",
                "coverage_status",
            )
        }
        for row in annual
    ]
    quality = [
        {
            key: row[key]
            for key in (
                "year",
                "mg_missing_date",
                "mg_missing_coordinates",
                "mg_zero_coordinates",
                "mg_out_of_range_coordinates",
                "mg_missing_br",
            )
        }
        for row in annual
    ]
    measurements = [
        {
            "year": row["year"],
            "inspection_seconds": round(row["inspection_seconds"], 3),
            "rows_per_second": round(row["national_rows"] / row["inspection_seconds"], 1),
            "process_peak_rss_mib": round(row["process_peak_rss_mib"], 1),
        }
        for row in annual
    ]
    report = (
        "# Diagnostico dos Dados Reais PRF\n\n"
        "Este documento mede a inspecao das fontes reais. Nao contem metricas preditivas nem constitui a conclusao cientifica.\n\n"
        "## Populacao de Entrada\n\n"
        + markdown_table(overview)
        + "\nOs intervalos acima descrevem ocorrencias encontradas, nao comprovam completude. Linhas MG ainda precisam de validacao do limite estadual, deduplicacao e associacao a segmentos.\n\n"
        + "## Qualidade Previa a Associacao Espacial\n\n"
        + markdown_table(quality)
        + "\nContagens de problemas sao diagnosticos por campo e podem se sobrepor. Nao foram removidas linhas nem inventados negativos.\n\n"
        + "## Benchmark de Inspecao\n\n"
        + markdown_table(measurements)
        + "\nUma medicao local por arquivo, com cache de sistema nao controlado. O RSS e o pico amostrado do processo e filhos, nao a alocacao exclusiva de cada CSV. O tempo inclui reinspecao de privacidade, parsing e agregacao; exclui download e o hash anterior.\n\n"
        + "## Contagens Mensais MG\n\n"
        + markdown_table(monthly)
        + "\nMeses ausentes nao foram preenchidos com zero. A ausencia nao comprova que nao houve acidentes.\n\n"
        + "## Limites e Proximas Etapas\n\n"
        + "- Obter e verificar a malha federal historica completa de MG, sem substituir pela rede atual.\n"
        + "- Documentar cobertura temporal e disponibilidade, sem usar a ultima ocorrencia como prova.\n"
        + "- Construir o grafo de segmentos, verificar conectividade e snapping de ate 500 metros.\n"
        + "- Ajustar RF, XGBoost, MLP, GraphSAGE e GATv2 apenas com treino/validacao; congelar antes dos testes de 2025/2026.\n"
        + "- Comparar AP, AUROC, F1, precisao, recall, cinco seeds, ablacoes, tempos e memoria; dados ainda ausentes nao recebem resultados substitutos.\n"
    )
    with atomic_directory(root, destination.relative_to(root)) as staging:
        write_csv(staging / "annual.csv", annual)
        write_csv(staging / "monthly-mg.csv", monthly)
        write_csv(staging / "inspection-benchmark.csv", measurements)
        (staging / "report.md").write_text(report, encoding="utf-8")
        summary["artifact_hashes"] = {
            path.name: hash_file(path) for path in staging.iterdir() if path.is_file()
        }
        write_json(staging / "manifest.json", summary)
    return summary


def graph_diagnostics(root: Path, output: str, osm_catalog: str | None = None) -> dict[str, Any]:
    import geopandas as gpd
    import httpx
    import networkx as nx

    from road_accidents.data.osm import (
        OVERPASS_URL,
        extract_mg_boundary,
        ingest_osm,
        load_osm_graph,
    )
    from road_accidents.data.storage import download_bytes, write_geoparquet, write_parquet
    from road_accidents.graph.adjacency import build_adjacency, node_order
    from road_accidents.graph.segments import build_segments
    from road_accidents.pipeline import MemorySampler

    root = root.resolve()
    destination = safe_path(root, output)
    work = root if osm_catalog else safe_path(root, "artifacts/diagnostics/osm-br040")
    query = (
        '[out:xml][timeout:60][date:"2018-12-31T23:59:59Z"];'
        "rel(315173)->.mg;"
        'way["ref"="BR-040"](-22.93,-51.05,-14.23,-39.85)->.road;'
        ".road out body;(way(r.mg); - .road;)->.border;.border out skel;"
        "(.road;.border;);node(w);out skel;.mg out body;"
    )
    started = time.perf_counter()
    memory = MemorySampler()
    memory.start()
    try:
        if osm_catalog:
            catalog = SourceCatalog.model_validate(read_json(safe_path(root, osm_catalog)))
            if len(catalog.sources) != 1 or catalog.sources[0].source_type != "osm_network":
                raise PipelineError("One historical OSM source is required.")
            snapshot = catalog.sources[0]
            if snapshot.media_type != "application/zip":
                raise PipelineError(
                    "Statewide diagnostics require the verified complete tile collection."
                )
            query = "Verified queries archived in collection.json inside the selected source ZIP."
            downloaded = started
        else:
            with httpx.Client(
                timeout=80, verify=ssl.create_default_context(), follow_redirects=False
            ) as client:
                payload = download_bytes(OVERPASS_URL, client=client, data={"data": query})
            downloaded = time.perf_counter()
            snapshot = ingest_osm(
                payload,
                work,
                effective_at="2018-12-31T23:59:59Z",
                source_url=OVERPASS_URL + "?" + urlencode({"data": query}),
            )
        print("Verifying historical network and boundary.", flush=True)
        boundary_document = extract_mg_boundary(snapshot, work)
        boundary = gpd.GeoDataFrame.from_features(boundary_document["features"], crs=4326)
        graph = load_osm_graph(snapshot, work)
        print(f"Building segments from {graph.number_of_edges()} directed edges.", flush=True)
        ingested = time.perf_counter()
        segments = build_segments(graph, snapshot.snapshot_id, boundary)
        segmented = time.perf_counter()
        adjacency = build_adjacency(segments)
        order = node_order(segments)
        line_graph = nx.Graph()
        line_graph.add_nodes_from(order.segment_id)
        line_graph.add_edges_from(
            zip(adjacency.segment_id_low, adjacency.segment_id_high, strict=True)
        )
        identifiers = set(order.segment_id)
        if (
            nx.number_of_selfloops(line_graph)
            or not set(adjacency.segment_id_low).union(adjacency.segment_id_high) <= identifiers
            or not order.node_index.tolist() == list(range(len(segments)))
            or not segments.length_m.gt(0).all()
            or not boundary.geometry.union_all().covers(segments.geometry).all()
        ):
            raise PipelineError("Real diagnostic graph violated its structural invariants.")
        components = sorted(
            (len(part) for part in nx.connected_components(line_graph)), reverse=True
        )
        finished = time.perf_counter()
    finally:
        memory.stop()
    summary = {
        "schema_version": "3.0.0",
        "status": "completed_graph_diagnostics",
        "scope": "historical_MG_federal_network"
        if osm_catalog
        else "historical_BR-040_diagnostic_not_statewide_network",
        "complete_study_network": bool(osm_catalog),
        "osm_catalog": osm_catalog,
        "model_training_performed": False,
        "created_at": datetime.now(UTC).isoformat(),
        "effective_at": snapshot.effective_at.isoformat() if snapshot.effective_at else None,
        "snapshot_id": snapshot.snapshot_id,
        "source_manifest": (work / snapshot.manifest_path).relative_to(root).as_posix(),
        "source_raw": (work / snapshot.raw_path).relative_to(root).as_posix(),
        "query": query,
        "code_hash": capture_code_state(Path(__file__).resolve().parents[3])["code_hash"],
        "osm_nodes": graph.number_of_nodes(),
        "osm_directed_edges": graph.number_of_edges(),
        "physical_segments": len(segments),
        "unique_segment_links": line_graph.number_of_edges(),
        "junction_links": len(adjacency),
        "connected_components": len(components),
        "largest_component_segments": components[0],
        "isolated_segments": len(list(nx.isolates(line_graph))),
        "length_km": float(segments.length_m.sum() / 1000),
        "missing_lanes": int(segments.lanes.isna().sum()),
        "missing_maxspeed": int(segments.maxspeed_kph.isna().sum()),
        "checks": [
            "positive_lengths",
            "mg_polygon",
            "canonical_node_order",
            "no_self_loops",
            "valid_edge_endpoints",
        ],
        "benchmark": {
            "download_seconds": downloaded - started,
            "inspection_loading_seconds": ingested - downloaded,
            "segmentation_seconds": segmented - ingested,
            "adjacency_validation_seconds": finished - segmented,
            "elapsed_seconds": finished - started,
            "sampled_peak_rss_mib": memory.peak_bytes / 1024**2,
        },
    }
    with atomic_directory(root, destination.relative_to(root)) as staging:
        write_geoparquet(staging / "segments.parquet", segments)
        write_parquet(staging / "adjacency.parquet", adjacency)
        write_parquet(staging / "nodes.parquet", order)
        write_json(staging / "mg-boundary.geojson", boundary_document)
        write_json(staging / "components.json", {"component_sizes": components})
        rows = [
            {"measurement": key, "value": value}
            for key, value in summary.items()
            if isinstance(value, (int, float))
        ]
        measurements = [
            {"measurement": key, "value": round(value, 3)}
            for key, value in summary["benchmark"].items()
        ]
        report = (
            (
                "# Grafo Historico Federal de MG\n\nRede OSM em 2018-12-31T23:59:59Z. Colecao completa de celulas historicas, incluindo referencias compostas e participacao em rotas federais. Nao equivale a uma garantia de completude cartografica do OSM.\n\n"
                if osm_catalog
                else "# Diagnostico do Grafo Historico BR-040 em MG\n\nRede OSM real em 2018-12-31T23:59:59Z. Recorte somente ref=BR-040, nao toda a malha federal de MG.\n\n"
            )
            + "## Estrutura\n\n"
            + markdown_table(rows)
            + "\n## Benchmark Local\n\n"
            + markdown_table(measurements)
            + "\nMedicao unica, RSS amostrado do processo e filhos. Download so e incluido quando efetuado nesta execucao; etapas separadas.\n\n"
            + "## Verificacoes e Limites\n\n"
            + "- Comprimentos positivos e segmentos inteiramente dentro do poligono historico de MG.\n"
            + "- Ordem de nos canonica, conexoes com endpoints validos e ausencia de autoarestas no grafo de segmentos.\n"
            + "- Componentes desconectados e atributos ausentes sao reportados, nao corrigidos por conexoes artificiais.\n"
            + "- A consulta preserva nos compartilhados; intersecoes apenas geometricas nao criam vizinhanca.\n"
            + (
                "- Cobertura de celulas verificada sem lacunas; atributos ausentes e vias nao mapeadas em 2018 permanecem limitacoes.\n"
                if osm_catalog
                else "- Referencias compostas, outras rodovias e vias identificadas apenas por relacoes podem faltar neste recorte.\n"
            )
            + "- Nenhum treino, avaliacao preditiva ou conclusao sobre ganho da topologia foi realizado.\n"
        )
        (staging / "report.md").write_text(report, encoding="utf-8")
        summary["artifact_hashes"] = {
            path.name: hash_file(path) for path in staging.iterdir() if path.is_file()
        }
        write_json(staging / "manifest.json", summary)
    return summary


def spatial_diagnostics(
    root: Path, catalog_path: str, graph_manifest_path: str, output: str
) -> dict[str, Any]:
    import geopandas as gpd

    from road_accidents.data.prf import curate_prf
    from road_accidents.data.storage import read_geoparquet, write_geoparquet, write_parquet
    from road_accidents.graph.matching import match_accidents
    from road_accidents.pipeline import MemorySampler

    root = root.resolve()
    catalog = SourceCatalog.model_validate(read_json(safe_path(root, catalog_path)))
    graph_path = safe_path(root, graph_manifest_path)
    graph = read_json(graph_path)
    statewide = graph.get("scope") == "historical_MG_federal_network"
    if not statewide and graph.get("scope") != "historical_BR-040_diagnostic_not_statewide_network":
        raise PipelineError("Spatial diagnostics require a verified historical graph.")
    prefix = "federal" if statewide else "br040"
    for name, digest in graph["artifact_hashes"].items():
        if hash_file(safe_path(graph_path.parent, name)) != digest:
            raise PipelineError("Diagnostic graph artifact failed integrity verification.")
    segments = read_geoparquet(graph_path.parent / "segments.parquet")
    boundary = gpd.read_file(graph_path.parent / "mg-boundary.geojson")
    destination = safe_path(root, output)
    started = time.perf_counter()
    annual = []
    known_ids: set[str] = set()
    memory = MemorySampler()
    memory.start()
    try:
        with atomic_directory(root, destination.relative_to(root)) as staging:
            for source in sorted(catalog.sources, key=lambda item: item.nominal_year or 0):
                measured_at = time.perf_counter()
                records = curate_prf(source, root, boundary)
                mg = records[records.state_code == "MG"]
                valid = mg[mg.quality_status == "valid"]
                repeated = int(valid.accident_id.isin(known_ids).sum())
                known_ids.update(records.accident_id.dropna().astype(str))
                road = (
                    mg
                    if statewide
                    else records[(records.state_code == "MG") & (records.highway_ref == "BR-040")]
                )
                matches = match_accidents(road, segments)
                accepted = matches[matches.match_status == "matched"]
                if not accepted.empty and not accepted.distance_m.le(500).all():
                    raise PipelineError("Accepted diagnostic matches exceed the distance limit.")
                row = {
                    "year": source.nominal_year,
                    "national_rows": len(records),
                    "mg_rows": len(mg),
                    "mg_valid_before_snapping": len(valid),
                    "mg_invalid_before_snapping": len(mg) - len(valid),
                    "mg_valid_ids_seen_in_earlier_source": repeated,
                    f"{prefix}_rows": len(road),
                    f"{prefix}_valid_before_snapping": int((road.quality_status == "valid").sum()),
                    f"{prefix}_matched_within_500m": len(accepted),
                    f"{prefix}_rejected_distance": int(
                        (matches.match_status == "rejected_distance").sum()
                    ),
                    f"{prefix}_no_candidate": int((matches.match_status == "no_candidate").sum()),
                    "matched_median_distance_m": float(accepted.distance_m.median())
                    if len(accepted)
                    else None,
                    "matched_p95_distance_m": float(accepted.distance_m.quantile(0.95))
                    if len(accepted)
                    else None,
                    "elapsed_seconds": time.perf_counter() - measured_at,
                    "snapshot_id": source.snapshot_id,
                }
                if len(matches) != row[f"{prefix}_valid_before_snapping"]:
                    raise PipelineError("Diagnostic spatial matching lost valid source rows.")
                annual.append(row)
                write_geoparquet(staging / f"curated/{source.nominal_year}.parquet", records)
                write_parquet(staging / f"matches/{source.nominal_year}.parquet", matches)
                write_json(
                    staging / f"quality/{source.nominal_year}.json",
                    {
                        "mg_exclusion_reasons": mg.exclusion_reason.value_counts().to_dict(),
                        "national_exclusion_reasons": records.exclusion_reason.value_counts().to_dict(),
                    },
                )
                print(
                    f"PRF {source.nominal_year}: curated {len(records)} rows; {prefix} matches {len(accepted)}/{len(matches)}.",
                    flush=True,
                )
                del records, mg, valid, road, matches, accepted
            memory.stop()
            result = {
                "schema_version": "3.0.0",
                "status": "completed_spatial_diagnostics",
                "scope": "MG_source_curation_and_statewide_federal_snapping"
                if statewide
                else "MG_source_curation_and_partial_BR-040_snapping",
                "complete_study_network": statewide,
                "model_training_performed": False,
                "coverage_inferred": False,
                "created_at": datetime.now(UTC).isoformat(),
                "catalog": catalog_path,
                "catalog_sha256": hash_file(safe_path(root, catalog_path)),
                "graph_manifest": graph_manifest_path,
                "graph_manifest_sha256": hash_file(graph_path),
                "code_hash": capture_code_state(Path(__file__).resolve().parents[3])["code_hash"],
                "elapsed_seconds": time.perf_counter() - started,
                "sampled_peak_rss_mib": memory.peak_bytes / 1024**2,
                "annual": annual,
            }
            write_csv(staging / "spatial-summary.csv", annual)
            overview = [
                {
                    key: row[key]
                    for key in (
                        "year",
                        "mg_rows",
                        "mg_valid_before_snapping",
                        "mg_invalid_before_snapping",
                        f"{prefix}_valid_before_snapping",
                        f"{prefix}_matched_within_500m",
                        f"{prefix}_rejected_distance",
                    )
                }
                for row in annual
            ]
            report = (
                (
                    "# Curadoria MG e Associacao a Malha Federal Historica\n\n"
                    if statewide
                    else "# Curadoria Real MG e Associacao Diagnostica BR-040\n\n"
                )
                + "Todos os registros das oito fontes foram processados pelo curador do projeto. Os arquivos originais permanecem intactos.\n\n"
                + markdown_table(overview)
                + "\n## Escopo e Limites\n\n"
                + "- Validade antes do snapping verifica ID, duplicatas na fonte, data/hora, UF e coordenadas dentro do limite historico de MG.\n"
                + "- IDs repetidos entre fontes sao contados separadamente. O build final aplica a deduplicacao global.\n"
                + (
                    "- O snapping abrange todos os registros validos de MG e a colecao estadual historica, incluindo referencias compostas e participacao em rotas federais.\n"
                    if statewide
                    else "- O snapping aqui abrange somente registros declarados BR-040 e a rede historica com ref exata BR-040. Nao e taxa de cobertura de toda a malha federal.\n"
                )
                + "- Empates usam o menor ID canonico e a distancia aceita e de ate 500 metros em EPSG:5880.\n"
                + (
                    "- Vias nao mapeadas em 2018, limites cartograficos e segmentos que cruzam a fronteira estadual podem causar exclusoes.\n"
                    if statewide
                    else "- Trechos com referencias compostas ou apenas em relacoes podem faltar neste recorte; rejeicoes nao devem ser extrapoladas ao estudo completo.\n"
                )
                + "- Nao foram criados alvos semanais, negativos, estatisticas de desempenho de modelos ou evidencias de disponibilidade historica.\n"
                + "- As tabelas curated preservam a auditoria nacional de cada linha; os arquivos matches registram candidatos, distancias e decisoes espaciais.\n"
            )
            (staging / "report.md").write_text(report, encoding="utf-8")
            result["artifact_hashes"] = {
                path.relative_to(staging).as_posix(): hash_file(path)
                for path in staging.rglob("*")
                if path.is_file()
            }
            write_json(staging / "manifest.json", result)
    finally:
        memory.stop()
    return result


def consolidate_diagnostics(
    root: Path, source_report: str, graph_report: str, spatial_report: str, output: str
) -> dict[str, Any]:
    import geopandas as gpd
    import matplotlib
    import psutil

    from road_accidents.data.storage import read_geoparquet
    from road_accidents.experiments.schemas import FAMILIES, ExperimentConfig, ModelConfig

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root = root.resolve()
    destination = safe_path(root, output)
    documents = {}
    for name, relative in {
        "sources": source_report,
        "graph": graph_report,
        "spatial": spatial_report,
    }.items():
        path = safe_path(root, relative)
        document = read_json(path)
        for artifact, digest in document["artifact_hashes"].items():
            if hash_file(safe_path(path.parent, artifact)) != digest:
                raise PipelineError("Input diagnostic report failed integrity verification.")
        documents[name] = document
    sources, graph, spatial = documents["sources"], documents["graph"], documents["spatial"]
    statewide = graph.get("scope") == "historical_MG_federal_network"
    prefix = "federal" if statewide else "br040"
    label = "Federal_MG" if statewide else "BR040"
    if bool(spatial.get("complete_study_network")) != statewide:
        raise PipelineError("Graph and spatial diagnostic scopes differ.")
    if (
        spatial["graph_manifest_sha256"] != hash_file(safe_path(root, graph_report))
        or spatial["catalog_sha256"] != sources["catalog_sha256"]
    ):
        raise PipelineError("Diagnostic reports do not share the same source versions.")
    with (safe_path(root, source_report).parent / "monthly-mg.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        monthly = list(csv.DictReader(stream))
    configuration = load_config(root / "configs/experiments/full.yaml", ExperimentConfig)
    model_rows = []
    for family in FAMILIES:
        model = load_config(safe_path(root, configuration.models[family]), ModelConfig)
        model_rows.append(
            {
                "model": family,
                "planned_parameters": json.dumps(model.parameters, sort_keys=True),
                "real_runs": 0,
                "status": "pending_real_dataset",
            }
        )
    years = [int(row["year"]) for row in sources["annual"]]
    mg_counts = [int(row["mg_rows"]) for row in sources["annual"]]
    source_rows = [
        {
            "year": row["year"],
            "national_rows": row["national_rows"],
            "MG_rows": row["mg_rows"],
            "observed_start": row["mg_observed_start"],
            "observed_end": row["mg_observed_end"],
            "proven_coverage": row["coverage_status"],
        }
        for row in sources["annual"]
    ]
    spatial_rows = [
        {
            "year": row["year"],
            "MG_valid_before_snapping": row["mg_valid_before_snapping"],
            "MG_invalid": row["mg_invalid_before_snapping"],
            f"{label}_eligible": row[f"{prefix}_valid_before_snapping"],
            f"{label}_matched": row[f"{prefix}_matched_within_500m"],
            f"{label}_match_percent": round(
                100 * row[f"{prefix}_matched_within_500m"] / row[f"{prefix}_valid_before_snapping"],
                2,
            ),
            "median_m": round(row["matched_median_distance_m"], 2),
            "p95_m": round(row["matched_p95_distance_m"], 2),
        }
        for row in spatial["annual"]
    ]
    mg_valid = sum(row["mg_valid_before_snapping"] for row in spatial["annual"])
    matched = sum(row[f"{prefix}_matched_within_500m"] for row in spatial["annual"])
    eligible = sum(row[f"{prefix}_valid_before_snapping"] for row in spatial["annual"])
    seasonal_totals = {
        year: sum(
            int(row["mg_rows"])
            for row in monthly
            if int(row["year"]) == year and int(row["month"][-2:]) <= 7
        )
        for year in (2025, 2026)
    }
    seasonal_change = 100 * (seasonal_totals[2026] / seasonal_totals[2025] - 1)
    graph_rows = [
        {"measurement": key, "value": graph[key]}
        for key in (
            "osm_nodes",
            "osm_directed_edges",
            "physical_segments",
            "unique_segment_links",
            "junction_links",
            "connected_components",
            "largest_component_segments",
            "isolated_segments",
            "length_km",
            "missing_lanes",
            "missing_maxspeed",
        )
    ]
    benchmark_rows = [
        {
            "operation": "eight_PRF_sources_inspection",
            "seconds": round(sources["elapsed_seconds"], 3),
            "sampled_RSS_MiB": round(
                max(row["process_peak_rss_mib"] for row in sources["annual"]), 1
            ),
            "scope": "all_national_records",
        },
        {
            "operation": f"historical_{label}_graph",
            "seconds": round(graph["benchmark"]["elapsed_seconds"], 3),
            "sampled_RSS_MiB": round(graph["benchmark"]["sampled_peak_rss_mib"], 1),
            "scope": "statewide_offline_construction"
            if statewide
            else "includes_download_partial_network",
        },
        {
            "operation": f"MG_curation_and_{label}_snapping",
            "seconds": round(spatial["elapsed_seconds"], 3),
            "sampled_RSS_MiB": round(spatial["sampled_peak_rss_mib"], 1),
            "scope": "eight_sources_with_audit_tables",
        },
    ]
    predictive = [
        {
            "model": family,
            "split": split,
            "runs_completed": 0,
            "average_precision": None,
            "auroc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "status": "not_executed",
            "reason": "PRF_temporal_coverage_pending"
            if statewide
            else "statewide_historical_network_and_coverage_pending",
        }
        for family in FAMILIES
        for split in ("test_2025", "test_2026")
    ]
    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "total_memory_GiB": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    with atomic_directory(root, destination.relative_to(root)) as staging:
        figure, axis = plt.subplots(figsize=(10, 4.5))
        bars = axis.bar(
            [str(year) for year in years], mg_counts, color=["#236B8E"] * 7 + ["#BD5C32"]
        )
        bars[-1].set_hatch("//")
        axis.set_ylabel("Registros UF=MG")
        axis.set_title("Registros observados por ano; 2026 somente janeiro a julho")
        axis.bar_label(bars, fontsize=9)
        axis.set_ylim(0, max(mg_counts) * 1.16)
        figure.tight_layout()
        figure.savefig(staging / "mg-annual.png", dpi=160)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(10, 4.5))
        for year, color in ((2025, "#236B8E"), (2026, "#BD5C32")):
            entries = [
                row for row in monthly if int(row["year"]) == year and int(row["month"][-2:]) <= 7
            ]
            axis.plot(
                [int(row["month"][-2:]) for row in entries],
                [int(row["mg_rows"]) for row in entries],
                marker="o",
                label=str(year),
                color=color,
            )
        axis.set_xlabel("Mes")
        axis.set_ylabel("Registros UF=MG")
        axis.set_title("Meses observados em comum; contagens nao sao taxas de risco")
        axis.legend()
        figure.tight_layout()
        figure.savefig(staging / "mg-common-months.png", dpi=160)
        plt.close(figure)
        graph_directory = safe_path(root, graph_report).parent
        segments = read_geoparquet(graph_directory / "segments.parquet").to_crs(5880)
        boundary = gpd.read_file(graph_directory / "mg-boundary.geojson").to_crs(5880)
        figure, axis = plt.subplots(figsize=(8, 8))
        boundary.boundary.plot(ax=axis, color="#647168", linewidth=0.9)
        segments.plot(ax=axis, color="#BD5C32", linewidth=1.2)
        axis.set_title(
            "Malha federal de MG, OSM 2018"
            if statewide
            else "BR-040 em MG, OSM 2018: recorte parcial por referencia exata"
        )
        axis.set_xlabel("EPSG:5880, metros")
        axis.set_ylabel("EPSG:5880, metros")
        figure.tight_layout()
        figure.savefig(staging / f"historical-{prefix}-map.png", dpi=180)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(10, 4.5))
        bars = axis.bar(
            [str(row["year"]) for row in spatial_rows],
            [row[f"{label}_match_percent"] for row in spatial_rows],
            color="#29836D",
        )
        axis.bar_label(bars, fmt="%.1f%%", fontsize=9)
        axis.set_ylim(0, 110)
        axis.set_yticks(range(0, 101, 20))
        axis.set_ylabel("Associados ate 500 m (%)")
        axis.set_title(
            "Associacao PRF a malha federal historica de MG"
            if statewide
            else "Snapping diagnostico BR-040; rede parcial, nao cobertura estadual"
        )
        figure.tight_layout()
        figure.savefig(staging / f"{prefix}-snapping.png", dpi=160)
        plt.close(figure)
        write_csv(staging / "source-summary.csv", source_rows)
        write_csv(staging / "spatial-summary.csv", spatial_rows)
        write_csv(staging / "processing-benchmarks.csv", benchmark_rows)
        write_csv(staging / "model-benchmarks.csv", predictive)
        write_json(staging / "hardware.json", hardware)
        report = (
            "# Relatorio de Execucao, Dados e Benchmarks\n\n"
            "**Status: parcial. O plano cientifico original foi preservado; benchmarks preditivos e conclusao final ainda nao foram executados.**\n\n"
            "## 1. Resumo Executivo\n\n"
            + f"Foram adquiridas e inspecionadas as oito fontes PRF de 2019 a 2026: **{sources['national_rows']:,} registros nacionais** e **{sources['mg_rows']:,} com UF=MG**. O curador espacial aprovou **{mg_valid:,} registros MG antes do snapping** e rejeitou **{sources['mg_rows'] - mg_valid}** por criterios intrinsecos. Todos os registros permanecem auditaveis em Parquet; os CSVs originais nao foram alterados.\n\n"
            + f"A rede **{label}**, em 2018-12-31T23:59:59Z, produziu **{graph['physical_segments']} segmentos** e **{graph['unique_segment_links']} conexoes unicas**. Foram aceitas **{matched:,} associacoes entre {eligible:,} ocorrencias elegiveis** ({100 * matched / eligible:.2f}%).\n\n"
            + (
                "A colecao estadual cobre todas as celulas planejadas a partir do limite historico de MG. Isso nao garante que toda via real estava mapeada no OSM em 2018. Falta comprovar a cobertura temporal dos CSVs PRF antes de construir alvos semanais.\n\n"
                if statewide
                else "Este recorte parcial por referencia exata nao representa a malha federal completa de MG; ainda faltam a rede estadual e as evidencias temporais.\n\n"
            )
            + "Os cinco modelos estao implementados, mas nao foram ajustados ou avaliados sobre dados reais neste checkpoint. Nao foram geradas metricas substitutas com dados sinteticos.\n\n"
            + "## 2. Objetivo e Protocolo Preservado\n\n"
            + "Investigar se a topologia da malha federal melhora a predicao semanal de ocorrencia de acidente por segmento em MG. A unidade e segmento-semana, e o alvo binario indica ao menos uma ocorrencia. Todos os modelos devem usar a mesma populacao elegivel e os mesmos atributos nao topologicos.\n\n"
            + "| Etapa | Periodo | Funcao |\n| --- | --- | --- |\n| Treino | 2019-2023 | Ajustar modelos e transformacoes |\n| Validacao | 2024 | Selecionar candidatos, referencia e limiares |\n| Teste principal | 2025 | Avaliar protocolo congelado |\n| Teste temporal | 2026 com cobertura comprovada | Aplicar os mesmos modelos, sem novo ajuste |\n\n"
            + "Semanas com sete dias civis, segunda a segunda, em America/Sao_Paulo. Semanas incompletas ou que cruzam particoes nao recebem alvo. Historicos de 1, 4, 13 e 52 semanas requerem cobertura e publicacao anterior a previsao; sem prova, permanecem nulos, nunca zero artificial.\n\n"
            + "## 3. Fontes Reais e Periodos Observados\n\n"
            + markdown_table(source_rows)
            + "\nO inicio e o fim acima sao datas presentes nos arquivos, nao declaracoes de completude. 2026 nao e anualizado. A fonte remota descompactada de 2026 tem o mesmo hash da copia local original fornecida pelo pesquisador.\n\n"
            + "![Registros MG por ano](mg-annual.png)\n\n"
            + "## 4. Analise Descritiva Sazonal\n\n"
            + f"Entre janeiro e julho, o snapshot de 2025 contem **{seasonal_totals[2025]:,}** registros MG e o de 2026 contem **{seasonal_totals[2026]:,}**, uma diferenca de **{seasonal_change:+.2f}%**. E uma comparacao descritiva dos arquivos observados. Nao demonstra mudanca de risco, incidencia por trafego, completude, significancia estatistica ou desempenho preditivo.\n\n"
            + "![Meses comuns MG](mg-common-months.png)\n\n"
            + "Nao ha denominador de exposicao veicular neste diagnostico. Mudancas em registro, revisoes, mobilidade e disponibilidade de dados podem alterar as contagens. Meses ausentes nao foram preenchidos com zero.\n\n"
            + "## 5. Operacoes nos CSVs e Auditoria\n\n"
            + "1. Conferencia dos hashes dos snapshots e reinspecao integral do schema por ocorrencia.\n"
            + "2. Leitura CP1252 com separador ponto e virgula; conversao explicita de datas, decimais e identificadores numericos, inclusive notacao cientifica.\n"
            + "3. Normalizacao de UF e identificadores, deteccao de duplicatas dentro da fonte, validacao de horario e coordenadas.\n"
            + "4. Verificacao espacial pelo poligono de MG obtido da relacao historica OSM 315173.\n"
            + "5. Preservacao da auditoria de todas as linhas, com motivos de invalidez separados da elegibilidade semanal.\n"
            + (
                "6. Associacao de todos os registros MG validos a rede federal estadual, com candidatos, distancias e decisao por registro.\n\n"
                if statewide
                else "6. Snapping diagnostico apenas de registros declarados BR-040, com candidatos, distancias e decisao por registro.\n\n"
            )
            + "Nenhum pre-processador aprendido foi ajustado. Nao houve remocao destrutiva de linhas, alteracao do original ou criacao de negativos fora de cobertura comprovada. Duplicatas entre fontes sao verificadas no build final; o diagnostico registra separadamente IDs ja vistos.\n\n"
            + "## 6. Grafo Historico Real\n\n"
            + markdown_table(graph_rows)
            + f"\n![Rede historica {label}](historical-{prefix}-map.png)\n\n"
            + "A malha dirigida foi simplificada e canonicalizada em segmentos fisicos. Arestas reciprocas da mesma geometria sao fundidas; geometrias paralelas distintas permanecem separadas. Cada segmento se torna um no, e a adjacencia depende de intersecoes OSM compartilhadas. Cruzamentos apenas geometricos nao criam conexoes. Segmentos isolados e circuitos sao preservados.\n\n"
            + "O grafo passou nas verificacoes de comprimentos positivos, geometrias dentro de MG, ordem canonica dos nos, endpoints validos e ausencia de autoarestas no grafo de segmentos. Componentes desconectados nao foram unidos artificialmente.\n\n"
            + (
                "A aquisicao retomavel usa celulas espaciais historicas, projeta atributos viarios na origem e preserva cada resposta original no ZIP. A verificacao rejeita lacunas, sobreposicoes, hashes divergentes e geometrias conflitantes para o mesmo ID. Vias com referencias compostas e membros de rotas federais estao incluidos. O limite estadual e reconstruido da relacao historica, sem usar areas atuais do Overpass.\n\n"
                if statewide
                else "O recorte usa `ref=BR-040`: outras rodovias, referencias compostas e vias identificadas apenas por relacoes podem faltar.\n\n"
            )
            + "Segmentos que cruzam a fronteira estadual sao omitidos pela regra atual. Incompletude cartografica em 2018 e alteracoes viarias posteriores continuam sendo limitacoes. Os numeros de conectividade nao demonstram ganho preditivo de GNNs.\n\n"
            + "## 7. Associacao PRF e Grafo\n\n"
            + markdown_table(spatial_rows)
            + f"\n![Associacao {label}]({prefix}-snapping.png)\n\n"
            + "Distancias em EPSG:5880 e limite inclusivo de 500 metros. Empates usam o menor identificador canonico, sem consultar labels. Mediana e percentil 95 referem-se apenas aos registros aceitos. Rejeicao espacial nao implica coordenada PRF incorreta nem ausencia de rodovia no mundo real.\n\n"
            + "## 8. Benchmarks Medidos de Processamento\n\n"
            + markdown_table(benchmark_rows)
            + "\nCada operacao teve uma execucao medida. Nao sao medias de repeticoes nem intervalos de confianca. O RSS foi amostrado para o processo e filhos; nao representa memoria exclusiva de cada operacao nem GPU. O cache do sistema nao foi controlado. As etapas possuem escopos diferentes e nao constituem comparacao direta de algoritmos preditivos.\n\n"
            + "Ambiente informado no momento da consolidacao:\n\n"
            + markdown_table([{"property": key, "value": value} for key, value in hardware.items()])
            + "\n## 9. Algoritmos e Experimento Planejado\n\n"
            + markdown_table(model_rows)
            + "\nRF e XGBoost usam as bibliotecas originais; MLP usa PyTorch; GraphSAGE e GATv2 usam PyTorch Geometric. O pre-processamento compartilhado e ajustado somente no treino. Apenas as GNNs recebem a adjacencia; IDs aprendidos, centralidades e labels futuros nao integram os atributos comuns. Os parametros acima sao configuracoes planejadas, nao resultado de busca ou selecao executada.\n\n"
            + f"Seeds preservadas: `{configuration.seeds}`. Matriz final prevista: **25 runs completos** e **30 ablacoes GNN**; as ablacoes removem atributos viarios, temporais ou vizinhanca. Os 55 runs selecionados sao avaliados em 2025; 2026 recebe os 25 completos, salvo declaracao previa diferente. Candidatos adicionais podem aumentar o custo de treino.\n\n"
            + "## 10. Benchmarks Preditivos\n\n"
            + "**Nao executados em dados reais.** O arquivo `model-benchmarks.csv` registra dez combinacoes de modelo e periodo com status `not_executed` e metricas vazias. Zero nao foi usado como substituto de metrica ausente.\n\n"
            + "Na execucao completa, a selecao usa AP media na validacao, calculada por `average_precision_score`. O limiar maximiza F1 em 2024, com o maior limiar em empate. AUROC, AP, F1, precisao e recall sao reportados por seed e por periodo; nulos preservam o motivo. A dispersao entre cinco seeds nao sera chamada de intervalo de confianca ou equivalencia estatistica. Tempos e memoria dos modelos so serao comparados quando medidos no mesmo experimento.\n\n"
            + "## 11. Evidencias e Bloqueios Cientificos\n\n"
            + "- As oito fontes PRF foram obtidas e preservadas por hash; nao falta autorizacao de revisao humana ou aceite de vulnerabilidade.\n"
            + (
                "- A aquisicao estadual foi concluida por celulas adaptativas em 2026-09-22, respeitando slots e limites do Overpass. As falhas anteriores e os diagnosticos parciais foram preservados.\n"
                if statewide
                else "- Consultas estaduais ao Overpass retornaram erros; somente o recorte diagnostico esta disponivel nesta versao.\n"
            )
            + "- A cobertura temporal das versoes PRF permanece desconhecida. Nome anual, data do ZIP, aquisicao e ultimo acidente nao sao prova suficiente pelo protocolo vigente.\n"
            + "- O balanco oficial de 07/08/2026 informa 72.542 sinistros em 2025, enquanto o arquivo adquirido tem 72.529 linhas. A diferenca de 13 registros pode refletir versoes ou criterios distintos; nao foi reconciliada e nao prova, sozinha, perda de dados.\n"
            + "- O mesmo balanco fala de janeiro a junho de 2026, mas nao declara a cobertura do ZIP adquirido. Por isso nao foi convertido automaticamente em evidencia de corte de julho.\n\n"
            + "## 12. Conclusao deste Checkpoint\n\n"
            + "O fluxo real de aquisicao PRF, curadoria estadual, construcao de um grafo historico e associacao espacial foi executado e deixou tabelas auditaveis. O processamento realizado cabe neste ambiente, segundo as medicoes apresentadas, mas isso nao antecipa o custo do treino estadual completo.\n\n"
            + "Ainda nao e possivel afirmar que topologia melhora ou piora a predicao. A evidencia temporal vinculada aos snapshots PRF continua pendente. Depois dos insumos completos, seguem build/validate, piloto no desenvolvimento, congelamento dos cinco modelos e ablacoes, testes separados, pacote e reproducao. Nenhum resultado deste diagnostico substitui essas etapas.\n\n"
            + "## 13. Proveniencia e Referencias\n\n"
            + "Os tres manifestos de entrada, o hash do codigo e os hashes das tabelas e figuras estao em `manifest.json`. Os relatorios originais preservam os identificadores das fontes e a consulta OSM exata.\n\n"
            + "- [Catalogo oficial PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf)\n"
            + "- [Dicionario PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dicionario-acidentes)\n"
            + "- [Balanco oficial PRF de 07/08/2026](https://www.gov.br/prf/pt-br/noticias/prf-acende-sinal-de-alerta-a-sociedade-mortes-e-atropelamentos-crescem-em-rodovias-federais)\n"
            + "- [Overpass e instancias publicas](https://wiki.openstreetmap.org/wiki/Overpass_API)\n"
        )
        (staging / "report.md").write_text(report, encoding="utf-8")
        manifest = {
            "schema_version": "3.0.0",
            "status": "completed_partial_research_report",
            "scientific_conclusion_available": False,
            "real_model_benchmarks_executed": False,
            "complete_study_network": statewide,
            "created_at": datetime.now(UTC).isoformat(),
            "code_hash": capture_code_state(Path(__file__).resolve().parents[3])["code_hash"],
            "inputs": {
                relative: hash_file(safe_path(root, relative))
                for relative in (source_report, graph_report, spatial_report)
            },
            "summary": {
                "national_rows": sources["national_rows"],
                "mg_rows": sources["mg_rows"],
                "mg_valid_before_snapping": mg_valid,
                f"{prefix}_eligible": eligible,
                f"{prefix}_matched": matched,
                "graph_segments": graph["physical_segments"],
                "graph_unique_links": graph["unique_segment_links"],
            },
            "artifact_hashes": {
                path.name: hash_file(path) for path in staging.iterdir() if path.is_file()
            },
        }
        write_json(staging / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect approved real PRF sources without fitting models."
    )
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    parser.add_argument("--prf-manifest")
    parser.add_argument("--graph-manifest")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--osm-manifest")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--source-report")
    parser.add_argument("--spatial-report")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    keys: tuple[str, ...]
    if arguments.graph and arguments.summary:
        parser.error("--graph and --summary are mutually exclusive")
    if arguments.graph and arguments.graph_manifest:
        parser.error("--graph and --graph-manifest are mutually exclusive")
    if arguments.summary:
        if not all((arguments.source_report, arguments.graph_manifest, arguments.spatial_report)):
            parser.error(
                "--summary requires --source-report, --graph-manifest and --spatial-report"
            )
        result = consolidate_diagnostics(
            arguments.work_dir,
            arguments.source_report,
            arguments.graph_manifest,
            arguments.spatial_report,
            arguments.output,
        )
        keys = ("status", "scientific_conclusion_available", "summary")
    elif arguments.graph:
        result = graph_diagnostics(arguments.work_dir, arguments.output, arguments.osm_manifest)
        keys = ("status", "physical_segments", "unique_segment_links", "benchmark")
    elif arguments.graph_manifest:
        if not arguments.prf_manifest:
            parser.error("--prf-manifest is required for spatial diagnostics")
        result = spatial_diagnostics(
            arguments.work_dir, arguments.prf_manifest, arguments.graph_manifest, arguments.output
        )
        keys = ("status", "elapsed_seconds", "sampled_peak_rss_mib")
    else:
        if not arguments.prf_manifest:
            parser.error("--prf-manifest is required for source diagnostics")
        result = source_diagnostics(arguments.work_dir, arguments.prf_manifest, arguments.output)
        keys = ("status", "national_rows", "mg_rows", "elapsed_seconds")
    print(json.dumps({key: result[key] for key in keys}))


if __name__ == "__main__":
    main()
