import math
import time
from collections import Counter

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from src.config import SimConfig
from src.algorithms.scheduling import (
    schedule_ai_driven,
    schedule_by_voronoi_threshold,
    schedule_greedy_coverage,
    schedule_random_greedy_coverage,
    schedule_random_same_off,
)
from src.ml.dataset import compute_run_features
from src.ml.model import load_metric_model, load_model, predict_run_metrics
from src.simulation.fault_tolerance import simulate_failures_and_recovery
from src.simulation.geometry import bounded_voronoi_cells, field_polygon, random_points
from src.simulation.metrics import compute_coverage_ratio, energy_savings, lifetime_estimate


SENSOR_PROFILES = {
    "homogeneous": {"radius_mult": 1.00, "active_energy_mult": 1.00, "sleep_ratio": 0.05, "coverage_target": 0.990, "failure_mult": 1.00},
    "heterogeneous": {"radius_mult": 1.00, "active_energy_mult": 1.00, "sleep_ratio": 0.05, "coverage_target": 0.990, "failure_mult": 1.00},
    "temperature": {"radius_mult": 0.82, "active_energy_mult": 0.85, "sleep_ratio": 0.04, "coverage_target": 0.970, "failure_mult": 0.95},
    "humidity": {"radius_mult": 0.96, "active_energy_mult": 1.00, "sleep_ratio": 0.05, "coverage_target": 0.985, "failure_mult": 1.00},
    "motion": {"radius_mult": 1.22, "active_energy_mult": 1.28, "sleep_ratio": 0.07, "coverage_target": 0.995, "failure_mult": 1.10},
    "mixed": {"radius_mult": 1.00, "active_energy_mult": 1.00, "sleep_ratio": 0.05, "coverage_target": 0.985, "failure_mult": 1.00},
}


SENSOR_EXPLANATIONS = {
    "homogeneous": "All nodes use the same radius and energy cost.",
    "heterogeneous": "Each node gets a random radius and energy draw to emulate uneven hardware.",
    "temperature": "Shorter range, lower power draw, slightly lower coverage target.",
    "humidity": "Balanced range and power, close to the default network behaviour.",
    "motion": "Longest range and highest power draw, so it protects coverage but spends more energy.",
    "mixed": "Blend of temperature, humidity, and motion nodes inside the same deployment.",
}


def _serialize_polygon(poly: Polygon | None):
    if poly is None or poly.is_empty:
        return None
    return [[float(x), float(y)] for x, y in list(poly.exterior.coords)]


def _serialize_active_cells(points: np.ndarray, boundary: Polygon, active_mask: np.ndarray) -> list[dict]:
    active_points = points[active_mask]
    active_idx = np.where(active_mask)[0]
    if len(active_points) == 0:
        return []
    cells = bounded_voronoi_cells(active_points, boundary)
    payload = []
    for node_idx, cell in zip(active_idx.tolist(), cells, strict=False):
        coords = _serialize_polygon(cell)
        if coords:
            payload.append({"node_index": int(node_idx), "coords": coords})
    return payload


def _field_shape(width: float, height: float) -> str:
    ratio = width / max(height, 1e-9)
    if 0.95 <= ratio <= 1.05:
        return "square"
    return "wide" if ratio > 1 else "tall"


def _effective_target_coverage(cfg: SimConfig, sensor_type: str) -> float:
    profile = SENSOR_PROFILES.get(sensor_type, SENSOR_PROFILES["homogeneous"])
    return min(0.999, max(0.90, float(cfg.target_coverage) * profile["coverage_target"] / SENSOR_PROFILES["homogeneous"]["coverage_target"]))


def _build_sensor_arrays(cfg: SimConfig, rng: np.random.Generator):
    sensor_type = cfg.sensor_type
    if sensor_type == "heterogeneous":
        radius_mults = rng.uniform(0.70, 1.35, size=cfg.n_nodes)
        energy_mults = rng.uniform(0.80, 1.35, size=cfg.n_nodes)
        labels = ["heterogeneous"] * cfg.n_nodes
    elif sensor_type == "mixed":
        labels = rng.choice(["temperature", "humidity", "motion"], size=cfg.n_nodes, p=[0.35, 0.35, 0.30]).tolist()
        radius_mults = np.asarray([SENSOR_PROFILES[label]["radius_mult"] for label in labels], dtype=float)
        energy_mults = np.asarray([SENSOR_PROFILES[label]["active_energy_mult"] for label in labels], dtype=float)
    else:
        profile = SENSOR_PROFILES.get(sensor_type, SENSOR_PROFILES["homogeneous"])
        radius_mults = np.full(cfg.n_nodes, float(profile["radius_mult"]), dtype=float)
        energy_mults = np.full(cfg.n_nodes, float(profile["active_energy_mult"]), dtype=float)
        labels = [sensor_type] * cfg.n_nodes

    radii = np.asarray(cfg.sensing_radius * radius_mults, dtype=float)
    active_costs = np.asarray(cfg.energy_active_cost * energy_mults, dtype=float)
    sleep_costs = np.asarray(active_costs * np.where(radius_mults > 1.1, 0.08, 0.05), dtype=float)
    return radii, labels, active_costs, sleep_costs


def _sensor_summary(labels: list[str], radii: np.ndarray, active_costs: np.ndarray) -> dict:
    counts = Counter(labels)
    summary = {}
    labels_arr = np.asarray(labels)
    for label, count in sorted(counts.items()):
        mask = labels_arr == label
        summary[label] = {
            "count": int(count),
            "avg_radius": float(np.mean(radii[mask])) if count else 0.0,
            "avg_active_energy": float(np.mean(active_costs[mask])) if count else 0.0,
            "explanation": SENSOR_EXPLANATIONS.get(label, "Node-specific profile."),
        }
    return summary


def _recovery_stats(fault_logs: list[dict] | None, target_coverage: float) -> dict:
    if not fault_logs:
        return {
            "total_failures": 0,
            "total_recoveries": 0,
            "recovery_success_rate_pct": 0.0,
            "rounds_with_failures": 0,
            "min_round_coverage": None,
            "avg_round_energy_saved_pct": None,
        }
    total_failures = sum(int(row.get("failed_count", 0)) for row in fault_logs)
    total_recoveries = sum(int(row.get("activated_count", 0)) for row in fault_logs)
    rounds_with_failures = sum(1 for row in fault_logs if row.get("failed_count", 0) > 0)
    rounds_recovered = sum(1 for row in fault_logs if row.get("failed_count", 0) > 0 and row.get("coverage", 0.0) + 1e-9 >= target_coverage)
    success_pct = 0.0 if rounds_with_failures == 0 else 100.0 * rounds_recovered / rounds_with_failures
    return {
        "total_failures": int(total_failures),
        "total_recoveries": int(total_recoveries),
        "recovery_success_rate_pct": float(success_pct),
        "rounds_with_failures": int(rounds_with_failures),
        "min_round_coverage": float(min(row.get("coverage", 0.0) for row in fault_logs)),
        "avg_round_energy_saved_pct": float(np.mean([row.get("energy_saved_pct", 0.0) for row in fault_logs])) if fault_logs else None,
    }


def _base_result(cfg: SimConfig, radii: np.ndarray, sensor_labels: list[str], active_costs: np.ndarray, sleep_costs: np.ndarray, coverage_all: float) -> dict:
    sensor_profile = SENSOR_PROFILES.get(cfg.sensor_type, SENSOR_PROFILES["homogeneous"])
    return {
        "n_nodes": int(cfg.n_nodes),
        "width": float(cfg.width),
        "height": float(cfg.height),
        "field_area": float(cfg.width * cfg.height),
        "field_shape": _field_shape(cfg.width, cfg.height),
        "density": float(cfg.n_nodes / (cfg.width * cfg.height)),
        "sensing_radius": float(cfg.sensing_radius),
        "sensing_radii": radii.tolist(),
        "sensor_type": cfg.sensor_type,
        "sensor_labels": sensor_labels,
        "sensor_summary": _sensor_summary(sensor_labels, radii, active_costs),
        "sensor_explanation": SENSOR_EXPLANATIONS.get(cfg.sensor_type, SENSOR_EXPLANATIONS["homogeneous"]),
        "threshold_coeff": float(cfg.threshold_coeff),
        "coverage_all": float(coverage_all),
        "coverage_loss_initial_pct": float(max(0.0, 100.0 * (1.0 - coverage_all))),
        "target_coverage": float(_effective_target_coverage(cfg, cfg.sensor_type)),
        "active_energy_costs": active_costs.tolist(),
        "sleep_energy_costs": sleep_costs.tolist(),
        "map_mode": bool(cfg.map_mode),
        "map_bounds": cfg.map_bounds,
        "sensor_model_basis": {
            "radius_multiplier": float(sensor_profile["radius_mult"]),
            "active_energy_multiplier": float(sensor_profile["active_energy_mult"]),
            "sleep_ratio": float(sensor_profile["sleep_ratio"]),
        },
    }


def _snapshot(active_nodes: int, backup_nodes: int, failed_nodes: int, coverage: float, energy_saved_pct: float, uncovered_area_pct: float, runtime_ms: float) -> dict:
    return {
        "active_nodes": int(active_nodes),
        "backup_nodes": int(backup_nodes),
        "failed_nodes": int(failed_nodes),
        "coverage": float(coverage),
        "energy_saved_pct": float(energy_saved_pct),
        "uncovered_area_pct": float(uncovered_area_pct),
        "runtime_ms": float(runtime_ms),
    }


def _evaluate_algorithm_run(
    *,
    name: str,
    points: np.ndarray,
    boundary: Polygon,
    radii: np.ndarray,
    sensor_labels: list[str],
    active_costs: np.ndarray,
    sleep_costs: np.ndarray,
    initial_active_mask: np.ndarray,
    cfg: SimConfig,
    algorithm_runtime_ms: float,
    rng_seed: int,
):
    run_started = time.perf_counter()
    initial_active_mask = np.asarray(initial_active_mask, dtype=bool)
    initial_backups = np.where(~initial_active_mask)[0].astype(int).tolist()
    cov_all = compute_coverage_ratio(points, radii, boundary)
    cov_sched = compute_coverage_ratio(points[initial_active_mask], radii[initial_active_mask], boundary)
    energy = energy_savings(initial_active_mask, active_costs, sleep_costs, rounds=1)
    lifetime = lifetime_estimate(initial_active_mask, active_costs, sleep_costs, cfg.battery_budget_per_node)

    result = {
        "algo": name,
        **_base_result(cfg, radii, sensor_labels, active_costs, sleep_costs, cov_all),
        "coverage_scheduled": float(cov_sched),
        "uncovered_area_pct": float(max(0.0, 100.0 * (1.0 - cov_sched))),
        "n_backups": int(len(initial_backups)),
        **energy,
        **lifetime,
        "algorithm_runtime_ms": float(algorithm_runtime_ms),
    }
    result["initial_snapshot"] = _snapshot(
        active_nodes=int(np.sum(initial_active_mask)),
        backup_nodes=int(len(initial_backups)),
        failed_nodes=0,
        coverage=float(cov_sched),
        energy_saved_pct=float(result["energy_saved_pct"]),
        uncovered_area_pct=float(result["uncovered_area_pct"]),
        runtime_ms=float(algorithm_runtime_ms),
    )

    final_active_mask = initial_active_mask.copy()
    final_backups = list(initial_backups)
    unavailable_mask = np.zeros(len(points), dtype=bool)
    fault_logs = None
    effective_target = _effective_target_coverage(cfg, cfg.sensor_type)
    if cfg.enable_fault_tolerance:
        target = min(float(effective_target), float(cov_sched)) if cov_sched > 0 else float(effective_target)
        final_active_mask, final_backups, unavailable_mask, fault_logs = simulate_failures_and_recovery(
            points,
            boundary,
            radii,
            initial_active_mask,
            initial_backups,
            cfg.n_rounds,
            cfg.failure_prob_per_round,
            cfg.failure_model,
            cfg.recovery_model,
            np.random.default_rng(rng_seed),
            target,
            active_costs=active_costs,
            sleep_costs=sleep_costs,
        )
        result.update(_recovery_stats(fault_logs, target))
    else:
        result.update(_recovery_stats(None, effective_target))

    result["round_summary"] = {
        "initial_active_nodes": int(result["n_on"]),
        "initial_backup_nodes": int(result["n_off"]),
        "final_active_nodes": int(fault_logs[-1]["n_active"]) if fault_logs else int(result["n_on"]),
        "final_backup_available": int(fault_logs[-1]["n_backup_available"]) if fault_logs else int(result["n_off"]),
        "final_unavailable_nodes": int(fault_logs[-1]["n_unavailable"]) if fault_logs else 0,
        "average_round_coverage": float(np.mean([row["coverage"] for row in fault_logs])) if fault_logs else float(cov_sched),
        "average_round_energy_saved_pct": float(np.mean([row["energy_saved_pct"] for row in fault_logs])) if fault_logs else float(result["energy_saved_pct"]),
    }
    final_coverage = float(fault_logs[-1]["coverage"]) if fault_logs else float(cov_sched)
    final_energy_saved = float(fault_logs[-1]["energy_saved_pct"]) if fault_logs else float(result["energy_saved_pct"])
    final_failed = int(fault_logs[-1]["n_unavailable"]) if fault_logs else 0
    final_active = int(fault_logs[-1]["n_active"]) if fault_logs else int(result["n_on"])
    result["final_snapshot"] = _snapshot(
        active_nodes=final_active,
        backup_nodes=int(len(final_backups)),
        failed_nodes=final_failed,
        coverage=final_coverage,
        energy_saved_pct=final_energy_saved,
        uncovered_area_pct=float(max(0.0, 100.0 * (1.0 - final_coverage))),
        runtime_ms=0.0,
    )
    result["round_plot_note"] = "Initial cards show pre-failure scheduling. Round plots show the network after simulated failures and recoveries across rounds."
    result["selected_area_note"] = "Field size follows the configured width and height inputs for this mid-review build."
    _attach_metric_predictions(result, cfg)
    result["total_runtime_ms"] = float(1000.0 * (time.perf_counter() - run_started) + algorithm_runtime_ms)
    result["final_snapshot"]["runtime_ms"] = float(result["total_runtime_ms"])

    initial_payload = serialize_run_payload(points, boundary, initial_active_mask, initial_backups, np.zeros(len(points), dtype=bool), result, None)
    final_payload = serialize_run_payload(points, boundary, final_active_mask, final_backups, unavailable_mask, result, fault_logs)

    return {
        "initial": initial_payload,
        "final": final_payload,
        "metrics": result,
        "fault_logs": fault_logs,
    }


def _attach_metric_predictions(result: dict, cfg: SimConfig):
    metric_model = load_metric_model()
    if metric_model is None:
        result["ml_predicted_metrics"] = None
        return
    sensor_code_map = {"homogeneous": 0.0, "temperature": 1.0, "humidity": 2.0, "motion": 3.0, "heterogeneous": 4.0, "mixed": 5.0}
    features = compute_run_features(
        cfg.n_nodes,
        cfg.width,
        cfg.height,
        cfg.sensing_radius,
        cfg.threshold_coeff,
        sensor_code_map.get(cfg.sensor_type, 0.0),
        cfg.failure_prob_per_round,
    ).reshape(1, -1)
    pred = predict_run_metrics(metric_model, features)[0]
    result["ml_predicted_metrics"] = {
        "coverage_scheduled": float(pred[0]),
        "energy_saved_pct": float(pred[1]),
        "estimated_network_lifetime_rounds": float(pred[2]),
    }


def run_voronoi(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    boundary = field_polygon(cfg.width, cfg.height)
    points = random_points(cfg.n_nodes, cfg.width, cfg.height, rng)
    radii, sensor_labels, active_costs, sleep_costs = _build_sensor_arrays(cfg, rng)
    effective_target = _effective_target_coverage(cfg, cfg.sensor_type)

    algo_started = time.perf_counter()
    if cfg.enable_ai:
        model = load_model()
        if model is not None:
            active_mask, backups = schedule_ai_driven(points, boundary, radii, model, target_coverage=effective_target)
            algo_name = "AI-Scheduling"
        else:
            active_mask, backups = schedule_by_voronoi_threshold(points, boundary, radii, cfg.threshold_coeff)
            algo_name = "Voronoi-Threshold"
    else:
        active_mask, backups = schedule_by_voronoi_threshold(points, boundary, radii, cfg.threshold_coeff)
        algo_name = "Voronoi-Threshold"
    algorithm_runtime_ms = 1000.0 * (time.perf_counter() - algo_started)

    evaluated = _evaluate_algorithm_run(
        name=algo_name,
        points=points,
        boundary=boundary,
        radii=radii,
        sensor_labels=sensor_labels,
        active_costs=active_costs,
        sleep_costs=sleep_costs,
        initial_active_mask=active_mask,
        cfg=cfg,
        algorithm_runtime_ms=algorithm_runtime_ms,
        rng_seed=cfg.seed + 77,
    )
    evaluated["metrics"]["control_effect_summary"] = {
        "threshold_coeff": "Higher threshold turns more small-area nodes into backups.",
        "failure_model": "Only affects round-by-round fault plots and recovery charts.",
        "recovery_model": "Greedy restores coverage best; nearest backup restores locally with less switching.",
        "sensor_type": SENSOR_EXPLANATIONS.get(cfg.sensor_type, SENSOR_EXPLANATIONS["homogeneous"]),
        "enable_ai": "AI mode uses a local RandomForest model trained on stored simulation datasets.",
    }

    initial_payload = evaluated["initial"]
    final_payload = evaluated["final"]
    metrics = evaluated["metrics"]
    fault_logs = evaluated["fault_logs"]
    return (
        points,
        boundary,
        np.asarray(initial_payload["active_mask"], dtype=bool),
        [int(x) for x in initial_payload["backups"]],
        np.asarray(final_payload["unavailable_mask"], dtype=bool),
        metrics,
        fault_logs,
    )


def _algo_result(name: str, points: np.ndarray, boundary, radii: np.ndarray, sensor_labels: list[str], active_costs: np.ndarray, sleep_costs: np.ndarray, active_mask: np.ndarray, cfg: SimConfig, runtime_ms: float) -> tuple[dict, list[int]]:
    cov = compute_coverage_ratio(points[active_mask], radii[active_mask], boundary)
    energy = energy_savings(active_mask, active_costs, sleep_costs, rounds=1)
    lifetime = lifetime_estimate(active_mask, active_costs, sleep_costs, cfg.battery_budget_per_node)
    backups = np.where(~active_mask)[0].astype(int).tolist()
    return {
        "algo": name,
        **_base_result(cfg, radii, sensor_labels, active_costs, sleep_costs, compute_coverage_ratio(points, radii, boundary)),
        "coverage_scheduled": float(cov),
        "uncovered_area_pct": float(max(0.0, 100.0 * (1.0 - cov))),
        "n_backups": int(len(backups)),
        **energy,
        **lifetime,
        "algorithm_runtime_ms": float(runtime_ms),
    }, backups


def run_compare(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    boundary = field_polygon(cfg.width, cfg.height)
    points = random_points(cfg.n_nodes, cfg.width, cfg.height, rng)
    radii, sensor_labels, active_costs, sleep_costs = _build_sensor_arrays(cfg, rng)
    effective_target = _effective_target_coverage(cfg, cfg.sensor_type)

    t0 = time.perf_counter()
    active_v, _backups_v = schedule_by_voronoi_threshold(points, boundary, radii, cfg.threshold_coeff)
    voronoi_eval = _evaluate_algorithm_run(
        name="Voronoi-Threshold",
        points=points,
        boundary=boundary,
        radii=radii,
        sensor_labels=sensor_labels,
        active_costs=active_costs,
        sleep_costs=sleep_costs,
        initial_active_mask=active_v,
        cfg=cfg,
        algorithm_runtime_ms=1000.0 * (time.perf_counter() - t0),
        rng_seed=cfg.seed + 1001,
    )

    target = float(voronoi_eval["metrics"]["initial_snapshot"]["coverage"])
    rng_random_same = np.random.default_rng(cfg.seed + 2001)
    t0 = time.perf_counter()
    active_r1, _backups_r1 = schedule_random_same_off(points, boundary, radii, int(voronoi_eval["metrics"]["initial_snapshot"]["backup_nodes"]), rng_random_same)
    random_same_eval = _evaluate_algorithm_run(
        name="Random-Same-OFF",
        points=points,
        boundary=boundary,
        radii=radii,
        sensor_labels=sensor_labels,
        active_costs=active_costs,
        sleep_costs=sleep_costs,
        initial_active_mask=active_r1,
        cfg=cfg,
        algorithm_runtime_ms=1000.0 * (time.perf_counter() - t0),
        rng_seed=cfg.seed + 2002,
    )

    rng_random_greedy = np.random.default_rng(cfg.seed + 3001)
    t0 = time.perf_counter()
    active_r2, _backups_r2 = schedule_random_greedy_coverage(points, boundary, radii, target, rng_random_greedy)
    random_greedy_eval = _evaluate_algorithm_run(
        name="Random-Greedy",
        points=points,
        boundary=boundary,
        radii=radii,
        sensor_labels=sensor_labels,
        active_costs=active_costs,
        sleep_costs=sleep_costs,
        initial_active_mask=active_r2,
        cfg=cfg,
        algorithm_runtime_ms=1000.0 * (time.perf_counter() - t0),
        rng_seed=cfg.seed + 3002,
    )

    t0 = time.perf_counter()
    active_gc, _backups_gc = schedule_greedy_coverage(points, boundary, radii, target)
    greedy_eval = _evaluate_algorithm_run(
        name="Greedy-Coverage",
        points=points,
        boundary=boundary,
        radii=radii,
        sensor_labels=sensor_labels,
        active_costs=active_costs,
        sleep_costs=sleep_costs,
        initial_active_mask=active_gc,
        cfg=cfg,
        algorithm_runtime_ms=1000.0 * (time.perf_counter() - t0),
        rng_seed=cfg.seed + 4002,
    )

    t0 = time.perf_counter()
    model = load_model()
    if model is not None:
        active_ai, _backups_ai = schedule_ai_driven(points, boundary, radii, model, target_coverage=effective_target)
        ai_name = "AI-Scheduling"
    else:
        active_ai, _backups_ai = schedule_by_voronoi_threshold(points, boundary, radii, cfg.threshold_coeff)
        ai_name = "AI-Fallback-Voronoi"
    ai_eval = _evaluate_algorithm_run(
        name=ai_name,
        points=points,
        boundary=boundary,
        radii=radii,
        sensor_labels=sensor_labels,
        active_costs=active_costs,
        sleep_costs=sleep_costs,
        initial_active_mask=active_ai,
        cfg=cfg,
        algorithm_runtime_ms=1000.0 * (time.perf_counter() - t0),
        rng_seed=cfg.seed + 5002,
    )

    return {
        "scenario": {
            "fairness_note": "All algorithms use the same generated field, sensor assumptions, and configured recovery settings.",
            "width": float(cfg.width),
            "height": float(cfg.height),
            "n_nodes": int(cfg.n_nodes),
            "seed": int(cfg.seed),
        },
        "initial": {
            "voronoi": voronoi_eval["initial"],
            "random_same_off": random_same_eval["initial"],
            "random_greedy_cov": random_greedy_eval["initial"],
            "greedy_cov": greedy_eval["initial"],
            "ai_based": ai_eval["initial"],
        },
        "final": {
            "voronoi": voronoi_eval["final"],
            "random_same_off": random_same_eval["final"],
            "random_greedy_cov": random_greedy_eval["final"],
            "greedy_cov": greedy_eval["final"],
            "ai_based": ai_eval["final"],
        },
    }


def experiment_density(cfg: SimConfig, area_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0)):
    rows = []
    for i, k in enumerate(area_multipliers):
        width = float(cfg.width * math.sqrt(float(k)))
        height = float(cfg.height * math.sqrt(float(k)))
        local = SimConfig(**dict(cfg.__dict__))
        local.width = width
        local.height = height
        local.seed = cfg.seed + i * 13
        _, _, _, backups, _unavailable, result, _ = run_voronoi(local)
        rows.append({
            "field_area": float(width * height),
            "field_width": float(width),
            "field_height": float(height),
            "density": float(result["density"]),
            "backup_nodes": int(len(backups)),
            "coverage": float(result["coverage_scheduled"]),
            "energy_saved_pct": float(result["energy_saved_pct"]),
            "estimated_network_lifetime_rounds": float(result["estimated_network_lifetime_rounds"]),
            "uncovered_area_pct": float(result["uncovered_area_pct"]),
        })
    return pd.DataFrame(rows)


def serialize_run_payload(points, boundary, active_mask, backups, unavailable_mask, metrics, fault_logs):
    return {
        "points": points.tolist(),
        "boundary": _serialize_polygon(boundary),
        "active_mask": active_mask.tolist(),
        "backups": [int(x) for x in backups],
        "unavailable_mask": unavailable_mask.tolist(),
        "metrics": metrics,
        "fault_logs": fault_logs,
        "sensing_radii": metrics.get("sensing_radii"),
        "sensor_labels": metrics.get("sensor_labels"),
        "voronoi_cells": _serialize_active_cells(points, boundary, np.asarray(active_mask, dtype=bool)),
    }
