import numpy as np
from shapely.geometry import Polygon

from src.simulation.geometry import sensing_coverage_union


def _normalize_costs(cost, n: int) -> np.ndarray:
    if np.isscalar(cost):
        return np.full(n, float(cost), dtype=float)
    arr = np.asarray(cost, dtype=float)
    if arr.shape != (n,):
        raise ValueError("energy cost array must match number of nodes")
    return arr


def compute_coverage_ratio(points: np.ndarray, sensing_radius, boundary: Polygon) -> float:
    if len(points) == 0 or boundary.area <= 0:
        return 0.0
    covered = sensing_coverage_union(points, sensing_radius, boundary)
    return float(covered.area / boundary.area)


def energy_savings(active_mask: np.ndarray, e_active, e_sleep, rounds: int = 1, unavailable_mask: np.ndarray | None = None) -> dict:
    """
    Computes energy savings comparing the scheduled subset to a baseline where all available nodes are active.
    This baseline model aligns with WSN lifetime maximization literature (e.g., Target Coverage Problem optimizations)
    where Savings = (Baseline - Scheduled) / Baseline.
    """
    n = len(active_mask)
    active_costs = _normalize_costs(e_active, n)
    sleep_costs = _normalize_costs(e_sleep, n)
    unavailable = np.zeros(n, dtype=bool) if unavailable_mask is None else np.asarray(unavailable_mask, dtype=bool)
    if unavailable.shape != (n,):
        raise ValueError("unavailable_mask must match number of nodes")

    active = np.asarray(active_mask, dtype=bool) & ~unavailable
    sleeping = (~active) & ~unavailable

    n_on = int(active.sum())
    n_off = int(sleeping.sum())
    n_unavailable = int(unavailable.sum())

    baseline = float(np.sum(active_costs) * rounds)
    scheduled = float((np.sum(active_costs[active]) + np.sum(sleep_costs[sleeping])) * rounds)
    saved = baseline - scheduled
    pct = 0.0 if baseline == 0 else 100.0 * saved / baseline
    return {
        "n_on": n_on,
        "n_off": n_off,
        "n_unavailable": n_unavailable,
        "baseline_energy": float(baseline),
        "scheduled_energy": float(scheduled),
        "energy_saved": float(saved),
        "energy_saved_pct": float(pct),
    }


def lifetime_estimate(active_mask: np.ndarray, e_active, e_sleep, battery_budget_per_node: float, unavailable_mask: np.ndarray | None = None) -> dict:
    """
    Calculates expected network lifetime in rounds based on pure energy consumption.
    The methodology complies with IEEE WSN evaluation frameworks, projecting:
    Lifetime (Rounds) = Total Network Battery / Energy Consumed Per Round.
    """
    n = max(len(active_mask), 1)
    active_costs = _normalize_costs(e_active, len(active_mask)) if len(active_mask) else np.array([], dtype=float)
    sleep_costs = _normalize_costs(e_sleep, len(active_mask)) if len(active_mask) else np.array([], dtype=float)
    unavailable = np.zeros(len(active_mask), dtype=bool) if unavailable_mask is None else np.asarray(unavailable_mask, dtype=bool)
    if len(active_mask) and unavailable.shape != (len(active_mask),):
        raise ValueError("unavailable_mask must match number of nodes")

    active = np.asarray(active_mask, dtype=bool) & ~unavailable
    sleeping = (~active) & ~unavailable

    scheduled_energy_per_round = max(float(np.sum(active_costs[active]) + np.sum(sleep_costs[sleeping])), 1e-9)
    baseline_energy_per_round = max(float(np.sum(active_costs)), 1e-9)
    total_battery = float(battery_budget_per_node) * n
    estimated_rounds = total_battery / scheduled_energy_per_round
    baseline_rounds = total_battery / baseline_energy_per_round
    improvement_pct = 0.0 if baseline_rounds <= 0 else 100.0 * (estimated_rounds - baseline_rounds) / baseline_rounds
    return {
        "estimated_network_lifetime_rounds": float(estimated_rounds),
        "baseline_network_lifetime_rounds": float(baseline_rounds),
        "lifetime_improvement_pct": float(improvement_pct),
    }
