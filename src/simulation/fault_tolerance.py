import numpy as np
from shapely.geometry import Polygon

from src.simulation.geometry import sensing_coverage_union
from src.simulation.metrics import compute_coverage_ratio, energy_savings


def _normalize_radii(sensing_radius, n: int) -> np.ndarray:
    if np.isscalar(sensing_radius):
        return np.full(n, float(sensing_radius), dtype=float)
    radii = np.asarray(sensing_radius, dtype=float)
    if radii.shape != (n,):
        raise ValueError("sensing_radius array must match number of nodes")
    return radii


def simulate_failures_and_recovery(
    points: np.ndarray,
    boundary: Polygon,
    sensing_radius,
    active_mask_init: np.ndarray,
    backups: list[int],
    n_rounds: int,
    failure_prob: float,
    failure_model: str,
    recovery_model: str,
    rng: np.random.Generator,
    target_coverage: float = 0.99,
    active_costs=None,
    sleep_costs=None,
):
    radii = _normalize_radii(sensing_radius, len(points))
    active = np.asarray(active_mask_init, dtype=bool).copy()
    backup_available = set(int(x) for x in backups)
    unavailable = np.zeros(len(points), dtype=bool)
    logs = []
    cumulative_failures = 0
    cumulative_recoveries = 0

    for r in range(1, n_rounds + 1):
        active_before_round = int(active.sum())
        active_idx = np.where(active & ~unavailable)[0]
        failed: list[int] = []

        if len(active_idx) > 0 and failure_prob > 0:
            effective_prob = float(np.clip(failure_prob, 0.0, 0.20))
            if failure_model == "random":
                mask = rng.random(len(active_idx)) < effective_prob
                failed = active_idx[mask].tolist()
            elif failure_model == "periodic" and r % 5 == 0:
                mask = rng.random(len(active_idx)) < min(0.30, effective_prob * 1.5)
                failed = active_idx[mask].tolist()
            elif failure_model == "clustered" and rng.random() < min(0.40, effective_prob * 1.25):
                center = points[rng.choice(active_idx)]
                mean_radius = float(np.mean(radii[active_idx])) if len(active_idx) else 0.0
                dists = np.linalg.norm(points[active_idx] - center, axis=1)
                mask = dists < (mean_radius * 1.35)
                cluster_nodes = active_idx[mask]
                if len(cluster_nodes):
                    submask = rng.random(len(cluster_nodes)) < min(0.65, effective_prob * 2.0)
                    failed = cluster_nodes[submask].tolist()
            elif failure_model == "region" and rng.random() < min(0.35, effective_prob * 1.2):
                cx, cy = boundary.centroid.x, boundary.centroid.y
                mask = (points[active_idx, 0] < cx) & (points[active_idx, 1] < cy)
                region_nodes = active_idx[mask]
                if len(region_nodes):
                    submask = rng.random(len(region_nodes)) < min(0.55, effective_prob * 1.8)
                    failed = region_nodes[submask].tolist()

            for fn in failed:
                active[int(fn)] = False
                unavailable[int(fn)] = True

        cumulative_failures += len(failed)
        coverage_after_failures = compute_coverage_ratio(points[active], radii[active], boundary)
        activated: list[int] = []

        if coverage_after_failures + 1e-9 < target_coverage and backup_available:
            if recovery_model == "nearest_backup":
                for fn in failed:
                    if not backup_available:
                        break
                    fn_pt = points[int(fn)]
                    best_b, best_dist = None, float("inf")
                    for b in list(backup_available):
                        dist = float(np.linalg.norm(points[b] - fn_pt))
                        if dist < best_dist:
                            best_dist = dist
                            best_b = b
                    if best_b is not None:
                        active[best_b] = True
                        backup_available.remove(best_b)
                        activated.append(int(best_b))
                coverage_after_failures = compute_coverage_ratio(points[active], radii[active], boundary)

            if recovery_model == "greedy_coverage" or coverage_after_failures + 1e-9 < target_coverage:
                while coverage_after_failures + 1e-9 < target_coverage and backup_available:
                    current_union = sensing_coverage_union(points[active], radii[active], boundary)
                    best, best_gain = None, 0.0
                    for b in list(backup_available):
                        candidate = current_union.union(sensing_coverage_union(points[[b]], radii[[b]], boundary))
                        gain = float(candidate.area - current_union.area)
                        if gain > best_gain + 1e-9:
                            best_gain, best = gain, b
                    if best is None or best_gain <= 1e-9:
                        break
                    active[best] = True
                    backup_available.remove(best)
                    activated.append(int(best))
                    coverage_after_failures = compute_coverage_ratio(points[active], radii[active], boundary)

        cumulative_recoveries += len(activated)
        round_energy = energy_savings(active, active_costs if active_costs is not None else 1.0, sleep_costs if sleep_costs is not None else 0.05, rounds=1, unavailable_mask=unavailable)
        logs.append({
            "round": r,
            "active_before_round": active_before_round,
            "coverage_after_failures": float(coverage_after_failures),
            "coverage": float(coverage_after_failures),
            "n_active": int(active.sum()),
            "n_backup_available": int(len(backup_available)),
            "n_unavailable": int(unavailable.sum()),
            "failed": [int(x) for x in failed],
            "activated": activated,
            "failed_count": int(len(failed)),
            "activated_count": int(len(activated)),
            "cumulative_failures": int(cumulative_failures),
            "cumulative_recoveries": int(cumulative_recoveries),
            "scheduled_energy": float(round_energy["scheduled_energy"]),
            "baseline_energy": float(round_energy["baseline_energy"]),
            "energy_saved_pct": float(round_energy["energy_saved_pct"]),
        })

    return active, sorted(backup_available), unavailable, logs
