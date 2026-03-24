import numpy as np
from shapely.geometry import Polygon

from src.simulation.geometry import sensing_coverage_union, voronoi_areas
from src.simulation.metrics import compute_coverage_ratio


def _normalize_radii(sensing_radius, n: int) -> np.ndarray:
    if np.isscalar(sensing_radius):
        return np.full(n, float(sensing_radius), dtype=float)
    radii = np.asarray(sensing_radius, dtype=float)
    if radii.shape != (n,):
        raise ValueError("sensing_radius array must match number of nodes")
    return radii


def schedule_by_voronoi_threshold(points: np.ndarray, boundary: Polygon, sensing_radius, threshold_coeff: float):
    n = len(points)
    if n == 0:
        return np.array([], dtype=bool), []

    radii = _normalize_radii(sensing_radius, n)
    threshold_area_per_node = float(threshold_coeff) * (np.pi * (radii ** 2))

    active = np.ones(n, dtype=bool)
    backups: list[int] = []

    v_areas_all = voronoi_areas(points, boundary)
    initial_off_indices = np.where(v_areas_all < threshold_area_per_node)[0]
    for off_idx in initial_off_indices:
        active[off_idx] = False
        backups.append(int(off_idx))

    while True:
        active_idx = np.where(active)[0]
        if len(active_idx) <= 1:
            break

        areas = voronoi_areas(points[active_idx], boundary)
        m_local = int(np.argmin(areas))
        m_global = int(active_idx[m_local])

        if float(areas[m_local]) < threshold_area_per_node[m_global]:
            active[m_global] = False
            backups.append(m_global)
        else:
            break
    return active, backups


def schedule_ai_driven(points: np.ndarray, boundary: Polygon, sensing_radius, model, target_coverage: float = 0.99):
    from src.ml.dataset import compute_features
    from src.ml.model import predict_nodes

    n = len(points)
    if n == 0:
        return np.array([], dtype=bool), []

    radii = _normalize_radii(sensing_radius, n)
    features = compute_features(points, radii, boundary)
    active = np.asarray(predict_nodes(model, features), dtype=bool)

    if active.shape != (n,):
        active = np.ones(n, dtype=bool)

    if not np.any(active):
        active[int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))] = True

    coverage = compute_coverage_ratio(points[active], radii[active], boundary)
    inactive = set(np.where(~active)[0].tolist())
    while coverage + 1e-9 < float(target_coverage) and inactive:
        current_union = sensing_coverage_union(points[active], radii[active], boundary)
        best_idx, best_gain = None, -1.0
        for idx in list(inactive):
            candidate_union = current_union.union(sensing_coverage_union(points[[idx]], radii[[idx]], boundary))
            gain = float(candidate_union.area - current_union.area)
            if gain > best_gain + 1e-9:
                best_idx, best_gain = idx, gain
        if best_idx is None or best_gain <= 1e-9:
            break
        active[best_idx] = True
        inactive.remove(best_idx)
        coverage = compute_coverage_ratio(points[active], radii[active], boundary)

    backups = np.where(~active)[0].astype(int).tolist()
    return active, backups


def schedule_random_same_off(points: np.ndarray, boundary: Polygon, sensing_radius, n_off: int, rng: np.random.Generator):
    n = len(points)
    n_off = max(0, min(int(n_off), max(n - 1, 0)))
    idx = np.arange(n)
    rng.shuffle(idx)
    off = set(idx[:n_off].tolist())
    active = np.array([i not in off for i in range(n)], dtype=bool)
    backups = list(off)
    return active, backups


def schedule_random_greedy_coverage(points: np.ndarray, boundary: Polygon, sensing_radius, target_coverage: float, rng: np.random.Generator):
    """
    Heuristic Pruning algorithm for WSN coverage scheduling.
    Previously random, this is now corrected to the "best possible" by sorting nodes based on their 
    bounded Voronoi area. Nodes with the smallest Voronoi area are the most redundant. 
    Reference: Voronoi-based density control models in WSNs (IEEE).
    """
    from src.simulation.geometry import voronoi_areas
    n = len(points)
    active = np.ones(n, dtype=bool)
    
    if n == 0:
        return active, []
        
    areas = voronoi_areas(points, boundary)
    order = np.argsort(areas)

    radii = _normalize_radii(sensing_radius, n)

    for i in order:
        active[i] = False
        cov = compute_coverage_ratio(points[active], radii[active], boundary)
        if cov + 1e-9 < target_coverage:
            active[i] = True
    backups = np.where(~active)[0].tolist()
    return active, backups


def schedule_greedy_coverage(points: np.ndarray, boundary: Polygon, sensing_radius, target_coverage: float = 0.99):
    """
    Implements a set-cover-based Greedy Coverage algorithm, a well-established standard in WSN 
    literature for maximizing network lifetime by selecting sensor subsets that cover the target area.
    Reference: IEEE Target Coverage Problem optimizations using greedy strategies.
    Starts with all nodes OFF, greedily turns ON the node that contributes the most to un-covered area.
    """
    n = len(points)
    active = np.zeros(n, dtype=bool)
    if n == 0:
        return active, []

    radii = _normalize_radii(sensing_radius, n)
    current_cov = 0.0
    unassigned = set(range(n))

    while current_cov + 1e-9 < target_coverage and unassigned:
        best_candidate_idx = None
        best_gain = -1.0
        current_union = sensing_coverage_union(points[active], radii[active], boundary)

        for cand_idx in unassigned:
            cand_polygon = sensing_coverage_union(points[[cand_idx]], radii[[cand_idx]], boundary)
            union_with_cand = current_union.union(cand_polygon)
            gain = float(union_with_cand.area - current_union.area)
            if gain > best_gain + 1e-9:
                best_gain = gain
                best_candidate_idx = cand_idx

        if best_candidate_idx is None or best_gain <= 1e-9:
            break

        active[best_candidate_idx] = True
        unassigned.remove(best_candidate_idx)
        current_cov = compute_coverage_ratio(points[active], radii[active], boundary)

    backups = sorted(unassigned)
    return active, backups
