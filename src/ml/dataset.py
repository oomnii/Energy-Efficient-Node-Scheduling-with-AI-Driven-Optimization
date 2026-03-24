import numpy as np

from src.algorithms.scheduling import schedule_by_voronoi_threshold, schedule_greedy_coverage
from src.simulation.geometry import field_polygon, random_points
from src.simulation.metrics import compute_coverage_ratio, energy_savings, lifetime_estimate


SENSOR_MULTIPLIERS = {
    "temperature": 0.82,
    "humidity": 0.96,
    "motion": 1.22,
    "homogeneous": 1.0,
    "heterogeneous": 1.0,
    "mixed": 1.0,
}


ACTIVE_ENERGY_MULTIPLIERS = {
    "temperature": 0.85,
    "humidity": 1.0,
    "motion": 1.28,
    "homogeneous": 1.0,
    "heterogeneous": 1.0,
    "mixed": 1.0,
}


def compute_features(points: np.ndarray, sensing_radius, boundary):
    features = []
    cx, cy = boundary.centroid.x, boundary.centroid.y
    if np.isscalar(sensing_radius):
        radii = np.full(len(points), float(sensing_radius), dtype=float)
    else:
        radii = np.asarray(sensing_radius, dtype=float)

    for i, pt in enumerate(points):
        dist_to_center = float(np.linalg.norm(pt - [cx, cy]))
        dists = np.linalg.norm(points - pt, axis=1)
        r = float(radii[i])
        density = int(np.sum(dists <= r) - 1)
        nearest = float(np.min(dists[dists > 0])) if len(points) > 1 and np.any(dists > 0) else r
        features.append([float(pt[0]), float(pt[1]), dist_to_center, density, nearest, r])
    return np.asarray(features, dtype=float)


def compute_run_features(n_nodes: int, width: float, height: float, sensing_radius: float, threshold_coeff: float, sensor_code: float, failure_prob: float) -> np.ndarray:
    density = 0.0 if width * height <= 0 else float(n_nodes / (width * height))
    aspect_ratio = 0.0 if height <= 0 else float(width / height)
    return np.asarray([
        float(n_nodes),
        float(width),
        float(height),
        float(sensing_radius),
        float(threshold_coeff),
        density,
        float(sensor_code),
        float(failure_prob),
        aspect_ratio,
    ], dtype=float)


def generate_training_data(
    num_samples: int = 18,
    num_nodes: int = 60,
    sensing_radius: float = 15.0,
    width: float = 100.0,
    height: float = 100.0,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng(42)
    all_features = []
    all_labels = []

    for _ in range(num_samples):
        sample_nodes = int(rng.integers(max(20, num_nodes - 25), num_nodes + 55))
        sample_width = float(rng.uniform(max(40.0, width * 0.65), width * 1.6))
        sample_height = float(rng.uniform(max(40.0, height * 0.65), height * 1.6))
        sample_radius = float(rng.uniform(max(6.0, sensing_radius * 0.65), sensing_radius * 1.45))
        target_coverage = float(rng.uniform(0.96, 0.995))
        sensor_bias = rng.choice(["homogeneous", "temperature", "humidity", "motion"], p=[0.35, 0.2, 0.2, 0.25])
        radius_mult = SENSOR_MULTIPLIERS.get(sensor_bias, 1.0)

        boundary = field_polygon(sample_width, sample_height)
        points = random_points(sample_nodes, sample_width, sample_height, rng)
        effective_radius = sample_radius * radius_mult
        features = compute_features(points, effective_radius, boundary)
        active_mask, _ = schedule_greedy_coverage(points, boundary, effective_radius, target_coverage=target_coverage)
        labels = active_mask.astype(int)
        all_features.append(features)
        all_labels.append(labels)

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    return X, y


def generate_metric_training_data(num_samples: int = 80, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(314)
    X_rows = []
    y_rows = []
    sensor_choices = ["homogeneous", "temperature", "humidity", "motion", "heterogeneous", "mixed"]

    for _ in range(num_samples):
        n_nodes = int(rng.integers(25, 220))
        width = float(rng.uniform(40.0, 180.0))
        height = float(rng.uniform(40.0, 180.0))
        sensing_radius = float(rng.uniform(8.0, 24.0))
        threshold_coeff = float(rng.uniform(0.005, 0.09))
        failure_prob = float(rng.uniform(0.0, 0.10))
        sensor_type = sensor_choices[int(rng.integers(0, len(sensor_choices)))]
        sensor_code = float(sensor_choices.index(sensor_type))

        boundary = field_polygon(width, height)
        points = random_points(n_nodes, width, height, rng)

        if sensor_type == "heterogeneous":
            eff_radius = rng.uniform(sensing_radius * 0.7, sensing_radius * 1.35, size=n_nodes)
            active_cost = rng.uniform(0.8, 1.35, size=n_nodes)
        elif sensor_type == "mixed":
            mixed_labels = rng.choice(["temperature", "humidity", "motion"], size=n_nodes, p=[0.35, 0.35, 0.30])
            eff_radius = np.asarray([sensing_radius * SENSOR_MULTIPLIERS[label] for label in mixed_labels], dtype=float)
            active_cost = np.asarray([ACTIVE_ENERGY_MULTIPLIERS[label] for label in mixed_labels], dtype=float)
        else:
            eff_radius = sensing_radius * SENSOR_MULTIPLIERS.get(sensor_type, 1.0)
            active_cost = np.full(n_nodes, ACTIVE_ENERGY_MULTIPLIERS.get(sensor_type, 1.0), dtype=float)

        active_mask, _ = schedule_by_voronoi_threshold(points, boundary, eff_radius, threshold_coeff)
        coverage = compute_coverage_ratio(points[active_mask], np.asarray(eff_radius)[active_mask] if not np.isscalar(eff_radius) else eff_radius, boundary)
        energy = energy_savings(active_mask, active_cost, np.asarray(active_cost) * 0.05 if not np.isscalar(active_cost) else active_cost * 0.05, rounds=1)
        lifetime = lifetime_estimate(active_mask, active_cost, np.asarray(active_cost) * 0.05 if not np.isscalar(active_cost) else active_cost * 0.05, battery_budget_per_node=100.0)

        X_rows.append(compute_run_features(n_nodes, width, height, sensing_radius, threshold_coeff, sensor_code, failure_prob))
        y_rows.append([
            float(coverage),
            float(energy["energy_saved_pct"]),
            float(lifetime["estimated_network_lifetime_rounds"]),
        ])

    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)
