import numpy as np
from shapely.geometry import box

from src.algorithms.scheduling import (
    schedule_ai_driven,
    schedule_by_voronoi_threshold,
    schedule_greedy_coverage,
)
from src.ml.model import train_model


def test_threshold_zero_keeps_all():
    boundary = box(0, 0, 100, 100)
    points = np.array([[10, 10], [90, 10], [10, 90], [90, 90], [50, 50]], dtype=float)
    active, backups = schedule_by_voronoi_threshold(points, boundary, sensing_radius=15.0, threshold_coeff=0.0)
    assert active.sum() == len(points)
    assert len(backups) == 0


def test_greedy_coverage():
    boundary = box(0, 0, 100, 100)
    points = np.array([[10, 10], [90, 10], [10, 90], [90, 90], [50, 50]], dtype=float)
    active, backups = schedule_greedy_coverage(points, boundary, sensing_radius=30.0, target_coverage=1.0)
    assert active.sum() > 0
    assert len(backups) < len(points)


def test_heterogeneous_greedy_coverage():
    boundary = box(0, 0, 100, 100)
    points = np.array([[10, 10], [90, 10], [10, 90], [90, 90], [50, 50]], dtype=float)
    radii = np.array([10.0, 10.0, 10.0, 10.0, 100.0])
    active, _backups = schedule_greedy_coverage(points, boundary, sensing_radius=radii, target_coverage=1.0)
    assert active[4]
    assert active.sum() < 5


def test_ai_driven_returns_valid_mask():
    boundary = box(0, 0, 100, 100)
    points = np.array([[10, 10], [90, 10], [10, 90], [90, 90], [50, 50]], dtype=float)
    X = np.array([[10, 10, 0, 1, 10, 15], [90, 10, 0, 1, 10, 15], [10, 90, 0, 1, 10, 15], [90, 90, 0, 1, 10, 15], [50, 50, 0, 4, 56, 15]], dtype=float)
    y = np.array([1, 1, 1, 1, 1], dtype=int)
    model = train_model(X, y)
    active, backups = schedule_ai_driven(points, boundary, sensing_radius=15.0, model=model)
    assert len(active) == len(points)
    assert active.dtype == bool
    assert len(backups) == int((~active).sum())
