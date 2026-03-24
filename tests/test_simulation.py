import numpy as np
from shapely.geometry import box
from src.simulation.geometry import voronoi_areas, field_polygon

def test_voronoi_areas_sum_to_boundary_approx():
    boundary = box(0, 0, 100, 100)
    points = np.array([[20,20],[80,20],[20,80],[80,80]], dtype=float)
    areas = voronoi_areas(points, boundary)
    assert abs(float(areas.sum()) - float(boundary.area)) < 1e-6

def test_field_polygon_creation():
    poly = field_polygon(100, 100)
    assert poly.area == 10000
    assert poly.bounds == (0.0, 0.0, 100.0, 100.0)

def test_degenerate_voronoi():
    boundary = box(0, 0, 100, 100)
    points = np.array([[50,50]], dtype=float)
    areas = voronoi_areas(points, boundary)
    assert len(areas) == 1
    assert abs(areas[0] - 10000.0) < 1e-6

def test_collinear_voronoi():
    boundary = box(0, 0, 100, 100)
    points = np.array([[50,20],[50,50],[50,80]], dtype=float)
    areas = voronoi_areas(points, boundary)
    assert len(areas) == 3
    assert abs(sum(areas) - 10000.0) < 1e-6
