import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


def field_polygon(width: float, height: float) -> Polygon:
    return box(0.0, 0.0, width, height)


def random_points(n: int, width: float, height: float, rng: np.random.Generator) -> np.ndarray:
    xs = rng.uniform(0, width, size=n)
    ys = rng.uniform(0, height, size=n)
    return np.column_stack([xs, ys])


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        extent = np.ptp(vor.points, axis=0)
        radius = float(np.max(extent)) * 2.0 + 1.0

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent) + 1e-12
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = (vor.points[p1] + vor.points[p2]) / 2
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region, strict=False), key=lambda item: item[0])]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def bounded_voronoi_cells(points: np.ndarray, boundary: Polygon) -> list[Polygon]:
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [boundary]
    try:
        vor = Voronoi(points, qhull_options="Qbb Qc Qz")
    except Exception:
        if len(points) == 2:
            midx = float(np.mean(points[:, 0]))
            left = boundary.intersection(box(boundary.bounds[0], boundary.bounds[1], midx, boundary.bounds[3]))
            right = boundary.intersection(box(midx, boundary.bounds[1], boundary.bounds[2], boundary.bounds[3]))
            return [left, right]
        return [boundary if i == 0 else Polygon() for i in range(len(points))]

    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=1e6)
    cells: list[Polygon] = []
    for region in regions:
        poly = Polygon(vertices[region])
        clipped = poly.intersection(boundary)
        cells.append(clipped if not clipped.is_empty else Polygon())
    return cells


def voronoi_areas(points: np.ndarray, boundary: Polygon) -> np.ndarray:
    cells = bounded_voronoi_cells(points, boundary)
    return np.array([float(c.area) for c in cells], dtype=float)


def sensing_coverage_union(points: np.ndarray, sensing_radius, boundary: Polygon) -> Polygon:
    if len(points) == 0:
        return Polygon()
    if np.isscalar(sensing_radius):
        radii = np.full(len(points), float(sensing_radius), dtype=float)
    else:
        radii = np.asarray(sensing_radius, dtype=float)
    circles = [Point(pt[0], pt[1]).buffer(float(r)) for pt, r in zip(points, radii, strict=False)]
    return unary_union(circles).intersection(boundary)
