from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Obstacle:
    center: np.ndarray  # (3,) in robot base frame
    radius: float        # meters


def crop_to_workspace(points: np.ndarray, bounds: dict) -> np.ndarray:
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    mask = (
        (points[:, 0] >= x0) & (points[:, 0] <= x1)
        & (points[:, 1] >= y0) & (points[:, 1] <= y1)
        & (points[:, 2] >= z0) & (points[:, 2] <= z1)
    )
    return points[mask]


def cluster_to_spheres(points: np.ndarray, voxel_size: float = 0.03,
                        min_points_per_cluster: int = 8) -> list:
    """Voxelize points into an occupancy grid and connected-component label
    it. Cheap and dependency-light (scipy.ndimage) compared to a general
    clustering algorithm, and good enough for coarse obstacle spheres -- this
    is not meant to reconstruct precise object shape, just to bound it.
    """
    if len(points) == 0:
        return []

    mins = points.min(axis=0)
    idx = np.floor((points - mins) / voxel_size).astype(int)
    grid_shape = idx.max(axis=0) + 1
    occ = np.zeros(grid_shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    labels, n_labels = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    voxel_labels = labels[idx[:, 0], idx[:, 1], idx[:, 2]]

    obstacles = []
    for label_id in range(1, n_labels + 1):
        cluster_pts = points[voxel_labels == label_id]
        if len(cluster_pts) < min_points_per_cluster:
            continue
        center = cluster_pts.mean(axis=0)
        radius = float(np.linalg.norm(cluster_pts - center, axis=1).max())
        obstacles.append(Obstacle(center=center, radius=max(radius, 0.02)))
    return obstacles


def filter_self_obstacles(obstacles: list, link_spheres: list, margin: float = 0.02) -> list:
    """Drop obstacle spheres that are really just the arm's own current
    pose. Heuristic: an obstacle is "self" if it overlaps any link sphere
    once you inflate the link sphere by `margin` for calibration slop.
    """
    kept = []
    for obs in obstacles:
        is_self = False
        for link_center, link_radius in link_spheres:
            dist = np.linalg.norm(obs.center - link_center)
            if dist <= (link_radius + obs.radius + margin):
                is_self = True
                break
        if not is_self:
            kept.append(obs)
    return kept
