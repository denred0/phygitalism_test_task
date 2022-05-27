import numpy as np
import open3d as o3d
import argparse
import csv
import pickle

from sklearn.cluster import DBSCAN
from tqdm import tqdm
from typing import List


def voxel_clustering_visualization(points: np.ndarray,
                                   colors: np.ndarray,
                                   voxel_size: float,
                                   type='barycenter') -> [List, List]:
    # nb_vox.astype(int) #this gives you the number of voxels per axis
    nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0)) / voxel_size)

    # non_empty_voxel_keys - voxels coords
    # inverse - inds of voxels connected with points
    # nb_pts_per_voxel - count of points in every voxel
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)

    # sort according voxels inds
    idx_pts_vox_sorted = np.argsort(inverse)

    voxel_grid = {}
    voxel_grid_colors = {}
    grid_center, grid_center_colors = [], []
    last_seen = 0

    for idx, vox in tqdm(enumerate(non_empty_voxel_keys), desc="Visualization", total=len(non_empty_voxel_keys)):
        # points and colors of every voxel
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]
        voxel_grid_colors[tuple(vox)] = colors[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]

        if type == "barycenter":
            grid_center.append(np.mean(voxel_grid[tuple(vox)], axis=0))
            grid_center_colors.append(np.mean(voxel_grid_colors[tuple(vox)], axis=0))
        elif type == "candidate_center":
            grid_center.append(voxel_grid[tuple(vox)][
                                   np.linalg.norm(
                                       voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)], axis=0),
                                       axis=1).argmin()])
            grid_center_colors.append(voxel_grid_colors[tuple(vox)][
                                          np.linalg.norm(
                                              voxel_grid_colors[tuple(vox)] - np.mean(
                                                  voxel_grid_colors[tuple(vox)],
                                                  axis=0),
                                              axis=1).argmin()])
        last_seen += nb_pts_per_voxel[idx]

    return grid_center, grid_center_colors


def voxel_clustering(points: np.ndarray,
                     colors: np.ndarray,
                     voxel_size: float,
                     type='barycenter') -> [List, List, List]:
    # nb_vox.astype(int) #this gives you the number of voxels per axis
    nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0)) / voxel_size)

    # non_empty_voxel_keys - voxels coords
    # inverse - inds of voxels connected with points
    # nb_pts_per_voxel - count of points in every voxel
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)

    # sort according voxels inds
    idx_pts_vox_sorted = np.argsort(inverse)

    voxel_grid = {}
    voxel_grid_colors = {}
    last_seen = 0
    center_points, cluster_points, cluster_colors = [], [], []

    for idx, vox in tqdm(enumerate(non_empty_voxel_keys), desc="Clustering", total=len(non_empty_voxel_keys), ):
        # points and colors of every voxel
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]
        voxel_grid_colors[tuple(vox)] = colors[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]

        if type == "barycenter":
            center_points.append(np.mean(voxel_grid[tuple(vox)], axis=0))
            cluster_points.append(voxel_grid[tuple(vox)])
            cluster_colors.append(voxel_grid_colors[tuple(vox)])

        elif type == "candidate_center":
            center_points.append(
                voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)]
                                                      - np.mean(voxel_grid[tuple(vox)], axis=0), axis=1).argmin()])
            cluster_points.append(voxel_grid[tuple(vox)])
            cluster_colors.append(voxel_grid_colors[tuple(vox)])

        last_seen += nb_pts_per_voxel[idx]

    return center_points, cluster_points, cluster_colors


def dbscan_clustering(points: np.array,
                      eps: float,
                      min_samples=10) -> [List, List, List]:
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    center_points, cluster_points, cluster_colors = [], [], []
    for i in tqdm(np.unique(labels), desc="Clustering"):
        # skip outliers
        if i == -1:
            continue

        # get inds of cluster
        inds = np.where(labels == i)
        center_points.append(np.mean(points[inds], axis=0))
        cluster_points.append(points[inds])

        # generate color for every cluster
        color = np.random.choice(range(255), size=3) / 255
        cluster_colors.append([color] * len(points[inds]))

    return center_points, cluster_points, cluster_colors


def dbscan_clustering_visuazlization(points: np.array,
                                     colors: np.array,
                                     eps: float,
                                     min_samples=10) -> [List, List]:
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    for i in tqdm(np.unique(labels), desc="Visualization"):
        inds = np.where(labels == i)
        color = np.random.choice(range(255), size=3) / 255
        colors[inds] = color

    return points, colors


def save_points(center_points, cluster_points, cluster_colors, type):

    assert len(center_points) == len(cluster_points) == len(cluster_colors), "Lenghts mismatch! Check your algorithm."

    # centers
    center_points_dict = {}
    for i, point in enumerate(center_points):
        center_points_dict[i] = point

    with open(f'center_points_{type}.pkl', 'wb') as f:
        pickle.dump(center_points_dict, f)

    # points
    cluster_points_dict = {}
    for i, point in enumerate(cluster_points):
        cluster_points_dict[i] = point

    with open(f'cluster_points_{type}.pkl', 'wb') as f:
        pickle.dump(cluster_points_dict, f)

    # colors
    cluster_color_dict = {}
    for i, color in enumerate(cluster_colors):
        cluster_color_dict[i] = color

    with open(f'cluster_colors_{type}.pkl', 'wb') as f:
        pickle.dump(cluster_color_dict, f)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='path to file with cloud data')
    parser.add_argument('type', type=str, help='Type of clustering (voxel, dbscan)')
    parser.add_argument('--radius', type=float, default=0.5, help='Radius of one segment')
    parser.add_argument('--vis', action="store_true", help='Draw visualization')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    data_path = opt.data
    type = opt.type  # (voxel, dbscan)
    radius = opt.radius
    draw_visualizaton = opt.vis

    raw_data = np.loadtxt(data_path)
    points = raw_data[:, :3]
    colors = raw_data[:, 3:6] / 255

    print(f"Points count: {len(points)}")

    if type == "dbscan":
        center_points, cluster_points, cluster_colors = dbscan_clustering(points, radius)
        save_points(center_points, cluster_points, cluster_colors, type)

        if draw_visualizaton:
            point_vis, colors_vis = dbscan_clustering_visuazlization(points, colors, radius)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
            o3d.visualization.draw_geometries([pcd],
                                              width=1280,
                                              height=720)

    elif type == "voxel":
        center_points, cluster_points, cluster_colors = voxel_clustering(points, colors, radius)
        save_points(center_points, cluster_points, cluster_colors, type)

        if draw_visualizaton:
            point_vis, colors_vis = voxel_clustering_visualization(points, colors, radius)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
            o3d.visualization.draw_geometries([pcd],
                                              width=1280,
                                              height=720)



    else:
        print(f"{type} clustering doesn't exist. Please choose from ('voxel', 'dbscan')")
