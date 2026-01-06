import numpy as np
from scipy.spatial import cKDTree

def calculate_degree_error(pred_q, gt_q):
    # Normalize
    pred_q = pred_q / np.linalg.norm(pred_q)
    gt_q = gt_q / np.linalg.norm(gt_q)

    # Dot product (clamp to handle numerical errors)
    dot = np.clip(np.sum(pred_q * gt_q), -1.0, 1.0)

    # Angle in degrees (Double cover aware: use abs(dot))
    return 2 * np.arccos(np.abs(dot)) * (180 / np.pi)

def compute_add_metric_rotation_only(pts, R_gt, R_pred):
    """
    ADD Metric considering only rotation error.
    """
    pts_pred = np.dot(pts, R_pred.T)
    pts_gt = np.dot(pts, R_gt.T)

    # Calculate Mean Euclidean Distance between corresponding points
    dist = np.linalg.norm(pts_pred - pts_gt, axis=1)

    return np.mean(dist)

def compute_add_metric(pts, R_gt, t_gt, R_pred, t_pred):
    """
    Standard ADD Metric: Average distance between corresponding transformed vertices.
    Used for Asymmetric objects.
    """
    pts_pred = (np.dot(R_pred, pts.T) + t_pred.reshape(3, 1)).T
    pts_gt = (np.dot(R_gt, pts.T) + t_gt.reshape(3, 1)).T

    dist = np.linalg.norm(pts_pred - pts_gt, axis=1)

    return np.mean(dist)

def compute_adds_metric(pts, R_gt, t_gt, R_pred, t_pred):
    """
    ADD-S Metric: Average distance to the NEAREST vertex.
    Used for Symmetric objects.
    """
    pts_pred = (np.dot(R_pred, pts.T) + t_pred.reshape(3, 1)).T
    pts_gt = (np.dot(R_gt, pts.T) + t_gt.reshape(3, 1)).T

    # Use KDTree for fast nearest neighbor lookup
    kdtree = cKDTree(pts_gt)
    distances, _ = kdtree.query(pts_pred)

    return np.mean(distances)

def calc_stats_rotation_only(data_dict):
    if not data_dict['deg']:
        return 0.0, 0.0, 0.0

    return np.mean(data_dict['deg']), np.mean(data_dict['add']), np.mean(data_dict['acc']) * 100

def calc_stats(data_list):
    """Helper to compute Accuracy (<0.1d), Mean Error, and Median Error."""
    if not data_list:
        return 0.0, 0.0, 0.0

    successes, errors = zip(*data_list)
    acc = np.mean(successes) * 100
    mean_err = np.mean(errors)
    median_err = np.median(errors)

    return acc, mean_err, median_err

