import numpy as np

def calculate_degree_error(pred_q, gt_q):
    # Normalize
    pred_q = pred_q / np.linalg.norm(pred_q)
    gt_q = gt_q / np.linalg.norm(gt_q)

    # Dot product (clamp to handle numerical errors)
    dot = np.clip(np.sum(pred_q * gt_q), -1.0, 1.0)

    # Angle in degrees (Double cover aware: use abs(dot))
    return 2 * np.arccos(np.abs(dot)) * (180 / np.pi)

def compute_add_metric(pts, R_pred, R_gt):
    """
    Computes ADD metric.
    Assumes perfect translation (T_pred = T_gt) since ResNet predicts Rotation only.
    """
    # Rotate the model points
    # Points are (N, 3), Rotation is (3, 3)
    # R * P^T -> (3, 3) * (3, N) = (3, N) -> Transpose to (N, 3)
    # Equivalent to P * R^T
    pts_pred = np.dot(pts, R_pred.T)
    pts_gt = np.dot(pts, R_gt.T)

    # Calculate Mean Euclidean Distance between corresponding points
    dist = np.linalg.norm(pts_pred - pts_gt, axis=1)

    return np.mean(dist)

def calc_stats(data_dict):
    if not data_dict['deg']:
      return 0, 0, 0

    return np.mean(data_dict['deg']), np.mean(data_dict['add']), np.mean(data_dict['acc']) * 100
