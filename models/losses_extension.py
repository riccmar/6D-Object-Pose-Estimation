import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseLoss(nn.Module):
    def __init__(self, w_x=1.0, w_r=1.0):
        super().__init__()
        self.w_x = w_x
        self.w_r = w_r
        self.l1 = nn.L1Loss()

    def forward(self, pred_t, pred_q, gt_t, gt_R, points):
        # Translation Loss
        loss_t = self.l1(pred_t, gt_t)

        # Rotation Loss (Point Matching)
        pred_q = F.normalize(pred_q, p=2, dim=1)
        b = pred_q.shape[0]
        x, y, z, w = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2], pred_q[:, 3]
        pred_R = torch.stack([
            1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
        ], dim=1).reshape(b, 3, 3)

        points_t = points.permute(0, 2, 1)
        pts_pred = torch.bmm(pred_R, points_t)
        pts_gt   = torch.bmm(gt_R, points_t)
        loss_r = self.l1(pts_pred, pts_gt)

        return (self.w_x * loss_t) + (self.w_r * loss_r)

def calc_add_distance(pred_t, pred_q, gt_t, gt_R, points):
    with torch.no_grad():
        pred_q = F.normalize(pred_q, p=2, dim=1)
        b = pred_q.shape[0]
        x, y, z, w = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2], pred_q[:, 3]
        pred_R = torch.stack([
            1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
        ], dim=1).reshape(b, 3, 3)

        points_t = points.permute(0, 2, 1)
        pts_pred = torch.bmm(pred_R, points_t) + pred_t.unsqueeze(2)
        pts_gt = torch.bmm(gt_R, points_t) + gt_t.unsqueeze(2)
        dist = torch.norm(pts_pred - pts_gt, dim=1).mean(dim=1)

        return dist.mean().item()