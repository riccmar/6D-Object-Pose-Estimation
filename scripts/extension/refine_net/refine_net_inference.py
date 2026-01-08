import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gdown
import cv2
import random
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_extension import RgbdFusionNetDataset
from models.models_extension import RGBD_Fusion_Net, PoseRefineNet
from utils.process_dataset import load_meshes
from utils.visualization import precompute_bbox_corners, draw_pose
from utils.evaluation_metrics import calculate_degree_error

def refine_net_inference(coarse_model_path, refine_model_path, device='cpu', num_samples=3, iterations=2):
    # Setup
    dataset_root = os.path.join(project_root, 'data/linemod')

    checkpoints_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Handle Google Drive URLs
    for name, path in [("Coarse", coarse_model_path), ("Refine", refine_model_path)]:
        if path.startswith('http'):
            print(f"{name} model path detected as URL. Downloading...")
            
            filename = 'coarse_model.pth' if name == "Coarse" else 'refine_model.pth'
            download_path = os.path.join(checkpoints_dir, filename)
            gdown.download(path, download_path, quiet=False, fuzzy=True)
            if name == "Coarse": 
                coarse_model_path = download_path
            else: 
                refine_model_path = download_path
            
            print(f"{name} model downloaded to: {download_path}")

    # Load Models
    print(f"Loading Coarse Model from {coarse_model_path}...")
    coarse_model = RGBD_Fusion_Net().to(device)

    if os.path.exists(coarse_model_path):
        checkpoint = torch.load(coarse_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            coarse_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            coarse_model.load_state_dict(checkpoint)
        print("Coarse model loaded successfully.")
    else:
        print(f"Error: Coarse weights not found at {coarse_model_path}")
        sys.exit(1)
    coarse_model.eval()

    print(f"Loading Refine Model from {refine_model_path}...")
    refine_model = PoseRefineNet().to(device)
    
    if os.path.exists(refine_model_path):
        checkpoint = torch.load(refine_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            refine_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            refine_model.load_state_dict(checkpoint)
        print("Refine model loaded successfully.")
    else:
        print(f"Error: Refine weights not found at {refine_model_path}")
        sys.exit(1)
    refine_model.eval()

    # Prepare Dataset
    print("Initializing Validation Dataset...")
    val_set = RgbdFusionNetDataset(dataset_root, split='val')

    # Load Meshes and Precompute Corners
    meshes = load_meshes(dataset_root)
    meshes = precompute_bbox_corners(meshes)
    
    output_dir = os.path.join(project_root, 'inference_results')
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        # Pick Random Sample
        idx = random.randint(0, len(val_set) - 1)
        print(f"\nProcessing Sample {i+1}/{num_samples} (Index: {idx})")

        # Get Raw Info
        raw_sample = val_set.samples[idx]
        rgb_path = raw_sample['rgb_path']
        cam_K = raw_sample['cam_K']
        obj_id = raw_sample['obj_id']

        # Get Processed Data
        data = val_set[idx]
        rgb_tensor = data['rgb'].unsqueeze(0).to(device)
        points = data['points'].unsqueeze(0).to(device)
        centroid_tensor = data['centroid'].unsqueeze(0).to(device)

        # Ground Truth
        gt_t = data['gt_t'].numpy()
        gt_R = data['gt_R'].numpy()
        gt_q = R.from_matrix(gt_R).as_quat()

        # Get Canonical Corners
        try:
            box_corners = meshes[obj_id]['bbox_3d']
        except KeyError:
            print(f"Warning: No mesh found for Object {obj_id}. Using unit box.")
            s = 50.0  # 50mm = 5cm half-extent
            box_corners = np.array([
                [-s,-s,-s], [s,-s,-s], [s,s,-s], [-s,s,-s],
                [-s,-s,s],  [s,-s,s],  [s,s,s],  [-s,s,s]
            ])

        # Inference (Coarse + Refine)
        with torch.no_grad():
            # 1. Coarse Pass
            pred_q, pred_t_res = coarse_model(rgb_tensor, points)
            pred_t_abs = centroid_tensor + pred_t_res
            pred_q = F.normalize(pred_q, p=2, dim=1)

            # 2. Refinement Loop
            points_T = points.transpose(1, 2) # (B, 3, N)

            for _ in range(iterations):
                b_size = pred_q.shape[0]
                x, y, z, w = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2], pred_q[:, 3]
                
                # Quaternion to Rotation Matrix manually
                pred_R_torch = torch.stack([
                    1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
                    2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
                    2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
                ], dim=1).reshape(b_size, 3, 3)

                # Transform points to local frame of current prediction
                points_centered = points_T - (pred_t_abs - centroid_tensor).unsqueeze(2)
                points_local = torch.bmm(pred_R_torch.transpose(1, 2), points_centered)

                # Get updates from RefineNet
                delta_q, delta_t = refine_model(points_local)

                # Update translation
                t_update = torch.bmm(pred_R_torch, delta_t.unsqueeze(2)).squeeze(2)
                pred_t_abs = pred_t_abs + t_update
                
                # Update rotation
                pred_q = F.normalize(pred_q + delta_q, p=2, dim=1)

            # Convert to Numpy
            pred_t = pred_t_abs.cpu().numpy()[0]
            pred_q_np = pred_q.cpu().numpy()[0]
            pred_R = R.from_quat(pred_q_np).as_matrix()

        # Calculate Errors
        t_err = np.linalg.norm(gt_t - pred_t)
        angle_err_deg = calculate_degree_error(pred_q_np, gt_q)

        # Draw Bounding Boxes and Axes
        img_full = cv2.imread(rgb_path)

        # Ground Truth (Green Box + Axes)
        img_gt_bgr = img_full.copy()
        try:
            img_gt_bgr = draw_pose(img_gt_bgr, cam_K, gt_R, gt_t * 1000, bbox_3d=box_corners, color=(0, 255, 0))
        except Exception as e:
            print(f"Error drawing GT: {e}")
        img_gt = cv2.cvtColor(img_gt_bgr, cv2.COLOR_BGR2RGB)

        # Prediction (Blue Box + Axes)
        img_pred_bgr = img_full.copy()
        try:
            img_pred_bgr = draw_pose(img_pred_bgr, cam_K, pred_R, pred_t * 1000, bbox_3d=box_corners, color=(255, 0, 0)) # Blue in BGR
        except Exception as e:
            print(f"Error drawing Pred: {e}")
        img_pred = cv2.cvtColor(img_pred_bgr, cv2.COLOR_BGR2RGB)

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        ax[0].imshow(img_gt)
        ax[0].set_title(f"Ground Truth (Obj {obj_id})")
        ax[0].axis('off')

        ax[1].imshow(img_pred)
        ax[1].set_title(f"Refined Prediction")
        ax[1].axis('off')

        plt.tight_layout()
        
        # Save result
        save_path = os.path.join(output_dir, f"refine_infer_{idx}.png")
        plt.savefig(save_path)
        plt.close(fig)

        # Written Output
        print(f"METRICS REPORT | Object ID: {obj_id}")
        print(f"Translation Error: {t_err*100:.2f} cm")
        print(f"Rotation Error:    {angle_err_deg:.2f}°")
        print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for RefineNet")
    parser.add_argument('--coarse_model_path', type=str, required=True, help='Path to the trained Coarse Rgbd Fusion Net or Google Drive URL')
    parser.add_argument('--refine_model_path', type=str, required=True, help='Path to the trained Refine Net or Google Drive URL')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--iterations', type=int, default=2, help='Number of refinement iterations')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    refine_net_inference(args.coarse_model_path, args.refine_model_path, device, args.num_samples, args.iterations)
