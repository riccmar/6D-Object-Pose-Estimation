import os
import sys
import torch
import numpy as np
import argparse
import gdown
import cv2
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_extension import RgbdFusionNetDataset
from models.models_extension import RGBD_Fusion_Net
from utils.evaluation_metrics import calculate_degree_error
from utils.visualization import draw_pose

def rgbd_fusion_inference(model_path, device='cpu', num_samples=1):
    # Setup
    dataset_root = os.path.join(project_root, 'data/linemod')

    # Handle Google Drive URL
    if model_path.startswith('http'):
        print(f"Model path detected as URL. Downloading from Google Drive...")
        
        checkpoints_dir = os.path.join(project_root, 'checkpoints', 'rgbd_fusion_net')
        os.makedirs(checkpoints_dir, exist_ok=True)

        download_path = os.path.join(checkpoints_dir, 'rgbd_fusion_net_downloaded.pth')
        gdown.download(model_path, download_path, quiet=False, fuzzy=True)
        model_path = download_path
        print(f"Model downloaded to: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Load Model
    print(f"Loading Model from {model_path}...")
    model = RGBD_Fusion_Net().to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # Prepare Dataset
    print("Initializing Validation Dataset...")
    val_set = RgbdFusionNetDataset(dataset_root, split='val')
    
    # Create output directory
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Inference Loop
    for i in range(num_samples):
        # Pick Random Sample
        idx = random.randint(0, len(val_set) - 1)
        print(f"Visualizing Sample {i+1}/{num_samples} (Index: {idx})")

        # Get Raw Info
        raw_sample = val_set.samples[idx]
        rgb_path = raw_sample['rgb_path']
        cam_K = raw_sample['cam_K']

        # Get Processed Data
        data = val_set[idx]
        rgb_tensor = data['rgb'].unsqueeze(0).to(device)
        points = data['points'].unsqueeze(0).to(device)
        gt_t = data['gt_t'].numpy()
        gt_R = data['gt_R'].numpy()
        gt_q = R.from_matrix(gt_R).as_quat() # For table display
        centroid = data['centroid'].numpy()

        # Inference
        with torch.no_grad():
            pred_q, pred_t_res = model(rgb_tensor, points)

            pred_t_res = pred_t_res.cpu().numpy()[0]
            pred_q = pred_q.cpu().numpy()[0]
            pred_q = pred_q / np.linalg.norm(pred_q)

            pred_R = R.from_quat(pred_q).as_matrix()
            pred_t = centroid + pred_t_res

        # Translation Error
        t_err = np.linalg.norm(gt_t - pred_t)

        # Rotation Error
        angle_err_deg = calculate_degree_error(pred_q, gt_q)


        # Draw Axes (Convert meters to mm for draw_pose)
        img_full = cv2.imread(rgb_path)
        img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)

        img_gt = img_full.copy()
        img_gt = draw_pose(img_gt, cam_K, gt_R, gt_t * 1000.0)

        img_pred = img_full.copy()
        img_pred = draw_pose(img_pred, cam_K, pred_R, pred_t * 1000.0)

        # Plotting
        fig = plt.figure(figsize=(12, 8))

        # Images
        ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
        ax1.imshow(img_gt)
        ax1.set_title(f"Ground Truth (Obj {raw_sample['obj_id']})")
        ax1.axis('off')

        ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
        ax2.imshow(img_pred)
        ax2.set_title("Inference Result")
        ax2.axis('off')

        # Data Table
        ax_table = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        ax_table.axis('off')

        gt_t_str = f"[{gt_t[0]:.3f}, {gt_t[1]:.3f}, {gt_t[2]:.3f}]"
        pred_t_str = f"[{pred_t[0]:.3f}, {pred_t[1]:.3f}, {pred_t[2]:.3f}]"
        t_err_str = f"{t_err*100:.2f} cm"

        gt_q_str = f"[{gt_q[0]:.2f}, {gt_q[1]:.2f}, {gt_q[2]:.2f}, {gt_q[3]:.2f}]"
        pred_q_str = f"[{pred_q[0]:.2f}, {pred_q[1]:.2f}, {pred_q[2]:.2f}, {pred_q[3]:.2f}]"
        r_err_str = f"{angle_err_deg:.2f}°"

        cell_text = [
            [gt_t_str, pred_t_str, t_err_str],
            [gt_q_str, pred_q_str, r_err_str]
        ]

        table = ax_table.table(
            cellText=cell_text,
            rowLabels=["Translation", "Rotation"],
            colLabels=["Ground Truth", "Inference", "Error Metric"],
            loc='center',
            cellLoc='center'
        )

        table.scale(1, 2)
        table.set_fontsize(12)

        plt.tight_layout()
        
        output_filename = f'inference_result_sample_{i}.png'
        plt.savefig(os.path.join(output_dir, output_filename))
        print(f"Sample {i+1} saved to: {output_filename}")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RGBD Fusion Net Inference")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of random samples to visualize')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    rgbd_fusion_inference(args.model_path, device, args.num_samples)