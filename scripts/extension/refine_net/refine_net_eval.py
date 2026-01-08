import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gdown
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R_scipy

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_extension import RgbdFusionNetDataset
from models.models_extension import RGBD_Fusion_Net, PoseRefineNet
from dataset.dataset_baseline import LINEMOD_ID_MAP
from utils.process_dataset import load_meshes
from utils.evaluation_metrics import compute_add_metric, compute_adds_metric


def refine_net_evaluation(coarse_model_path, refine_model_path, device='cpu', batch_size=32, refine_iters=2):
    # Setup Paths
    dataset_root = os.path.join(project_root, 'data/linemod')

    checkpoints_dir = os.path.join(project_root, 'checkpoints', 'refine_net')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Handle Google Drive URLs
    for name, path in [('Coarse', coarse_model_path), ('Refine', refine_model_path)]:
        if path.startswith('http'):
            print(f"{name} model path detected as URL. Downloading...")

            download_path = os.path.join(checkpoints_dir, f"{name.lower()}_model_downloaded.pth")
            gdown.download(path, download_path, quiet=False, fuzzy=True)
            if name == 'Coarse': 
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
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load Meshes
    meshes = load_meshes(dataset_root)
    # Convert meshes to meters
    for obj_id in meshes:
        if meshes[obj_id]['diameter'] > 10.0:
            meshes[obj_id]['vertices'] /= 1000.0
            meshes[obj_id]['diameter'] /= 1000.0

    results = {}
    SYMMETRIC_IDS = [10, 11]

    print(f"Evaluating on {len(val_set)} images with {refine_iters} refinement iterations...")

    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Move to device
            rgb = batch['rgb'].to(device)
            points = batch['points'].to(device)
            gt_t = batch['gt_t'].cpu().numpy()
            gt_R = batch['gt_R'].cpu().numpy()
            centroid = batch['centroid'].to(device)
            obj_ids = batch['obj_id'].numpy()

            # Coarse Prediction
            pred_q, pred_t_res = coarse_model(rgb, points)
            pred_t_abs = centroid + pred_t_res
            pred_q = F.normalize(pred_q, p=2, dim=1)

            # Refinement Loop
            points_T = points.transpose(1, 2) # (B, 3, N)

            for _ in range(refine_iters):
                # Convert Quat to Rotation Matrix
                b = pred_q.shape[0]
                x, y, z, w = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2], pred_q[:, 3]
                pred_R_torch = torch.stack([
                    1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
                    2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
                    2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
                ], dim=1).reshape(b, 3, 3)

                points_centered = points_T - (pred_t_abs - centroid).unsqueeze(2)
                points_local = torch.bmm(pred_R_torch.transpose(1, 2), points_centered)

                # Predict Residuals
                delta_q, delta_t = refine_model(points_local)

                # Update Translation: t_new = t_old + R * delta_t
                t_update = torch.bmm(pred_R_torch, delta_t.unsqueeze(2)).squeeze(2)
                pred_t_abs = pred_t_abs + t_update

                # Update Rotation: q_new = q_old + delta_q
                pred_q = F.normalize(pred_q + delta_q, p=2, dim=1)

            # Process Final Predictions
            pred_t_final = pred_t_abs.cpu().numpy()
            pred_q_final = pred_q.cpu().numpy()

            for i in range(len(rgb)):
                obj_id = int(obj_ids[i])
                
                if obj_id not in meshes:
                    continue
                if obj_id not in results:
                    results[obj_id] = []

                mesh_pts = meshes[obj_id]['vertices']
                diameter = meshes[obj_id]['diameter']

                # Final Prediction
                q = pred_q_final[i]
                t_pred = pred_t_final[i]
                
                # Normalize just in case
                q = q / np.linalg.norm(q)
                R_pred = R_scipy.from_quat(q).as_matrix()

                # Gt
                R_gt_i = gt_R[i]
                t_gt_i = gt_t[i]

                # Metric Calculation
                if obj_id in SYMMETRIC_IDS:
                    err = compute_adds_metric(mesh_pts, R_gt_i, t_gt_i, R_pred, t_pred)
                else:
                    err = compute_add_metric(mesh_pts, R_gt_i, t_gt_i, R_pred, t_pred)

                # Accuracy Thresholds
                is_correct_01d = 1.0 if err < (0.1 * diameter) else 0.0
                is_correct_2cm = 1.0 if err < 0.02 else 0.0
                
                results[obj_id].append((is_correct_01d, is_correct_2cm, err))

    # Reporting
    header = f"{'OBJECT':<15} | {'SAMPLES':<8} | {'Acc (<2cm)':<12} | {'Acc (<0.1d)':<12} | {'Mean ADD error':<15}"
    print("\n" + "="*len(header))
    print(header)
    print("-" * len(header))

    global_results = []

    for obj_id in sorted(results.keys()):
        name = LINEMOD_ID_MAP.get(obj_id, str(obj_id))
        res_list = results[obj_id]
        
        successes_01d, successes_2cm, errors = zip(*res_list)
        acc_01d = np.mean(successes_01d) * 100
        acc_2cm = np.mean(successes_2cm) * 100
        mean_err = np.mean(errors)
        
        global_results.extend(res_list)
        print(f"{obj_id:>2} {name:<12} | {len(res_list):<8} | {acc_2cm:>11.2f}% | {acc_01d:>11.2f}% | {mean_err*1000:>13.2f}mm")

    print("="*len(header))
    
    if global_results:
        g_s_01d, g_s_2cm, g_errs = zip(*global_results)
        g_acc_01d = np.mean(g_s_01d) * 100
        g_acc_2cm = np.mean(g_s_2cm) * 100
        g_mean_err = np.mean(g_errs)
        
        print(f"{'AVERAGE':<15} | {len(global_results):<8} | {g_acc_2cm:>11.2f}% | {g_acc_01d:>11.2f}% | {g_mean_err*1000:>13.2f}mm")
    print("="*len(header))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse_model_path', type=str, required=True, help='Path to the trained Coarse Rgbd Fusion Net or Google Drive URL')
    parser.add_argument('--refine_model_path', type=str, required=True, help='Path to the trained Refine Net or Google Drive URL')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--iters', type=int, default=2, help='Number of Refinement Iterations')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    refine_net_evaluation(args.coarse_model_path, args.refine_model_path, device, args.batch_size, args.iters)