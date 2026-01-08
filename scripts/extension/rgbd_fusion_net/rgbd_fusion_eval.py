import os
import sys
import torch
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
from models.models_extension import RGBD_Fusion_Net
from dataset.dataset_baseline import LINEMOD_ID_MAP
from utils.process_dataset import load_meshes
from utils.evaluation_metrics import compute_add_metric, compute_adds_metric

def rgbd_fusion_net_evaluation(model_path, device='cpu', batch_size=32):
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
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load Meshes for ADD/ADD-S Calculation
    meshes = load_meshes(dataset_root)

    # Convert meshes from mm to meters
    for obj_id in meshes:
        if meshes[obj_id]['diameter'] > 10.0:
            meshes[obj_id]['vertices'] /= 1000.0
            meshes[obj_id]['diameter'] /= 1000.0

    results = {} 
    
    SYMMETRIC_IDS = [10, 11] 

    print(f"Evaluating on {len(val_set)} images...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Move to device
            rgb = batch['rgb'].to(device)
            points = batch['points'].to(device)
            gt_t = batch['gt_t'].cpu().numpy() # Needed in CPU for metric
            gt_R = batch['gt_R'].cpu().numpy()
            centroid = batch['centroid'].to(device)
            obj_ids = batch['obj_id'].numpy()

            # Forward Pass
            pred_q, pred_t_res = model(rgb, points)

            # Process Predictions (Ensure Float32 for metrics)
            pred_q = pred_q.float()
            pred_t_res = pred_t_res.float()

            pred_t_abs = (centroid + pred_t_res).cpu().numpy()
            pred_q = pred_q.cpu().numpy()

            for i in range(len(rgb)):
                obj_id = int(obj_ids[i])
                
                if obj_id not in meshes:
                    continue

                if obj_id not in results:
                    results[obj_id] = []

                mesh_pts = meshes[obj_id]['vertices']
                diameter = meshes[obj_id]['diameter']

                # Normalize predicted quaternion
                q = pred_q[i] / np.linalg.norm(pred_q[i])
                R_pred = R_scipy.from_quat(q).as_matrix()
                t_pred = pred_t_abs[i]

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
        
        # Unpack locally
        successes_01d, successes_2cm, errors = zip(*res_list)
        acc_01d = np.mean(successes_01d) * 100
        acc_2cm = np.mean(successes_2cm) * 100
        mean_err = np.mean(errors)
        
        # Add to global
        global_results.extend(res_list)

        # Mean Err converted to mm for readability: * 1000
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
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    rgbd_fusion_net_evaluation(args.model_path, device, batch_size=args.batch_size)