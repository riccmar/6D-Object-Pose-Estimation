import os
import sys
import argparse
import gdown
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_extension import RgbdFusionNetDataset, ExtensionPipelineDataset
from models.models_extension import ExtensionPoseSystem
from utils.process_dataset import load_meshes
from utils.evaluation_metrics import compute_add_metric, compute_adds_metric, calc_stats_ext, compute_angular_error
from dataset.dataset_baseline import LINEMOD_ID_MAP

# Symmetric objects: 10 = Eggbox, 11 = Glue
SYMMETRIC_IDS = [10, 11]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Seed set to: {seed}")

def print_report(global_metrics, per_object_metrics, detection_failures, total_frames, symmetric_ids=SYMMETRIC_IDS):
    # Global Stats
    g_acc_01d, g_mean_err, g_med_err, g_acc_2cm, g_mean_rot = calc_stats_ext(global_metrics['full'])
    fail_rate = (detection_failures / total_frames) * 100 if total_frames > 0 else 0

    print("\n" + "="*100)
    print("GLOBAL RESULTS (Extension Pipeline)")
    print("="*100)
    print(f"Total Samples Processed: {total_frames}")
    print(f"Detection Failures: {detection_failures}/{total_frames} ({fail_rate:.1f}%)")
    print(f"Accuracy (< 2cm):    {g_acc_2cm:.2f}%")
    print(f"Accuracy (< 0.1d):   {g_acc_01d:.2f}%")
    print(f"Mean Error: {g_mean_err:.2f} mm")
    print(f"Median Error: {g_med_err:.2f} mm")
    print(f"Mean Rot Error: {g_mean_rot:.2f}°")
    print("="*100)

    # Per Object Stats
    print("\n" + "="*100)
    print(f"PER-OBJECT BREAKDOWN")
    print("(* = Symmetric Object, using ADD-S metric)")
    print("="*100)
    print(f"{'ID':<4} {'Name':<12} | {'Count':<6} | {'Acc(<2cm)':>10} {'Acc(<0.1d)':>10} {'Mean(mm)':>10} {'Rot(°)':>9} | {'Fail':>6}")
    print("-" * 100)

    for obj_id in sorted(per_object_metrics.keys()):
        data = per_object_metrics[obj_id]
        obj_name = LINEMOD_ID_MAP.get(obj_id, "Unknown")
        is_sym = "*" if obj_id in symmetric_ids else " "
        disp_name = f"{obj_name}{is_sym}"

        acc_01d, err_full, _, acc_2cm, rot_err = calc_stats_ext(data['full'])
        fails = data['failures']
        total = data['count']
        fail_p = (fails/total)*100 if total else 0

        print(f"{obj_id:<4} {disp_name:<12} | {total:<6} | "
                f"{acc_2cm:>9.1f}% {acc_01d:>9.1f}% {err_full:>8.1f}mm {rot_err:>8.1f}° | {fail_p:>5.0f}%")
    print("="*100)

def pipeline_evaluation(pipeline, val_loader, meshes, yolo_conf=0.25, refine_iters=2, symmetric_ids=SYMMETRIC_IDS):
    per_object_metrics = {}
    global_metrics = {'full': []}
    detection_failures = 0
    total_frames = 0

    print(f"Starting Evaluation on {len(val_loader.dataset)} frames...\n")
    
    for batch in tqdm(val_loader):
        # Extract batch data
        rgb_paths = batch['rgb_path']
        depth_paths = batch['depth_path']
        obj_ids = batch['obj_id']
        cam_Ks = batch['cam_K']
        gt_t_batch = batch['gt_t']
        gt_R_batch = batch['gt_R']
        
        # Iterate through batch items
        current_bs = len(rgb_paths)

        for i in range(current_bs):
            rgb_path = rgb_paths[i]
            depth_path = depth_paths[i]
            obj_id = int(obj_ids[i])
            cam_K = cam_Ks[i].numpy()
            gt_t = gt_t_batch[i].numpy()
            gt_R = gt_R_batch[i].numpy()
            
            # Handle depth_scale
            depth_scale = 1.0
            if 'depth_scale' in batch:
                depth_scale = float(batch['depth_scale'][i])

            if obj_id not in per_object_metrics:
                per_object_metrics[obj_id] = {'full': [], 'failures': 0, 'count': 0}

            per_object_metrics[obj_id]['count'] += 1
            total_frames += 1

            # Run Pipeline
            pred_R, pred_t = pipeline.run(
                rgb_path, depth_path, cam_K, 
                depth_scale=depth_scale, 
                refine_iters=refine_iters, 
                yolo_conf=yolo_conf,
                target_obj_id=obj_id,
            )

            if pred_R is None or pred_t is None:
                detection_failures += 1
                per_object_metrics[obj_id]['failures'] += 1
                continue

            # Metric computation
            mesh_pts = meshes[obj_id]['vertices']
            diameter = meshes[obj_id]['diameter']
            threshold = 0.1 * diameter
            
            is_symmetric = obj_id in symmetric_ids

            if is_symmetric:
                err = compute_adds_metric(mesh_pts, gt_R, gt_t, pred_R, pred_t)
            else:
                err = compute_add_metric(mesh_pts, gt_R, gt_t, pred_R, pred_t)
            
            rot_err = compute_angular_error(gt_R, pred_R)
                
            res_01d = 1 if err < threshold else 0

            res_2cm = 1 if err < 0.02 else 0

            per_object_metrics[obj_id]['full'].append((res_01d, res_2cm, err, rot_err))
            global_metrics['full'].append((res_01d, res_2cm, err, rot_err))

    print_report(global_metrics, per_object_metrics, detection_failures, total_frames, symmetric_ids)

if __name__ == "__main__":
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_path', type=str, required=True, help='Path to the trained YOLO or Google Drive URL')
    parser.add_argument('--rgbdfusion_path', type=str, required=True, help='Path to the trained Rgbd Fusion Model or Google Drive URL')
    parser.add_argument('--refine_path', type=str, required=True, help='Path to the trained Refine Model or Google Drive URL')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--yolo_conf', type=float, default=0.25, help='Confidence threshold for YOLO')
    parser.add_argument('--iters', type=int, default=2, help='Number of refinement iterations')

    args = parser.parse_args()

    # Handle Downloads
    path_map = {'Yolo': args.yolo_path, 'Rgbd_Fusion': args.rgbdfusion_path, 'Refine': args.refine_path}
    for name, path in path_map.items():
        if path.startswith('http'):
            print(f"{name} model path detected as URL. Downloading...")

            dest_dir = os.path.join(project_root, 'checkpoints')
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, f'{name}_downloaded.pt' if name=='Yolo' else f'{name}_downloaded.pth')
            gdown.download(path, dest_path, quiet=False, fuzzy=True)
            if name == 'Yolo': 
                args.yolo_path = dest_path
            elif name == 'Rgbd_Fusion': 
                args.rgbdfusion_path = dest_path
            elif name == 'Refine': 
                args.refine_path = dest_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset_root = os.path.join(project_root, 'data', 'linemod')

    # Dynamically generate Class Mapping
    data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')
    if os.path.exists(data_dir):
        existing_obj_ids = sorted([int(x) for x in os.listdir(data_dir) if x.isdigit()])
        obj_id_to_class_id = {oid: i for i, oid in enumerate(existing_obj_ids)}
        print(f"Class Mapping loaded: {obj_id_to_class_id}")
    else:
        print("Warning: Data directory not found, using default class map.")
        obj_id_to_class_id = {
            1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 9: 6, 
            10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
        }

    # Initialize System
    pipeline = ExtensionPoseSystem(args.yolo_path, args.rgbdfusion_path, args.refine_path, obj_id_to_class_id, device)

    # Initialize Dataset
    val_dataset = ExtensionPipelineDataset(dataset_root, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load Meshes
    meshes = load_meshes(dataset_root)
    for obj_id in meshes:
        meshes[obj_id]['vertices'] /= 1000.0
        meshes[obj_id]['diameter'] /= 1000.0

    pipeline_evaluation(pipeline, val_loader, meshes, yolo_conf=args.yolo_conf, refine_iters=args.iters)