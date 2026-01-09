import os
import sys
import argparse
import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_baseline import BaselineDataset, LINEMOD_ID_MAP
from models.models_baseline import BaselinePoseSystem
from utils.process_dataset import load_meshes
from utils.evaluation_metrics import compute_add_metric, compute_adds_metric, calc_stats, compute_angular_error

# Symmetric objects: 10 = Eggbox, 11 = Glue
SYMMETRIC_IDS = [10, 11]

def print_report(global_metrics, per_object_metrics, detection_failures, total_frames, symmetric_ids=SYMMETRIC_IDS):
    # Global Stats
    g_acc_rot, g_err_rot, g_med_rot, g_acc2cm_rot, g_rot_err_rot = calc_stats(global_metrics['rot'])
    g_acc_trans, g_err_trans, g_med_trans, g_acc2cm_trans, g_rot_err_trans = calc_stats(global_metrics['trans'])
    g_acc_full, g_err_full, g_med_full, g_acc2cm_full, g_rot_err_full = calc_stats(global_metrics['full'])

    fail_rate = (detection_failures / total_frames) * 100 if total_frames > 0 else 0

    print("\n" + "="*135)
    print("GLOBAL RESULTS (All Objects Averaged)")
    print("="*135)
    print(f"Total Samples Processed: {total_frames}")
    print(f"{'Metric Component':<30} | {'ADD-0.1d':<10} | {'Acc < 2cm':<10} | {'Mean ADD':<11} | {'Median ADD':<12} | {'Mean Rot Err':<12}")
    print("-" * 135)
    print(f"{'1. Rotation Only (ResNet)':<30} | {g_acc_rot:>6.2f}%    | {g_acc2cm_rot:>6.2f}%    | {g_err_rot:>8.2f} mm | {g_med_rot:>8.2f} mm | {g_rot_err_rot:>8.2f} deg")
    print(f"{'2. Translation Only (Pinhole)':<30} | {g_acc_trans:>6.2f}%    | {g_acc2cm_trans:>6.2f}%    | {g_err_trans:>8.2f} mm | {g_med_trans:>8.2f} mm | {g_rot_err_trans:>8.2f} deg")
    print(f"{'3. Full Baseline System':<30} | {g_acc_full:>6.2f}%    | {g_acc2cm_full:>6.2f}%    | {g_err_full:>8.2f} mm | {g_med_full:>8.2f} mm | {g_rot_err_full:>8.2f} deg")
    print("-" * 135)
    print(f"Detection Failures: {detection_failures}/{total_frames} ({fail_rate:.1f}%)")
    print("="*135)

    # Per Object Stats
    print("\n" + "="*165)
    print(f"PER-OBJECT BREAKDOWN")
    print("="*165)
    print(f"{'ID':<4} {'Name':<12} | {'Count':<6} | {'Rot Only (ResNet)':<38} | {'Trans Only (Pinhole)':<38} | {'Baseline (Full)':<38} | {'Fail'}")
    print(f"{'':<4} {'':<12} | {'':<6} | {'Acc<0.1d':<8} {'Mean(mm)':<9} {'Acc<2cm':<8} {'Rot(°)':<7} | {'Acc<0.1d':<8} {'Mean(mm)':<9} {'Acc<2cm':<8} {'Rot(°)':<7} | {'Acc<0.1d':<8} {'Mean(mm)':<9} {'Acc<2cm':<8} {'Rot(°)':<7} |")
    print("-" * 165)

    for obj_id in sorted(per_object_metrics.keys()):
        data = per_object_metrics[obj_id]
        obj_name = LINEMOD_ID_MAP.get(obj_id, "Unknown")
        is_sym = "*" if obj_id in symmetric_ids else " "

        # Add star to name if symmetric
        disp_name = f"{obj_name}{is_sym}"

        acc_rot, err_rot, _, acc2cm_rot, rot_err_rot = calc_stats(data['rot'])
        acc_trans, err_trans, _, acc2cm_trans, rot_err_trans = calc_stats(data['trans'])
        acc_full, err_full, _, acc2cm_full, rot_err_full = calc_stats(data['full'])

        fails = data['failures']
        total = data['count']
        fail_p = (fails/total)*100 if total else 0

        print(f"{obj_id:<4} {disp_name:<12} | {total:<6} | "
                f"{acc_rot:>6.1f}%  {err_rot:>6.1f}mm  {acc2cm_rot:>6.1f}%  {rot_err_rot:>5.1f}° | "
                f"{acc_trans:>6.1f}%  {err_trans:>6.1f}mm  {acc2cm_trans:>6.1f}%  {rot_err_trans:>5.1f}° | "
                f"{acc_full:>6.1f}%  {err_full:>6.1f}mm  {acc2cm_full:>6.1f}%  {rot_err_full:>5.1f}° | "
                f"{fail_p:.0f}%")
    print("="*165)
    print("(* indicates ADD-S metric was used)")

def pipeline_evaluation(pipeline, val_loader, meshes, conf_threshold=0.5, symmetric_ids=SYMMETRIC_IDS):
    """
    Evaluates the full Baseline Pose Estimation Pipeline on the provided DataLoader.
    """
    # Storage structure: {obj_id: {'rot': [], 'trans': [], 'full': [], 'failures': 0, 'count': 0}}
    per_object_metrics = {}
    global_metrics = {'rot': [], 'trans': [], 'full': []}

    detection_failures = 0
    total_frames = 0

    print(f"Starting Evaluation on {len(val_loader.dataset)} frames...\n")

    for batch in tqdm(val_loader):
        # Unpack Batch
        images = batch['image']
        obj_ids = batch['obj_id']
        gt_Rs = batch['gt_R']
        gt_ts = batch['gt_t']
        cam_Ks = batch['cam_K']

        # Determine current batch size
        current_batch_size = images.shape[0]

        for i in range(current_batch_size):
            full_image_np = images[i].numpy()
            obj_id = int(obj_ids[i].item())
            gt_R = gt_Rs[i].numpy()
            gt_t = gt_ts[i].numpy()
            cam_K = cam_Ks[i].numpy()

            # Initialize dict entry
            if obj_id not in per_object_metrics:
                per_object_metrics[obj_id] = {
                    'rot': [],
                    'trans': [],
                    'full': [],
                    'failures': 0,
                    'count': 0
                }

            per_object_metrics[obj_id]['count'] += 1
            total_frames += 1

            # Mesh Info
            mesh_pts = meshes[obj_id]['vertices']
            diameter = meshes[obj_id]['diameter']
            threshold = 0.1 * diameter
            real_height = meshes[obj_id]['height']

            # Run Pipeline
            pred_R, pred_t = pipeline.predict(full_image_np, cam_K, real_height, conf=conf_threshold)

            if pred_R is None or pred_t is None:
                detection_failures += 1
                per_object_metrics[obj_id]['failures'] += 1
                continue

            # Metric computation
            is_symmetric = obj_id in symmetric_ids

            def calc_error(R_est, t_est, R_ref, t_ref):
                if is_symmetric:
                    return compute_adds_metric(mesh_pts, R_ref, t_ref, R_est, t_est)
                else:
                    return compute_add_metric(mesh_pts, R_ref, t_ref, R_est, t_est)

            # 1. Rotation Only (ResNet + Gt Translation)
            # We test ResNet module by giving it the cheat of perfect translation
            err_rot = calc_error(pred_R, gt_t, gt_R, gt_t)
            res_rot = 1 if err_rot < threshold else 0
            res_2cm_rot = 1 if err_rot < 20.0 else 0
            rot_error_rot = compute_angular_error(gt_R, pred_R)

            per_object_metrics[obj_id]['rot'].append((res_rot, res_2cm_rot, err_rot, rot_error_rot))
            global_metrics['rot'].append((res_rot, res_2cm_rot, err_rot, rot_error_rot))

            # 2. Translation Only (Gt Quaternion + Pinhole)
            # We test Pinhole module by giving it the cheat of perfect rotation
            err_trans = calc_error(gt_R, pred_t, gt_R, gt_t)
            res_trans = 1 if err_trans < threshold else 0
            res_2cm_trans = 1 if err_trans < 20.0 else 0
            rot_error_trans = 0.0

            per_object_metrics[obj_id]['trans'].append((res_trans, res_2cm_trans, err_trans, rot_error_trans))
            global_metrics['trans'].append((res_trans, res_2cm_trans, err_trans, rot_error_trans))

            # 3. Full Baseline (ResNet + Pinhole)
            # No cheats here, full prediction
            err_full = calc_error(pred_R, pred_t, gt_R, gt_t)
            res_full = 1 if err_full < threshold else 0
            res_2cm_full = 1 if err_full < 20.0 else 0
            rot_error_full = compute_angular_error(gt_R, pred_R)

            per_object_metrics[obj_id]['full'].append((res_full, res_2cm_full, err_full, rot_error_full))
            global_metrics['full'].append((res_full, res_2cm_full, err_full, rot_error_full))
    
    print_report(global_metrics, per_object_metrics, detection_failures, total_frames, symmetric_ids)

    return global_metrics, per_object_metrics, detection_failures, total_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_path', type=str, required=True, help='Path to YOLO weights')
    parser.add_argument('--resnet_path', type=str, required=True, help='Path to ResNet weights')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--yolo_conf', type=float, default=0.25, help='Confidence threshold for YOLO detection')

    args = parser.parse_args()

    # Handle Google Drive URLs
    if args.yolo_path.startswith('http'):
        print("YOLO model path detected as URL. Downloading YOLO model from Drive...")

        dest_dir = os.path.join(project_root, 'checkpoints', 'yolo')
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, 'yolo_downloaded.pt')
        gdown.download(args.yolo_path, dest_path, quiet=False, fuzzy=True)
        args.yolo_path = dest_path
        print(f"YOLO model downloaded to: {args.yolo_path}")

    if args.resnet_path.startswith('http'):
        print("ResNet model path detected as URL. Downloading ResNet model from Drive...")

        dest_dir = os.path.join(project_root, 'checkpoints', 'resnet')
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, 'resnet_downloaded.pth')
        gdown.download(args.resnet_path, dest_path, quiet=False, fuzzy=True)
        args.resnet_path = dest_path
        print(f"ResNet model downloaded to: {args.resnet_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset_root = os.path.join(project_root, 'data', 'linemod')

    # Initialize System
    pipeline = BaselinePoseSystem(args.yolo_path, args.resnet_path, device)

    # Initialize Dataset
    val_dataset = BaselineDataset(dataset_root, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load Meshes
    meshes = load_meshes(dataset_root)

    pipeline_evaluation(pipeline, val_loader, meshes, conf_threshold=args.yolo_conf)