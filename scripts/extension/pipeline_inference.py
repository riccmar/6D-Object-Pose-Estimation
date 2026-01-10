import os
import sys
import argparse
import gdown
import numpy as np
import cv2
import torch
import random
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_extension import ExtensionPipelineDataset
from models.models_extension import ExtensionPoseSystem
from utils.process_dataset import load_meshes
from utils.visualization import precompute_bbox_corners, draw_pose, project_dense_mesh
from utils.evaluation_metrics import calculate_degree_error
from scipy.spatial.transform import Rotation as R

def pipeline_inference(pipeline, dataset, meshes,  obj_id, yolo_conf=0.25, refine_iters=2):
    # Find all samples for the target object
    samples = [idx for idx, s in enumerate(dataset.samples) if int(s['obj_id']) == obj_id]
    
    if not samples:
        print(f"No samples found for object ID {obj_id}")
        return

    # Pick a random sample
    idx = random.choice(samples)
    
    sample = dataset.samples[idx]
    
    rgb_path = sample['rgb_path']
    depth_path = sample['depth_path']
    cam_K = sample['cam_K']
    gt_R = sample['cam_R']
    gt_t = sample['cam_t']
    depth_scale = sample['depth_scale']

    print(f"Processing sample index {idx} for object {obj_id}...")

    # Run Pipeline
    res = pipeline.run(
        rgb_path, depth_path, cam_K, 
        depth_scale=depth_scale, 
        refine_iters=refine_iters, 
        yolo_conf=yolo_conf,
        target_obj_id=obj_id
    )

    if res[0] is None:
        print("Detection failed! No object detected.")
        return

    pred_R, pred_t = res

    # Calculate Metrics
    gt_q = R.from_matrix(gt_R).as_quat()
    pred_q = R.from_matrix(pred_R).as_quat()
    
    t_err_m = np.linalg.norm(gt_t - pred_t)
    r_err_deg = calculate_degree_error(pred_q, gt_q)

    print("\n" + "="*60)
    print(f"METRICS REPORT | Object ID: {obj_id}")
    print("="*60)
    print("TRANSLATION (meters) [x, y, z]:")
    print(f"   GT:   [{gt_t[0]:.4f}, {gt_t[1]:.4f}, {gt_t[2]:.4f}]")
    print(f"   Pred: [{pred_t[0]:.4f}, {pred_t[1]:.4f}, {pred_t[2]:.4f}]")
    print(f"   Error: {t_err_m*100:.2f} cm")
    print("-" * 60)
    print("ROTATION (quaternions) [x, y, z, w]:")
    print(f"   GT:   [{gt_q[0]:.3f}, {gt_q[1]:.3f}, {gt_q[2]:.3f}, {gt_q[3]:.3f}]")
    print(f"   Pred: [{pred_q[0]:.3f}, {pred_q[1]:.3f}, {pred_q[2]:.3f}, {pred_q[3]:.3f}]")
    print(f"   Error: {r_err_deg:.2f}°")
    print("="*60 + "\n")

    # Load original image
    img = cv2.imread(rgb_path)
    
    # 1. Image with GT 3D Bounding Box (Green)
    img_gt = img.copy()
    mesh_data = meshes[obj_id]
    bbox_3d = mesh_data.get('bbox_3d')
    
    img_gt = draw_pose(img_gt, cam_K, gt_R, gt_t*1000, bbox_3d=bbox_3d, label=f"GT Obj {obj_id}", color=(0, 255, 0)) # Green

    # 2. Image with Predicted 3D Bounding Box (Blue)
    img_pred = img.copy()
    img_pred = draw_pose(img_pred, cam_K, pred_R, pred_t*1000, bbox_3d=bbox_3d, label=f"Pred Obj {obj_id}", color=(255, 0, 0)) # Blue

    # 3. Image with Mesh
    img_mesh = img.copy()
    vertices = meshes[obj_id]['vertices']
    img_mesh = project_dense_mesh(img_mesh, vertices, cam_K, pred_R, pred_t*1000, color=(0, 0, 255)) # Red

    # Save Images
    output_dir = os.path.join(project_root, 'inference_results')
    os.makedirs(output_dir, exist_ok=True)

    img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
    img_mesh_rgb = cv2.cvtColor(img_mesh, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(img_gt_rgb)
    ax[0].set_title(f"Ground Truth (Obj {obj_id})")
    ax[0].axis('off')

    ax[1].imshow(img_pred_rgb)
    ax[1].set_title("Prediction")
    ax[1].axis('off')

    ax[2].imshow(img_mesh_rgb)
    ax[2].set_title("Dense Mesh Projection")
    ax[2].axis('off')

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"obj_{obj_id}_pipeline_result.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_path', type=str, required=True, help='Path to YOLO weights')
    parser.add_argument('--rgbdfusion_path', type=str, required=True, help='Path to Rgbd Fusion weights')
    parser.add_argument('--refine_path', type=str, required=True, help='Path to Refine weights')
    parser.add_argument('--yolo_conf', type=float, default=0.25, help='Confidence threshold for YOLO')
    parser.add_argument('--iters', type=int, default=2, help='Number of refinement iterations')
    parser.add_argument('--obj_id', type=int, default=None, help='Object ID to test')

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

    # Dynamically generate Class Mapping (same as pipeline_eval.py)
    data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')
    if os.path.exists(data_dir):
        existing_obj_ids = sorted([int(x) for x in os.listdir(data_dir) if x.isdigit()])
        obj_id_to_class_id = {oid: i for i, oid in enumerate(existing_obj_ids)}
    else:
        obj_id_to_class_id = {
            1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 9: 6, 
            10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
        }

    if args.obj_id is None:
        args.obj_id = random.choice(list(obj_id_to_class_id.keys()))
        print(f"No object ID provided. Randomly selected Object ID: {args.obj_id}")
    elif args.obj_id not in obj_id_to_class_id:
        print(f"Error: Object ID {args.obj_id} is not valid. Available IDs: {sorted(list(obj_id_to_class_id.keys()))}")
        sys.exit(1)

    # Initialize System
    pipeline = ExtensionPoseSystem(args.yolo_path, args.rgbdfusion_path, args.refine_path, obj_id_to_class_id, device)

    # Initialize Dataset
    val_dataset = ExtensionPipelineDataset(dataset_root, split='val')

    # Load Meshes (for bbox)
    meshes = load_meshes(dataset_root)
    meshes = precompute_bbox_corners(meshes)

    pipeline_inference(pipeline, val_dataset, meshes, args.obj_id, yolo_conf=args.yolo_conf, refine_iters=args.iters)
