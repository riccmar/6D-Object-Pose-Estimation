import os
import sys
import cv2
import argparse
import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_baseline import BaselineDataset, LINEMOD_ID_MAP
from models.models_baseline import BaselinePoseSystem
from utils.process_dataset import load_meshes
from utils.visualization import precompute_bbox_corners, draw_pose
from utils.evaluation_metrics import compute_add_metric, compute_adds_metric

# Symmetric objects: 10 = Eggbox, 11 = Glue
SYMMETRIC_IDS = [10, 11]

def pipeline_inference(loader, pipeline, meshes, symmetric_ids, num_samples=3, conf=0.5):
    """Picks random samples and visualizes Gt vs Predicted Pose with Error Metric."""
    # Precompute BBOX corners for visualization
    meshes = precompute_bbox_corners(meshes)
    
    # Ensure num_samples doesn't exceed dataset size
    num_samples = min(num_samples, len(loader.dataset))
    indices = np.random.choice(len(loader.dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        sample = loader.dataset[idx]

        image_raw = sample['image']
        obj_id = sample['obj_id']
        gt_R = sample['gt_R']
        gt_t = sample['gt_t']
        cam_K = sample['cam_K']

        # Run Prediction
        real_height = meshes[obj_id]['height']
        # Pipeline expects numpy array
        pred_R, pred_t = pipeline.predict(image_raw, cam_K, real_height, conf=conf)

        if pred_R is None or pred_t is None:
            print(f"Skipping sample {idx}: Detection failed.")
            continue

        # Calculate error for this sample
        mesh_pts = meshes[obj_id]['vertices']
        is_symmetric = obj_id in symmetric_ids

        if is_symmetric:
            try:
                error_mm = compute_adds_metric(mesh_pts, gt_R, gt_t, pred_R, pred_t)
                metric_name = "ADD-S"
            except Exception as e:
                print(f"Error computing ADD-S for {obj_id}: {e}")
                error_mm = float('inf')
                metric_name = "ADD-S (Error)"
        else:
            error_mm = compute_add_metric(mesh_pts, gt_R, gt_t, pred_R, pred_t)
            metric_name = "ADD"

        # Create visualizations
        bbox_3d = meshes[obj_id]['bbox_3d']

        # Prepare image for OpenCV (RGB -> BGR)
        image_bgr = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)

        # Draw GT (Green), Pred (Yellow) then combine
        img_gt = draw_pose(image_bgr.copy(), cam_K, gt_R, gt_t, bbox_3d, label="Gt", color=(0, 255, 0))
        img_pred = draw_pose(image_bgr.copy(), cam_K, pred_R, pred_t, bbox_3d, label="Pred", color=(0, 255, 255))
        combined_img = np.hstack((img_gt, img_pred))

        # Add text overlay, draw a black rectangle at the top for legibility
        cv2.rectangle(combined_img, (0, 0), (combined_img.shape[1], 40), (0, 0, 0), -1)

        obj_name = LINEMOD_ID_MAP.get(obj_id, str(obj_id))
        text = f"Obj: {obj_id} {obj_name} | {metric_name} Error: {error_mm:.2f} mm"
        # Color text: Green if < 10% diameter, Red otherwise
        diameter = meshes[obj_id]['diameter']
        text_color = (0, 255, 0) if error_mm < (0.1 * diameter) else (0, 0, 255)

        cv2.putText(combined_img, text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # Save to disk
        output_dir = os.path.join(project_root, 'inference_results')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"inference_{idx}_{obj_id}.png")
        cv2.imwrite(save_path, combined_img)
        print(f"Saved result to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predictions from Baseline Pose Estimation System")
    parser.add_argument('--yolo_path', type=str, required=True, help='Path to YOLO weights')
    parser.add_argument('--resnet_path', type=str, required=True, help='Path to ResNet weights')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of random samples to visualize')
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
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load Meshes
    meshes = load_meshes(dataset_root)

    print(f"Inferencing and visualizing {args.num_samples} random samples...")
    pipeline_inference(val_loader, pipeline, meshes, SYMMETRIC_IDS, num_samples=args.num_samples, conf=args.yolo_conf)