import os
import sys
import torch
import numpy as np
import argparse
import gdown
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R_scipy
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_baseline import RotationResNetDataset, LINEMOD_ID_MAP
from models.models_baseline import load_rotationresnet_model
from utils.process_dataset import load_meshes
from utils.evaluation_metrics import calculate_degree_error, compute_add_metric_rotation_only, calc_stats_rotation_only

def resnet_evaluation(model_path, device='cpu', batch_size=32):
    # Setup
    dataset_root = os.path.join(project_root, 'data/linemod')
    
    # Handle Google Drive URL
    if model_path.startswith('http'):
        print(f"Model path detected as URL. Downloading from Google Drive...")
        
        checkpoints_dir = os.path.join(project_root, 'checkpoints', 'resnet')
        os.makedirs(checkpoints_dir, exist_ok=True)

        download_path = os.path.join(checkpoints_dir, 'resnet_downloaded.pth')
        gdown.download(model_path, download_path, quiet=False, fuzzy=True)
        model_path = download_path
        print(f"Model downloaded to: {model_path}")

    # Load Model
    model = load_rotationresnet_model(model_path, device)
    if model is None:
        sys.exit(1)

    # Prepare Dataset
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Initializing Validation Dataset...")
    val_set = RotationResNetDataset(dataset_root, split='val', transform=val_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load Meshes
    meshes = load_meshes(dataset_root)

    # Storage for results
    per_object_results = {}
    
    # Evaluation loop
    print(f"Evaluating on {len(val_set)} images...")
    
    with torch.no_grad():
        for images, labels, obj_ids in tqdm(val_loader):
            images = images.to(device)
            gt_quat = labels.numpy()
            obj_ids = obj_ids.numpy()

            pred_q = model(images).cpu().numpy()

            for i in range(len(images)):
                obj_id = obj_ids[i]
                
                # Skip if mesh missing
                if obj_id not in meshes:
                    continue

                if obj_id not in per_object_results:
                    per_object_results[obj_id] = {'deg': [], 'add': [], 'acc': []}

                mesh_pts = meshes[obj_id]['vertices']
                diameter = meshes[obj_id]['diameter']

                # Normalize Quaternions
                q_gt = gt_quat[i] / np.linalg.norm(gt_quat[i])
                q_pred = pred_q[i] / np.linalg.norm(pred_q[i])

                # Degree Error
                deg_err = calculate_degree_error(q_pred, q_gt)

                # ADD Error
                R_gt = R_scipy.from_quat(q_gt).as_matrix()
                R_pred = R_scipy.from_quat(q_pred).as_matrix()
                
                add_err = compute_add_metric_rotation_only(mesh_pts, R_pred, R_gt)
                
                # Accuracy (< 10% Diameter)
                is_accurate = 1 if add_err < 0.1 * diameter else 0

                per_object_results[obj_id]['deg'].append(deg_err)
                per_object_results[obj_id]['add'].append(add_err)
                per_object_results[obj_id]['acc'].append(is_accurate)

    # Print Results
    print("\n" + "="*80)
    print(f"{'OBJECT':<15} | {'SAMPLES':<7} | {'RESULTS (Deg | ADD | Acc)':<30}")
    print("-" * 80)

    global_stats = {'deg': [], 'add': [], 'acc': []}

    for obj_id in sorted(per_object_results.keys()):
        name = LINEMOD_ID_MAP.get(obj_id, str(obj_id))
        data = per_object_results[obj_id]
        
        num_samples = len(data['deg'])
        m_deg, m_add, m_acc = calc_stats_rotation_only(data)

        # Add to global
        global_stats['deg'].extend(data['deg'])
        global_stats['add'].extend(data['add'])
        global_stats['acc'].extend(data['acc'])

        print(f"{obj_id:>2} {name:<12} | {num_samples:<7} | {m_deg:>5.1f}° {m_add:>6.1f}mm {m_acc:>5.0f}%")

    print("="*80)
    g_deg, g_add, g_acc = calc_stats_rotation_only(global_stats)
    print(f"{'AVERAGE':<15} | {len(global_stats['deg']):<7} | {g_deg:>5.1f}° {g_add:>6.1f}mm {g_acc:>5.0f}%")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    resnet_evaluation(args.model_path, device, batch_size=args.batch_size)