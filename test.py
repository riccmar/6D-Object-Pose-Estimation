from tqdm import tqdm

# Configuration
MODEL_SAVE_DIR = '/content/drive/MyDrive/Project/models/ResNet/best/'

# Best model of the MSE model
MSE_MODEL = 'best_e30_b32_lr0.0001_t20251216_202630_MSE.pth'
PATH_MSE_MODEL = os.path.join(MODEL_SAVE_DIR, MSE_MODEL)

# Best model of the Quat Loss model
QUAT_MODEL = 'best_e30_b64_lr1e-05_t20251217_092913_QUAT.pth'
PATH_QUAT_MODEL = os.path.join(MODEL_SAVE_DIR, QUAT_MODEL)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Models
print("Loading models...")
model_mse = load_model(PATH_MSE_MODEL)
model_quat = load_model(PATH_QUAT_MODEL)

if model_mse is None or model_quat is None:
    raise ValueError("One or both model paths are incorrect.")

# Evaluation loop
errors_mse = []
errors_quat = []

print(f"Evaluating on {len(val_loader.dataset)} validation images...")

with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.to(DEVICE)

        # Ground Truth (numpy)
        gt_quats = labels.numpy()

        # Model MSE
        pred_mse = model_mse(images).cpu().numpy()

        # Model Quat
        pred_quat = model_quat(images).cpu().numpy()

        # Calculate errors for this batch
        for i in range(len(gt_quats)):
            err_m = calculate_degree_error(pred_mse[i], gt_quats[i])
            err_q = calculate_degree_error(pred_quat[i], gt_quats[i])

            errors_mse.append(err_m)
            errors_quat.append(err_q)

# Final Results
mean_mse = np.mean(errors_mse)
mean_quat = np.mean(errors_quat)

median_mse = np.median(errors_mse)
median_quat = np.median(errors_quat)

print("\n" + "="*50)
print(f"{'METRIC':<20} | {'MSE MODEL':<12} | {'QUAT MODEL':<15}")
print("="*50)
print(f"{'Mean Error (°)':<20} | {f'{mean_mse:.2f}°':<12} | {f'{mean_quat:.2f}°':<15} ")
print("-" * 50)
print(f"{'Median Error (°)':<20} | {f'{median_mse:.2f}°':<12} | {f'{median_quat:.2f}°':<15} ")
print("="*50)

# Optional: Accuracy Metric (< 5 degrees)
acc_5_mse = np.mean(np.array(errors_mse) < 5.0) * 100
acc_5_quat = np.mean(np.array(errors_quat) < 5.0) * 100

print(f"Accuracy (< 5°):     | {f'{acc_5_mse:.1f}% ':<12} | {f'{acc_5_quat:.1f}%':<15} ")






import trimesh
from scipy.spatial.transform import Rotation as R_scipy

# --- 0. NAMES MAPPING ---
ID_TO_NAME = {
    1: 'Ape', 2: 'Benchvise', 3: 'Bowl', 4: 'Camera', 5: 'Can',
    6: 'Cat', 7: 'Cup', 8: 'Driller', 9: 'Duck', 10: 'Eggbox',
    11: 'Glue', 12: 'Holepuncher', 13: 'Iron', 14: 'Lamp', 15: 'Phone'
}

# --- 1. SETUP HELPERS ---
def load_meshes(dataset_root):
    """Loads 3D meshes for all objects in the dataset."""
    models_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'models')
    meshes = {}
    print("Pre-loading 3D models for ADD metric...")
    
    for filename in sorted(os.listdir(models_dir)):
        if filename.endswith(".ply") and filename.startswith("obj_"):
            try:
                obj_id = int(filename.split('_')[1].split('.')[0])
                ply_path = os.path.join(models_dir, filename)
                mesh = trimesh.load(ply_path)
                vertices = np.array(mesh.vertices)
                extents = vertices.max(axis=0) - vertices.min(axis=0)
                diameter = np.linalg.norm(extents)
                meshes[obj_id] = {'vertices': vertices, 'diameter': diameter}
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    return meshes

def compute_add_metric(pts, R_pred, R_gt):
    """Computes ADD metric assuming T_pred = T_gt."""
    pts_pred = np.dot(pts, R_pred.T)
    pts_gt = np.dot(pts, R_gt.T)
    dist = np.linalg.norm(pts_pred - pts_gt, axis=1)
    return np.mean(dist)

# --- 2. PREPARE DATA ---
# Re-init val_set with return_info=True
val_set_eval = RotationResNetDataset(DATA_ROOT, split='val', transform=val_transform, return_info=True)
val_loader_eval = DataLoader(val_set_eval, batch_size=1, shuffle=False, num_workers=4)
meshes = load_meshes(DATA_ROOT)

# --- 3. LOAD MODELS ---
MODEL_SAVE_DIR = '/content/drive/MyDrive/Project/models/ResNet/best/'
MSE_MODEL = 'best_e30_b32_lr0.0001_t20251216_202630_MSE.pth' 
QUAT_MODEL = 'best_e30_b64_lr1e-05_t20251217_092913_QUAT.pth'

print("Loading models...")
model_mse = load_model(os.path.join(MODEL_SAVE_DIR, MSE_MODEL))
model_quat = load_model(os.path.join(MODEL_SAVE_DIR, QUAT_MODEL))

# --- 4. STORAGE INITIALIZATION ---
# We store results globally AND per object
# Structure: {obj_id: {'mse': {'deg':[], 'add':[], ...}, 'quat': {...}}}
per_object_results = {}

print(f"Evaluating on {len(val_set_eval)} images...")

with torch.no_grad():
    for images, labels, obj_ids in tqdm(val_loader_eval):
        images = images.to(DEVICE)
        gt_quats = labels.numpy()
        obj_ids = obj_ids.numpy()

        pred_q_mse = model_mse(images).cpu().numpy()
        pred_q_quat = model_quat(images).cpu().numpy()

        for i in range(len(images)):
            obj_id = int(obj_ids[i])
            if obj_id not in meshes: continue
            
            # Initialize dict for this object if new
            if obj_id not in per_object_results:
                per_object_results[obj_id] = {
                    'mse': {'deg': [], 'add': [], 'acc': []},
                    'quat': {'deg': [], 'add': [], 'acc': []}
                }

            mesh_pts = meshes[obj_id]['vertices']
            diameter = meshes[obj_id]['diameter']

            # Normalize
            q_gt = gt_quats[i] / np.linalg.norm(gt_quats[i])
            q_m = pred_q_mse[i] / np.linalg.norm(pred_q_mse[i])
            q_q = pred_q_quat[i] / np.linalg.norm(pred_q_quat[i])

            # Degree Error
            d_m = np.clip(np.sum(q_m * q_gt), -1, 1)
            d_q = np.clip(np.sum(q_q * q_gt), -1, 1)
            if d_m < 0: d_m = -d_m # Double cover
            if d_q < 0: d_q = -d_q
            deg_m = 2 * np.arccos(d_m) * (180 / np.pi)
            deg_q = 2 * np.arccos(d_q) * (180 / np.pi)

            # ADD Error
            R_gt = R_scipy.from_quat(q_gt).as_matrix()
            R_m = R_scipy.from_quat(q_m).as_matrix()
            R_q = R_scipy.from_quat(q_q).as_matrix()

            add_m = compute_add_metric(mesh_pts, R_m, R_gt)
            add_q = compute_add_metric(mesh_pts, R_q, R_gt)

            # Store PER OBJECT
            per_object_results[obj_id]['mse']['deg'].append(deg_m)
            per_object_results[obj_id]['mse']['add'].append(add_m)
            per_object_results[obj_id]['mse']['acc'].append(1 if add_m < 0.1 * diameter else 0)

            per_object_results[obj_id]['quat']['deg'].append(deg_q)
            per_object_results[obj_id]['quat']['add'].append(add_q)
            per_object_results[obj_id]['quat']['acc'].append(1 if add_q < 0.1 * diameter else 0)

# --- 5. REPORTING ---

# Helper to calculate mean stats from a list
def calc_stats(data_dict):
    if not data_dict['deg']: return 0, 0, 0
    return np.mean(data_dict['deg']), np.mean(data_dict['add']), np.mean(data_dict['acc']) * 100

print("\n" + "="*110)
print(f"{'OBJECT':<12} | {'MSE MODEL (Deg | ADD | Acc)':<38} | {'QUAT MODEL (Deg | ADD | Acc)':<38}")
print("-" * 110)

# Global aggregators
global_mse = {'deg': [], 'add': [], 'acc': []}
global_quat = {'deg': [], 'add': [], 'acc': []}

for obj_id in sorted(per_object_results.keys()):
    name = ID_TO_NAME.get(obj_id, str(obj_id))
    data = per_object_results[obj_id]
    
    # Calculate Object Stats
    m_deg, m_add, m_acc = calc_stats(data['mse'])
    q_deg, q_add, q_acc = calc_stats(data['quat'])
    
    # Add to Global
    global_mse['deg'].extend(data['mse']['deg'])
    global_mse['add'].extend(data['mse']['add'])
    global_mse['acc'].extend(data['mse']['acc'])
    
    global_quat['deg'].extend(data['quat']['deg'])
    global_quat['add'].extend(data['quat']['add'])
    global_quat['acc'].extend(data['quat']['acc'])

    # Print Row
    print(f"{name:<12} | {m_deg:>5.1f}° {m_add:>6.1f}mm {m_acc:>5.0f}%        | {q_deg:>5.1f}° {q_add:>6.1f}mm {q_acc:>5.0f}%")

print("="*110)

# Print Global
g_m_deg, g_m_add, g_m_acc = calc_stats(global_mse)
g_q_deg, g_q_add