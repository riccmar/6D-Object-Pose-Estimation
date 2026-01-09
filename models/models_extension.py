import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
from scipy.spatial.transform import Rotation as R

class RGBD_Fusion_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. RGB Branch (CNN)
        self.cnn = models.resnet18(pretrained=True)
        self.rgb_extractor = nn.Sequential(*list(self.cnn.children())[:-1]) # Output: (B, 512, 1, 1)

        # 2. Point Branch (PointNet)
        self.point_mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.point_mlp2 = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU()
        )

        # 3. Fusion Branch
        self.fusion_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU()
        )

        # Heads
        self.rot_head = nn.Linear(128, 4)   # Quaternion
        self.trans_head = nn.Linear(128, 3) # Residual Translation

    def forward(self, img, points):
        # 1. RGB Features
        rgb_feat = self.rgb_extractor(img) # (B, 512, 1, 1)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1) # (B, 512)

        # 2. Point Features
        p_feat = self.point_mlp1(points) # (B, N, 128)
        p_feat = torch.max(p_feat, 1)[0] # Global Max Pooling -> (B, 128)
        p_feat = self.point_mlp2(p_feat) # (B, 512)

        # 3. Fusion
        feat = torch.cat([rgb_feat, p_feat], dim=1) # (B, 1024)
        feat = self.fusion_mlp(feat)

        # 4. Output
        pred_q = self.rot_head(feat)
        pred_t_residual = self.trans_head(feat)

        return pred_q, pred_t_residual

class PoseRefineNet(nn.Module):
    # We process purely geometric data (point clouds) for pose refinement
    def __init__(self, num_points=500):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)

        # Delta Rotation Head
        self.rot_head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

        # Delta Translation Head
        self.trans_head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        # x: (B, 3, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Global Max Pooling
        x = torch.max(x, 2)[0]

        # Predict Deltas
        r = self.rot_head(x)
        t = self.trans_head(x)

        return r, t

class ExtensionPoseSystem:
    def __init__(self, yolo_path, pose_path, refine_path, class_map, device='cuda'):
        self.device = device
        self.num_points = 500
        self.class_map = class_map

        # Load YOLO (Segmentation)
        if os.path.exists(yolo_path):
            self.yolo = YOLO(yolo_path)
            self.yolo.to(self.device)
            print(f"YOLO loaded from {yolo_path}")
        else:
            raise FileNotFoundError(f"YOLO path not found: {yolo_path}")

        # Load Pose Model
        self.pose_model = RGBD_Fusion_Net().to(self.device)
        self._load_weights(self.pose_model, pose_path, "Coarse")

        # Load Refine Model
        self.refine_model = PoseRefineNet().to(self.device)
        self._load_weights(self.refine_model, refine_path, "Refine")

        # Preprocessing Transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_weights(self, model, path, name):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            print(f"{name} Model loaded from {path}")
        else:
            print(f"Warning: {name} path not found: {path}")

    def run(self, rgb_path, depth_path, cam_K, depth_scale=1.0, refine_iters=2, yolo_conf=0.25, target_obj_id=None):
        # Load Data
        rgb_pil = Image.open(rgb_path).convert("RGB")
        depth = cv2.imread(depth_path, -1)
        if depth is None: 
            return None, None

        w_img, h_img = rgb_pil.size

        # YOLO Inference
        results = self.yolo(rgb_pil, verbose=False, retina_masks=True, conf=yolo_conf)

        target_cls = None
        if target_obj_id in self.class_map:
            target_cls = self.class_map[target_obj_id]
        
        best_box = None
        best_mask = None
        max_conf = -1

        # Check if we detected anything
        if not results or not results[0].boxes:
            return None, None

        # Iterate through detections to find the target object with highest confidence
        boxes = results[0].boxes
        masks = results[0].masks
        
        if masks is None: 
            return None, None

        for i, box in enumerate(boxes):
            # Check Class
            if target_cls is not None:
                if int(box.cls[0]) != target_cls:
                    continue

            # Check Confidence
            if box.conf > max_conf:
                max_conf = float(box.conf)
                best_box = box.xyxy[0].cpu().numpy().astype(int)

                # Extract Mask (H, W) corresponding to this box
                best_mask = masks.data[i].cpu().numpy()

        if best_box is None or best_mask is None: 
            return None, None

        x1, y1, x2, y2 = best_box

        # Process Mask & Crop
        if best_mask.shape != (h_img, w_img):
            best_mask = cv2.resize(best_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        binary_mask = (best_mask > 0.5).astype(np.uint8)

        # Clamp box coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            return None, None

        # Crop RGB
        rgb_crop = rgb_pil.crop((x1, y1, x2, y2))
        rgb_tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)

        # Point Cloud Input
        depth_crop = depth[y1:y2, x1:x2]
        mask_crop = binary_mask[y1:y2, x1:x2]

        # Combine Depth with Yolo Mask
        valid_indices = (depth_crop > 0) & (mask_crop > 0)
        ys, xs = np.nonzero(valid_indices)

        if len(ys) == 0: 
            return None, None

        # Random Sample
        replace = len(ys) < self.num_points
        choice = np.random.choice(len(ys), self.num_points, replace=replace)
        ys, xs = ys[choice], xs[choice]

        depth_vals = depth_crop[ys, xs] * depth_scale
        if np.mean(depth_vals) > 100: depth_vals *= 0.001 

        # Back-project
        u, v = xs + x1, ys + y1
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        fx, fy = cam_K[0, 0], cam_K[1, 1]

        z_3d = depth_vals.astype(np.float32)
        x_3d = (u - cx) * z_3d / fx
        y_3d = (v - cy) * z_3d / fy
        cloud = np.stack([x_3d, y_3d, z_3d], axis=1)

        centroid = np.mean(cloud, axis=0)
        cloud = cloud - centroid

        points_tensor = torch.tensor(cloud, dtype=torch.float32).unsqueeze(0).to(self.device)
        centroid_tensor = torch.tensor(centroid, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Run Inference
        with torch.no_grad():
            # Coarse Pass
            pred_q, pred_t_res = self.pose_model(rgb_tensor, points_tensor)
            pred_t_abs = centroid_tensor + pred_t_res
            pred_q = F.normalize(pred_q, p=2, dim=1)

            # Refinement Loop
            points_T = points_tensor.transpose(1, 2)
            
            for _ in range(refine_iters):
                b = pred_q.shape[0]
                x, y, z, w = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2], pred_q[:, 3]
                pred_R_torch = torch.stack([
                    1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
                    2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
                    2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
                ], dim=1).reshape(b, 3, 3)

                points_centered = points_T - (pred_t_abs - centroid_tensor).unsqueeze(2)
                points_local = torch.bmm(pred_R_torch.transpose(1, 2), points_centered)

                delta_q, delta_t = self.refine_model(points_local)

                t_update = torch.bmm(pred_R_torch, delta_t.unsqueeze(2)).squeeze(2)
                pred_t_abs = pred_t_abs + t_update
                pred_q = F.normalize(pred_q + delta_q, p=2, dim=1)

            pred_t = pred_t_abs.cpu().numpy()[0]
            pred_q_np = pred_q.cpu().numpy()[0]
            pred_R = R.from_quat(pred_q_np).as_matrix()

            return pred_R, pred_t