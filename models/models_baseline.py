import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

# Rotation Module (ResNet-based)
class RotationResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(RotationResNet, self).__init__()
        self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.backbone(x)
        # x = F.normalize(x, p=2, dim=1)

        return x

def load_rotationresnet_model(path, device, pretrained=False):
    """
    Utility to load RotationResNet from a checkpoint path.
    """
    model = RotationResNet(pretrained=pretrained)
    
    if not os.path.exists(path):
        print(f"Error: Could not find model at {path}")
        return None
        
    print(f"Loading RotationResNet from {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("RotationResNet loaded successfully and set to eval mode.")

    return model

# Translation Module (Pinhole Camera Model)
class PinholeTranslationLayer:
    def __init__(self):
        pass

    def forward(self, bbox, cam_K, real_height):
        # Calculates 3D translation using Pinhole Camera Model.

        x, y, w, h = bbox
        fx = cam_K[0, 0]
        fy = cam_K[1, 1]
        cx = cam_K[0, 2]
        cy = cam_K[1, 2]

        # Avoid division by zero
        if h <= 0 or w <= 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Estimate Depth (tz)
        # tz = (fy * real_height) / h
        if w > h:
            tz = fx * (real_height / w)
        else:
            tz = fy * (real_height / h)

        # Estimate Centroid
        cx_bbox = x + (w / 2.0)
        cy_bbox = y + (h / 2.0)

        # Back-project to 3D
        tx = (cx_bbox - cx) * tz / fx
        ty = (cy_bbox - cy) * tz / fy

        return np.array([tx, ty, tz], dtype=np.float32)


# Full Pipeline (YOLO + Pinhole + ResNet)
class BaselinePoseSystem:
    def __init__(self, yolo_path, resnet_path, device='cuda'):
        self.device = device

        # Load YOLO (Detection)
        if os.path.exists(yolo_path):
            self.yolo = YOLO(yolo_path)
            self.yolo.to(self.device)
            print("YOLO Loaded.")
        else:
            raise FileNotFoundError(f"YOLO model not found at {yolo_path}.")

        # Load ResNet (Rotation)
        self.rot_net = load_rotationresnet_model(resnet_path, self.device, pretrained=False)
        if self.rot_net is None:
            print("Warning: ResNet path not found, initializing with ImageNet pretrained weights.")
            self.rot_net = RotationResNet(pretrained=True).to(self.device)
            self.rot_net.eval()

        # Load Translation Layer (Math)
        self.trans_layer = PinholeTranslationLayer()

        # Define Transform (for ResNet input)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, full_image_np, cam_K, real_height, conf=0.5):
        """
        Full Pipeline Inference:
        1. Detect Object (YOLO)
        2. Crop & Transform
        3. Predict Rotation (ResNet)
        4. Calculate Translation (Pinhole)
        """

        # Step 1: Yolo Detection
        # Ultralytics YOLO expects BGR when passed a numpy array.
        # Baseline dataset provides RGB, so we flip the channels back.
        image_bgr = cv2.cvtColor(full_image_np, cv2.COLOR_RGB2BGR)

        # Run inference on the full image
        results = self.yolo(image_bgr, verbose=False, conf=conf)

        pred_bbox = None
        # Get the highest confidence box
        if len(results[0].boxes) > 0:
            best_box = sorted(results[0].boxes.data.tolist(), key=lambda x: x[4], reverse=True)[0]
            x1, y1, x2, y2 = best_box[:4]
            w = x2 - x1
            h = y2 - y1
            pred_bbox = [x1, y1, w, h]
        else:
            # Detection failed
            return None, None

        # Step 2: Prepare inputs
        x, y, w, h = map(int, pred_bbox)

        # Clamp to image dimensions
        H, W, _ = full_image_np.shape
        x = max(0, x); y = max(0, y)
        w = min(w, W - x); h = min(h, H - y)

        if w <= 0 or h <= 0:
          return None, None

        # Crop
        crop = full_image_np[y:y+h, x:x+w]

        # Transform for ResNet
        crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)

        # Step 3: Predict pose
        # A. Rotation (ResNet)
        with torch.no_grad():
            pred_quat = self.rot_net(crop_tensor).cpu().numpy()[0]
            pred_quat = pred_quat / np.linalg.norm(pred_quat) # Normalize
            pred_R = R.from_quat(pred_quat).as_matrix()

        # B. Translation (Pinhole Layer)
        pred_t = self.trans_layer.forward(pred_bbox, cam_K, real_height)

        return pred_R, pred_t