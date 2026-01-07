import os
import yaml
import hashlib
import torch
import numpy as np
import cv2
import random
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class YoloSegDataset(Dataset):
    """
    Dataset class specifically for YOLO Segmentation in the extension phase.
    It retrieves both RGB images and their corresponding masks, ensuring both exist.
    """
    def __init__(self, dataset_root, split='train'):
        self.dataset_root = dataset_root
        self.split = split
        self.data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')
        
        self.samples = []
        
        # Check if data dir exists
        if not os.path.isdir(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist.")
            return

        # Iterate over object folders
        for obj_id_str in sorted(os.listdir(self.data_dir)):
            if not obj_id_str.isdigit():
                continue

            obj_id = int(obj_id_str)
            gt_path = os.path.join(self.data_dir, obj_id_str, 'gt.yml')

            if not os.path.exists(gt_path):
                print(f"Warning: gt.yml not found for object {obj_id_str}. Skipping.")
                continue

            with open(gt_path, 'r') as f:
                gt_data = yaml.safe_load(f)

            for frame_idx_str, anns in gt_data.items():
                frame_idx = int(frame_idx_str)

                # Filter frames with no annotations
                if not anns:
                    continue

                unique_id = f"{obj_id}_{frame_idx}"
                
                # Deterministic Hash for Split (Same as baseline for consistency)
                hash_val = int(hashlib.md5(unique_id.encode()).hexdigest(), 16) % 100
                is_train = hash_val < 80 

                if (split == 'train' and is_train) or (split in ['val', 'test'] and not is_train):
                    # Verify files exist before adding
                    obj_str = f"{obj_id:02d}"
                    fr_str = f"{frame_idx:04d}"
                    rgb = os.path.join(self.data_dir, obj_str, 'rgb', f"{fr_str}.png")
                    mask = os.path.join(self.data_dir, obj_str, 'mask', f"{fr_str}.png")
                    
                    if os.path.exists(rgb) and os.path.exists(mask):
                        self.samples.append((obj_id, frame_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_id, frame_idx = self.samples[idx]
        obj_id_str = f"{obj_id:02d}"
        frame_idx_str = f"{frame_idx:04d}"
        
        base_path = os.path.join(self.data_dir, obj_id_str)
        rgb_path = os.path.join(base_path, 'rgb', f"{frame_idx_str}.png")
        mask_path = os.path.join(base_path, 'mask', f"{frame_idx_str}.png")

        return {
            "obj_id": obj_id,
            "frame_idx": frame_idx,
            "rgb_path": rgb_path,
            "mask_path": mask_path
        }

class RgbdFusionNetDataset(Dataset):
    def __init__(self, dataset_root, split='train', transform=None):
        self.dataset_root = dataset_root
        self.split = split
        self.num_points = 500

        self.data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')

        self.samples = []
        print(f"Initializing {split} dataset with BG Augmentation...")

        # Indexing
        if os.path.exists(self.data_dir):
            obj_folders = sorted([f for f in os.listdir(self.data_dir)
                                  if os.path.isdir(os.path.join(self.data_dir, f)) and f.isdigit()])

            for obj_id_str in obj_folders:
                obj_id = int(obj_id_str)
                base_dir = os.path.join(self.data_dir, obj_id_str)
                gt_path = os.path.join(base_dir, "gt.yml")
                info_path = os.path.join(base_dir, "info.yml")

                if not os.path.exists(gt_path) or not os.path.exists(info_path):
                    continue

                with open(gt_path, 'r') as f: 
                    gt_data = yaml.safe_load(f)
                with open(info_path, 'r') as f: 
                    info_data = yaml.safe_load(f)

                for frame_idx_str in gt_data.keys():
                    frame_idx = int(frame_idx_str)

                    # Hash Split (80% Train / 20% Val)
                    unique_id = f"{obj_id}_{frame_idx}"
                    hash_val = int(hashlib.md5(unique_id.encode()).hexdigest(), 16) % 100
                    is_train = hash_val < 80

                    if (split == 'train' and is_train) or (split == 'val' and not is_train):
                        anns = gt_data[frame_idx_str]
                        if not anns: 
                            continue

                        target_ann = None
                        for item in anns:
                            if item['obj_id'] == obj_id:
                                target_ann = item
                                break

                        if target_ann is None: 
                            continue

                        ann = target_ann
                        t_raw = np.array(ann['cam_t_m2c'])
                        if np.linalg.norm(t_raw) > 1.0: 
                            t_raw = t_raw / 1000.0

                        self.samples.append({
                            "obj_id": obj_id,
                            "rgb_path": os.path.join(base_dir, "rgb", f"{frame_idx:04d}.png"),
                            "depth_path": os.path.join(base_dir, "depth", f"{frame_idx:04d}.png"),
                            "mask_path": os.path.join(base_dir, "mask", f"{frame_idx:04d}.png"),
                            "cam_K": np.array(info_data[frame_idx]['cam_K']).reshape(3, 3),
                            "cam_R": np.array(ann['cam_R_m2c']).reshape(3, 3),
                            "cam_t": t_raw,
                            "bbox": ann['obj_bb'],
                            "depth_scale": info_data[frame_idx].get('depth_scale', 1.0)
                        })

        # Augmentation Pipeline
        if transform:
            self.transform = transform
        else:
            if self.split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    # Color Jitter
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    # Gaussian Blur (Random probability inside)
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                    transforms.ToTensor(),
                    # Random Erasing (Occlusion)
                    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.samples)

    def jitter_bbox(self, bbox, img_w, img_h):
        x, y, w, h = bbox

        noise_str = 0.10 if self.split == 'train' else 0.0
        if noise_str > 0:
            x_shift = int(np.random.uniform(-noise_str, noise_str) * w)
            y_shift = int(np.random.uniform(-noise_str, noise_str) * h)
            scale = np.random.uniform(0.9, 1.1)

            w_new = int(w * scale); h_new = int(h * scale)
            x_new = x + x_shift - (w_new - w) // 2
            y_new = y + y_shift - (h_new - h) // 2
            x_new = max(0, min(x_new, img_w - 1)); y_new = max(0, min(y_new, img_h - 1))
            w_new = min(w_new, img_w - x_new); h_new = min(h_new, img_h - y_new)

            return [x_new, y_new, w_new, h_new]
        
        return bbox

    def augment_rgb_background(self, rgb_crop, mask_crop):
        """ Replaces background pixels with noise or black """
        # 50% Probability Check
        if random.random() > 0.5:
            return rgb_crop  # Keep original background

        # Else, apply augmentation
        img_np = np.array(rgb_crop)
        mask_np = np.array(mask_crop)

        aug_type = random.choice(['noise', 'black', 'color'])

        if aug_type == 'black':
            img_np[mask_np == 0] = 0
        elif aug_type == 'color':
            color = np.random.randint(0, 255, (3,))
            img_np[mask_np == 0] = color
        elif aug_type == 'noise':
            noise = np.random.randint(0, 255, img_np.shape, dtype=np.uint8)
            img_np[mask_np == 0] = noise[mask_np == 0]

        return Image.fromarray(img_np)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self.split == 'val': np.random.seed(idx)

        try:
            rgb = Image.open(item["rgb_path"]).convert("RGB")
            depth = cv2.imread(item["depth_path"], -1)
            mask = cv2.imread(item["mask_path"], 0)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        w_img, h_img = rgb.size
        bbox = self.jitter_bbox(item["bbox"], w_img, h_img)
        x, y, w, h = bbox
        x1, y1 = np.clip(x, 0, w_img), np.clip(y, 0, h_img)
        x2, y2 = np.clip(x + w, 0, w_img), np.clip(y + h, 0, h_img)
        if (x2-x1) <= 0 or (y2-y1) <= 0: 
            x1, y1, x2, y2 = 0, 0, w_img, h_img

        rgb_crop = rgb.crop((x1, y1, x2, y2))

        if mask is None: 
            mask = np.ones_like(depth, dtype=np.uint8)
        mask_crop = mask[y1:y2, x1:x2]

        # Background Augmentation
        if self.split == 'train':
            # Resize mask to match RGB crop size
            mask_pil = Image.fromarray(mask_crop).resize(rgb_crop.size, Image.NEAREST)
            # Call the augmentation function
            rgb_crop = self.augment_rgb_background(rgb_crop, mask_pil)

        rgb_tensor = self.transform(rgb_crop)
        depth_crop = depth[y1:y2, x1:x2]

        ys, xs = np.nonzero(mask_crop > 0)
        if len(ys) == 0: ys, xs = np.nonzero(depth_crop > 0)
        if len(ys) == 0: ys, xs = np.array([h//2]*self.num_points), np.array([w//2]*self.num_points)

        replace = len(ys) < self.num_points
        choice = np.random.choice(len(ys), self.num_points, replace=replace)
        ys, xs = ys[choice], xs[choice]

        depth_vals = depth_crop[ys, xs] * item["depth_scale"]
        if np.mean(depth_vals) > 100: depth_vals *= 0.001

        u, v = xs + x1, ys + y1
        cx, cy = item["cam_K"][0, 2], item["cam_K"][1, 2]
        fx, fy = item["cam_K"][0, 0], item["cam_K"][1, 1]

        z_3d = depth_vals.astype(np.float32)
        x_3d = (u - cx) * z_3d / fx
        y_3d = (v - cy) * z_3d / fy
        cloud = np.stack([x_3d, y_3d, z_3d], axis=1)

        if self.split == 'train':
            noise = np.random.normal(0, 0.003, cloud.shape).astype(np.float32)
            cloud += noise

        cloud = np.nan_to_num(cloud, nan=0.0)
        centroid = np.mean(cloud, axis=0)
        cloud = cloud - centroid

        if self.split == 'val': np.random.seed(None)

        return {
            "rgb": rgb_tensor,
            "points": torch.tensor(cloud, dtype=torch.float32),
            "gt_R": torch.tensor(item["cam_R"], dtype=torch.float32),
            "gt_t": torch.tensor(item["cam_t"], dtype=torch.float32),
            "centroid": torch.tensor(centroid, dtype=torch.float32),
            "obj_id": item["obj_id"]
        }