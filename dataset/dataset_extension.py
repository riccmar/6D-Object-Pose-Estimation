import os
import yaml
import hashlib
from torch.utils.data import Dataset
from dataset.dataset_baseline import LINEMOD_ID_MAP

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

            for frame_idx_str in gt_data.keys():
                frame_idx = int(frame_idx_str)
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
