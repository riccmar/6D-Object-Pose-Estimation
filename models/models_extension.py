import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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