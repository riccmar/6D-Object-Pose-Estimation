import torch
import torch.nn as nn
from torchvision import models

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