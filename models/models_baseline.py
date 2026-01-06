import torch
import torch.nn as nn
import os
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