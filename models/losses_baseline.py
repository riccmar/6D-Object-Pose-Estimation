import torch
import torch.nn as nn
import torch.nn.functional as F

class QuaternionLoss(nn.Module):
    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, pred_q, target_q):
        """
        pred_q: [batch_size, 4] (Network output)
        target_q: [batch_size, 4] (Ground Truth from dataset)
        """
        # Normalize the predictions
        pred_q = F.normalize(pred_q, p=2, dim=1)

        # Normalize target
        target_q = F.normalize(target_q, p=2, dim=1)

        # Calculate Dot Product
        dot_product = torch.sum(pred_q * target_q, dim=1)

        # Handle Double Cover
        # If dot_product is 1.0/-1.0 -> Perfect match
        # If dot_product is 0.0 -> Terrible match (90 degrees off)
        loss = 1.0 - torch.abs(dot_product)

        # Return mean over the batch
        return loss.mean()
