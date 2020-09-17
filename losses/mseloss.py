import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.MSELoss()
    def forward(self, inputs, targets):
        loss = self.out(inputs, targets)
        return loss, {'T': loss.item()}