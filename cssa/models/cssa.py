import torch
import torch.nn as nn
import torch.nn.functional as F

class CSSA(nn.Module):
    def __init__(self, in_channels, reduction_factor=4, cssa_thresh=2e-3):
        super(CSSA, self).__init__()
        reduced_channels = in_channels // reduction_factor
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.cssa_thresh = cssa_thresh

    def forward(self, x):
        identity = x
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.relu(self.fc1(x_avg))
        x = torch.sigmoid(self.fc2(x))
        
        x = identity * x
        x[x < self.cssa_thresh] = 0
        return x
