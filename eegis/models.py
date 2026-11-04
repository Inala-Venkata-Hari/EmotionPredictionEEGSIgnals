import torch
from torch import nn

class EEGNet(nn.Module):
    """Compact temporal, depthwise spatial, separable temporal EEG classifier."""
    def __init__(self, channels=14, classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,8,(1,33),padding=(0,16),bias=False),nn.BatchNorm2d(8),
            nn.Conv2d(8,16,(channels,1),groups=8,bias=False),nn.BatchNorm2d(16),
            nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(.35),
            nn.Conv2d(16,16,(1,17),padding=(0,8),groups=16,bias=False),
            nn.Conv2d(16,16,1,bias=False),nn.BatchNorm2d(16),nn.ELU(),
            nn.AvgPool2d((1,8)),nn.Dropout(.35),nn.Flatten(),nn.Linear(64,classes))
    def forward(self,x):
        return self.net(x)
