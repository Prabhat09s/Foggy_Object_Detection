import torch
import torch.nn as nn
from .attention import FeatureAttention

class FeatureConversation(nn.Module):
    def __init__(self, in_channels):
        super(FeatureConversation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.attention = FeatureAttention(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.attention(out)
        out = self.conv2(out)
        return self.act(out + res)
