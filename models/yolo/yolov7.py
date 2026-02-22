import torch
import torch.nn as nn
import torch
import torch.nn as nn

from .common import Conv
from .blocks import ELAN, MP, SPPCSPC

class YOLOv8(nn.Module):
    def __init__(self, num_classes=8, ch=3):
        super().__init__()

        # Backbone
        self.layer1 = Conv(ch, 32, 3, 2)
        self.layer2 = Conv(32, 64, 3, 2)
        self.layer3 = C2f(64, 64)
        self.layer4 = Conv(64, 128, 3, 2)
        self.layer5 = C2f(128, 128)
        self.layer6 = Conv(128, 256, 3, 2)
        self.layer7 = C2f(256, 256)
        self.layer8 = Conv(256, 512, 3, 2)
        self.layer9 = C2f(512, 512)

        # Neck (Feature Fusion)
        self.upsample = nn.Upsample(scale_factor=2)
        self.concat = Concat()

        # Head
        self.detect_p3 = nn.Conv2d(128, num_classes + 4, 1)
        self.detect_p4 = nn.Conv2d(256, num_classes + 4, 1)
        self.detect_p5 = nn.Conv2d(512, num_classes + 4, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        p3 = self.layer3(x)

        x = self.layer4(p3)
        p4 = self.layer5(x)

        x = self.layer6(p4)
        p5 = self.layer7(x)

        x = self.layer8(p5)
        p6 = self.layer9(x)

        out3 = self.detect_p3(p3)
        out4 = self.detect_p4(p4)
        out5 = self.detect_p5(p6)

        return [out3, out4, out5]