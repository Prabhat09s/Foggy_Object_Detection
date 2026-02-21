import torch
import torch.nn as nn
from .common import Conv, Concat

class MP(nn.Module):
    # MPConv block
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), 1))

class ELAN(nn.Module):
    # ELAN block for YOLOv7
    def __init__(self, c1, c2, c3, n=4, e=1):
        super(ELAN, self).__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c1, c3, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c3, c3, 3, 1) for _ in range(n)]
        )
        self.cv4 = Conv(c3 * (n + 2), c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x), self.cv2(x)]
        for i, m in enumerate(self.cv3):
            y.append(m(y[-1]))
        
        return self.cv4(torch.cat(y, 1))
