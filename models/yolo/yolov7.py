import torch
import torch.nn as nn
from .common import Conv, Concat
from .blocks import ELAN, MP, SPPCSPC

class YOLOv7(nn.Module):
    def __init__(self, num_classes=8, ch=3):
        super(YOLOv7, self).__init__()
        
        # Backbone (Simplified YOLOv7-tiny style for demonstration/speed, 
        # normally this is much deeper with more ELAN blocks)
        self.backbone = nn.Sequential(
            Conv(ch, 32, 3, 2),       # 0-P1/2
            Conv(32, 64, 3, 2),       # 1-P2/4
            ELAN(64, 64, 32, n=2),    # 2
            Conv(64, 128, 3, 2),      # 3-P3/8
            ELAN(128, 128, 64, n=2),  # 4
            Conv(128, 256, 3, 2),     # 5-P4/16
            ELAN(256, 256, 128, n=2), # 6
            Conv(256, 512, 3, 2),     # 7-P5/32
            ELAN(512, 512, 256, n=2)  # 8
        )
        
        # Head
        self.sppcspc = SPPCSPC(512, 256)
        
        # In a real YOLOv7, there is a complex PA-NET structure here.
        # For this term paper implementation, we will use a simpler detection head 
        # attached to the features, or return features for the loss to handle.
        # Here we define the final detection layers for 3 scales.
        
        self.det_p3 = Conv(128, 256, 3, 1) # From layer 4
        self.det_p4 = Conv(256, 512, 3, 1) # From layer 6
        self.det_p5 = Conv(256, 1024, 3, 1) # From layer 8 (via SPP)

        # Detect heads (1x1 convs to map to num_anchors * (classes + 5))
        # Assuming 3 anchors per scale
        self.nl = 3
        self.nc = num_classes
        self.no = num_classes + 5
        self.na = 3
        
        self.m = nn.ModuleList([
            nn.Conv2d(256, self.na * self.no, 1),
            nn.Conv2d(512, self.na * self.no, 1),
            nn.Conv2d(1024, self.na * self.no, 1)
        ])
        
        self.anchors = torch.tensor([
            [10,13, 16,30, 33,23],
            [30,61, 62,45, 59,119],
            [116,90, 156,198, 373,326]
        ]).float().view(3, 1, -1, 2)

    def forward(self, x):
        # Backbone
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        p2 = self.backbone[2](x)
        x = self.backbone[3](p2)
        p3 = self.backbone[4](x)
        x = self.backbone[5](p3)
        p4 = self.backbone[6](x)
        x = self.backbone[7](p4)
        p5 = self.backbone[8](x)
        
        # Head
        p5 = self.sppcspc(p5)
        
        # Simple Feature Pyramid (simplified for brevity vs full PA-Net)
        out_p3 = self.det_p3(p3)
        out_p4 = self.det_p4(p4)
        out_p5 = self.det_p5(p5)
        
        x = [out_p3, out_p4, out_p5]
        
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

    # Initialize grids
    stride = [8, 16, 32]
    grid = [torch.zeros(1)] * 3
    anchor_grid = [torch.zeros(1)] * 3
