import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_features):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, n_res_blocks=3):
        super(Encoder, self).__init__()
        
        # Initial convolution
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_res_blocks):
            model += [ResBlock(in_features)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
