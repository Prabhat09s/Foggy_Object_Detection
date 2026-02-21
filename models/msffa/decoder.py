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

class Decoder(nn.Module):
    def __init__(self, out_channels=3, n_res_blocks=3):
        super(Decoder, self).__init__()
        
        in_features = 256
        out_features = in_features // 2

        model = []
        
        # Residual blocks
        for _ in range(n_res_blocks):
            model += [ResBlock(in_features)]

        # Upsampling
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
