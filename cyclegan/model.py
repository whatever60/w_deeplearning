import torch
from torch import nn


class Block(nn.Module):
    # For discriminator, [Conv, InstanceNorm, LeakyReLU]
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )  # kernel_size = 4 in CycleGAN, padding model reflect to reduce artifact, LeakyReLU for Disc

    def forward(self, x):
        return self.model(x)


class Disc(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        out_channels = features[0]
        self.init = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )  # No InstanceNorm in init block

        layers = []
        for feature in features[1:]:
            in_channels, out_channels = out_channels, feature
            layers.append(Block(in_channels, out_channels, stride=1 if out_channels == features[-1] else 2))
        self.layers = nn.Sequential(*layers)
        in_channels = out_channels
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.init(x)
        x = self.layers(x)
        return self.final(x)


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, down, use_act, kernel_size, padding, stride):
#         super().__init__()
#         self.model


class ResidualBlock(nn.Module):
    # does not change resolution or channels
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels)
        )  # ReLU for Gen. Only use ReLU after the first Conv
    
    def forward(self, x):
        return x + self.model(x)


class Gen(nn.Module):
    def __init__(self, in_channels, num_residual, num_features):
        # num_residual = 9 for larger (256 x 256) images, and 6 for smaller (128 x 128) images
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.Sequential(
            nn.Conv2d(num_features * 1, num_features * 2, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features * 4),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residual)]
        )
        self.up_blocks = nn.Sequential(
            nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # they use this output_padding,
            nn.InstanceNorm2d(num_features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 2, num_features * 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_features * 1),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_features * 1, in_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.init(x)
        x = self.down_blocks(x)
        x = self.residual_blocks(x)
        x = self.up_blocks(x)
        return self.final(x)


def test():
    torch.random.manual_seed(42)
    x = torch.randn((10, 3, 256, 256))
    gen = Gen(3, 9, 64)
    fake = gen(x)
    print(x.shape)
    print(x[0, 0, 0, :5])
    disc = Disc(3, (64, 128, 256, 512))
    fake_index = disc(fake)
    print(fake_index.shape)
    print(fake_index[0, 0, 0, :5])


if __name__ == '__main__':
    test()
