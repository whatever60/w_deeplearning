import torch
from torch import nn


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        act=None,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            ShiftedReLU(inplace=True) if act == 'shifted' else nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class ShiftedReLU(nn.Module):
    def __init__(self, shift=0.5, inplace=True):
        super().__init__()
        self.model = nn.ReLU(inplace=inplace)
        self.shift = shift

    def forward(self, x):
        return self.model(x - self.shift)


class RectifiedTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.ReLU(), nn.Tanh())

    def forward(self, x):
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, features=(128, 64, 32), block=LinearBlock):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i, feature in enumerate(features):
            act = None if i else 'shifted'
            self.encoder.append(block(in_dim, feature, act))
            in_dim = feature

        feature = in_dim // 2
        self.bottleneck = block(in_dim, feature)
        in_dim = feature

        for feature in features[::-1]:
            self.decoder.append(nn.Linear(in_dim, feature))
            self.decoder.append(block(feature * 2, feature))
            in_dim = feature
        # self.final_linear = nn.Linear(in_dim, out_dim)
        self.final_linear = nn.Sequential(nn.Linear(in_dim, out_dim), RectifiedTanh())
        # self.apply(weight_init)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        for i, skip_connection in enumerate(reversed(skip_connections)):
            x = self.decoder[i * 2](x)
            x = torch.cat([skip_connection, x], dim=1)
            x = self.decoder[i * 2 + 1](x)
        return self.final_linear(x)
