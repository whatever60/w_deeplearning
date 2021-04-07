import torch
from torch import nn
from torch.random import seed


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        # stride is usually 2, but there are exceptions.
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


class Disc(nn.Module):
    def __init__(self, in_channels, features):
        # 256 -> 30 x 30
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels * 2,
                out_channels=features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )  # in_channels * 2 because real image and generated image will be concatenated

        layers = []
        for i in range(len(features) - 1):
            stride = 1 if i == len(features) - 2 else 2
            # stride = 2 except for the last one
            layers.append(CNNBlock(features[i], features[i + 1], stride))
        layers.append(
            nn.Conv2d(
                in_channels=features[-1],
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # x: real image, y: fake image
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down, act, use_dropout):
        super().__init__()
        modules = nn.ModuleList()

        if down:
            modules.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    4,
                    2,
                    1,
                    bias=False,
                    padding_mode="reflect",
                )
            )
        else:
            modules.append(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            )
        modules.append(nn.BatchNorm2d(out_channels))

        if act == "relu":
            modules.append(nn.ReLU())
        else:
            modules.append(nn.LeakyReLU(0.2))

        if use_dropout:
            modules.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Gen(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        # 256

        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
                    nn.LeakyReLU(0.2),
                ),  # 128
                Block(
                    features * 1,
                    features * 2,
                    down=True,
                    act="leaky",
                    use_dropout=False,
                ),  # 64
                Block(
                    features * 2,
                    features * 4,
                    down=True,
                    act="leaky",
                    use_dropout=False,
                ),  # 32
                Block(
                    features * 4,
                    features * 8,
                    down=True,
                    act="leaky",
                    use_dropout=False,
                ),  # 16
                Block(
                    features * 8,
                    features * 8,
                    down=True,
                    act="leaky",
                    use_dropout=False,
                ),  # 8
                Block(
                    features * 8,
                    features * 8,
                    down=True,
                    act="leaky",
                    use_dropout=False,
                ),  # 4
                Block(
                    features * 8,
                    features * 8,
                    down=True,
                    act="leaky",
                    use_dropout=False,
                ),  # 2
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(),
        )  # 1
        self.up = nn.ModuleList(
            [
                Block(
                    features * 8,
                    features * 8,
                    down=False,
                    act="relu",
                    use_dropout=True
                ),
                Block(
                    features * 8 * 2,
                    features * 8,
                    down=False,
                    act="relu",
                    use_dropout=True,
                ),
                Block(
                    features * 8 * 2,
                    features * 8,
                    down=False,
                    act="relu",
                    use_dropout=True,
                ),
                Block(
                    features * 8 * 2,
                    features * 8,
                    down=False,
                    act="relu",
                    use_dropout=False,
                ),
                Block(
                    features * 8 * 2,
                    features * 4,
                    down=False,
                    act="relu",
                    use_dropout=False,
                ),
                Block(
                    features * 4 * 2,
                    features * 2,
                    down=False,
                    act="relu",
                    use_dropout=False,
                ),
                Block(
                    features * 2 * 2,
                    features,
                    down=False,
                    act="relu",
                    use_dropout=False,
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        features * 2, in_channels, kernel_size=4, stride=2, padding=1
                    ),
                    nn.Tanh(),
                ),
            ]
        )

    def forward(self, x):
        connection = []
        for layer in self.down:
            x = layer(x)
            connection.append(x)
        x = self.bottleneck(x)
        x = self.up[0](x)
        for layer in self.up[1:]:
            x = layer(torch.cat([x, connection.pop()], dim=1))
        return x


def test():
    torch.random.manual_seed(42)
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Gen(in_channels=3, features=64)
    preds = model(x)
    model = Disc(in_channels=3, features=[64, 128, 256, 512])
    disc = model(preds, y)
    print(preds.shape)
    print(preds[0, 0, 0, :5])
    print(disc.shape)
    print(disc[0, 0, 0, :5])


if __name__ == "__main__":
    test()
