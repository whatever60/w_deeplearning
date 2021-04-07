from math import log2

import torch
from torch import nn
from torch.nn import functional as F


class WSConv2d(nn.Module):
    """
    Equalized learning rate
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        gain=2,
        use_act=True,
    ):
        super().__init__()
        self.model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias, self.model.bias = self.model.bias, None
        nn.init.normal_(self.model.weight)
        nn.init.zeros_(self.bias)
        self.use_act = use_act

    def forward(self, x):
        x = self.model(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
        return F.leaky_relu(x, 0.2) if self.use_act else x
        # return self.act(x) if self.use_act else x


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pn):
        super().__init__()
        Norm = PixelNorm if use_pn else nn.Identity
        # They use conv-act-norm instead of conv-norm-act, for no explicit reason.
        self.model = nn.Sequential(
            WSConv2d(in_channels, out_channels, 3, 1, 1),
            Norm(),
            WSConv2d(out_channels, out_channels, 3, 1, 1),
            Norm(),
        )

    def forward(self, x):
        return self.model(x)


class Gen(nn.Module):
    def __init__(self, z_dim, in_channels, factors):
        super().__init__()
        init = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, 3, 1, 1),
            PixelNorm(),
        )
        init_rgb = WSConv2d(in_channels, 3, 1, 1, 0, use_act=False)
        prog_blocks, to_rgbs = nn.ModuleList([init]), nn.ModuleList([init_rgb])

        for in_, out in zip(factors[:-1], factors[1:]):
            in_, out = int(in_channels * in_), int(in_channels * out)
            prog_blocks.append(ConvBlock(in_, out, use_pn=True))
            to_rgbs.append(WSConv2d(out, 3, 1, 1, 0, use_act=False))

        self.prog_blocks = prog_blocks
        self.to_rgbs = to_rgbs
        self.upsample = F.interpolate
        # self.alpha = 1e-5

    def forward(self, x: torch.tensor):
        # steps = 0 (4 x 4), steps = 1 (8 x 8)
        alpha = self.alpha
        steps = self.steps
        # ------ For the first block ------
        x = x.view(*x.shape, 1, 1)
        out = self.prog_blocks[0](x)
        # ------ Go up the Gen ------
        for step in range(1, steps + 1):
            up = self.upsample(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](up)
        # ------ Normally, this would be the output ------
        out = self.to_rgbs[steps](out)
        # ------ However, we want some fade in ------
        if steps > 0:  # There is only one upsample.
            up = self.to_rgbs[steps - 1](up)
            out = alpha * out + (1 - alpha) * up
        return torch.tanh(out)


class Disc(nn.Module):
    def __init__(self, in_channels, factors):
        super().__init__()
        final = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, 3, 1, 1),
            WSConv2d(in_channels, in_channels, 4, 1, 0),
            WSConv2d(
                in_channels, 1, 1, 1, 0, use_act=False
            ),  # This should be a "WSLinear" layer. But, same thing.
        )  # mirror of init in Gen
        final_rgb = WSConv2d(3, in_channels, 1, 1, 0)
        prog_blocks, from_rgbs = nn.ModuleList([final]), nn.ModuleList([final_rgb])

        for in_, out in zip(factors[:-1], factors[1:]):
            in_, out = int(in_channels * in_), int(in_channels * out)
            prog_blocks.append(ConvBlock(out, in_, use_pn=False))
            from_rgbs.append(WSConv2d(3, out, 1, 1, 0))

        self.prog_blocks = prog_blocks
        self.from_rgbs = from_rgbs
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.alpha = 1e-5

    def minibatch_std(self, x):
        batch_stat = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_stat], dim=1)

    def forward(self, x):
        # steps = 0 (4 x 4), steps = 1 (8 x 8), ...
        alpha = self.alpha
        steps = self.steps
        # ------ Normally, this would be the input ------
        out = self.from_rgbs[steps](x)
        # ------ However, we want some fade in ------
        if steps > 0:  # There are two downsamples.
            out = self.downsample(self.prog_blocks[steps](out))
            down = self.from_rgbs[steps - 1](self.downsample(x))
            out = alpha * out + (1 - alpha) * down
        # ------ Go dowm the Disc ------
        for step in range(steps - 1, 0, -1):
            out = self.prog_blocks[step](out)
            out = self.downsample(out)
        # ------ For the last block ------
        out = self.minibatch_std(out)
        return self.prog_blocks[0](out).view(out.shape[0], -1)


if __name__ == "__main__":
    torch.manual_seed(42)
    FACTORS = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    Z_DIM = 100
    IN_CHANNELS = 256
    x = torch.randn((2, Z_DIM))
    gen = Gen(Z_DIM, IN_CHANNELS, FACTORS)
    disc = Disc(IN_CHANNELS, FACTORS)

    for SIZE in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        # for SIZE in [128]:
        steps = int(log2(SIZE / 4))
        disc.steps = gen.steps = steps
        fake = gen(x)
        d_fake = disc(fake)
        assert fake.shape == (2, 3, SIZE, SIZE)
        assert d_fake.shape == (2, 1)
        print(f"====== Success! At img size: {SIZE} ======")
        print(fake[0][0][2][:4])
        print(d_fake)
