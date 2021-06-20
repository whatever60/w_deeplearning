import torch
from torch import nn
import einops
from einops.layers.torch import Rearrange


class MixerBlock(nn.Module):
    def __init__(self, seq_length, emb_dim, tokens_hid_dim, channels_hid_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.token_mixing = nn.Sequential(
            nn.Linear(seq_length, tokens_hid_dim),
            nn.GELU(),
            nn.Linear(tokens_hid_dim, seq_length),
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.channel_mixing = nn.Sequential(
            nn.Linear(emb_dim, channels_hid_dim),
            nn.GELU(),
            nn.Linear(channels_hid_dim, emb_dim),
        )

    def forward(self, x):  # [batch_size, seq_length, emb_dim]
        y = self.norm1(x)  # Note that norm is always applied on `hid_dim`
        x = self.token_mixing(y.permute(0, 2, 1)).permute(0, 2, 1) + x

        y = self.norm2(x)
        x = self.channel_mixing(y) + x
        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        num_layers,
        seq_length,
        emb_dim,
        tokens_hid_dim,
        channels_hid_dim,
        num_classes,
    ):
        # assume both the input image and patch are square.
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    seq_length=seq_length,
                    emb_dim=emb_dim,
                    tokens_hid_dim=tokens_hid_dim,
                    channels_hid_dim=channels_hid_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):  # [batch_size, num_patches, emb_dim]
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # [batch_size, num_patches, emb_dim]
        x = x.mean(dim=1)  # [batch_size, emb_dim]
        logit = self.head(x)
        return logit


class MLPMixerVision(MLPMixer):
    def __init__(self, num_layers, in_channels, input_size, patch_size, emb_dim, tokens_hid_dim, channels_hid_dim, num_classes):
        assert not input_size % patch_size
        num_patches = (input_size // patch_size) ** 2
        super().__init__(num_layers, num_patches, emb_dim, tokens_hid_dim, channels_hid_dim, num_classes)
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim, patch_size, patch_size),
            Rearrange('b h x y -> b (x y) h')
        )


class MLPMixerNLP(MLPMixer):
    def __init__(self, num_layers, num_tokens, seq_length, emb_dim, tokens_hid_dim, channels_hid_dim, num_classes):
        super().__init__(num_layers, seq_length, emb_dim, tokens_hid_dim, channels_hid_dim, num_classes)
        self.embedding = nn.Embedding(num_tokens, emb_dim)


def test_mlpmixer():
    batch_size = 8
    in_channels = 3
    input_size = 64
    patch_size = 16
    model = MLPMixerVision(4, 10, in_channels, input_size, patch_size, 32, 64, 64)
    input_ = torch.randn(batch_size, in_channels, input_size, input_size)
    logit = model(input_)
    rprint(logit.shape)  # [batch_size, num_classes]


def test_longsequence():
    device = 'cuda:2'
    batch_size = 200
    num_tokens = 20000
    seq_length = 10000
    num_layers = 6
    num_classes = 10
    input_ = torch.randint(num_tokens, size=(batch_size, seq_length), device=device)
    labels = torch.randint(num_classes, size=(batch_size,), device=device)
    model = MLPMixerNLP(num_layers, num_tokens, seq_length, 30, 30, 30, num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    preds = model(input_)
    loss = criterion(preds, labels)
    loss.backward()


def test_patch_embedding():
    """
    Validate that patch embedding can be implemented by Linear as well as Conv2d
    """
    batch_size = 8
    in_channels = 3
    input_size = 60
    patch_size = 5
    emb_dim = 32

    input_ = torch.randn(batch_size, in_channels, input_size, input_size)
    input_linear = einops.rearrange(
        input_, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
    )

    conv = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)
    linear = nn.Linear(in_channels * patch_size * patch_size, emb_dim)
    rprint(linear.weight.shape)
    rprint(conv.weight.shape)

    count_trainable_parameter = lambda m: sum(
        p.numel() for p in m.parameters() if p.requires_grad
    )
    rprint(count_trainable_parameter(linear))
    rprint(count_trainable_parameter(conv))

    linear.weight = nn.Parameter(conv.weight.clone().flatten(start_dim=1))
    linear.bias = nn.Parameter(conv.bias.clone())

    out_linear = linear(input_linear)  # [batch_size, num_patches, emb_dim]
    # [batch_size, num_patches, emb_dim]
    out_conv = einops.rearrange(conv(input_), "b c h w -> b (h w) c")
    rprint(torch.allclose(out_linear, out_conv, atol=1e-6, rtol=0))
    rprint(torch.allclose(out_linear, out_conv, atol=1e-5, rtol=0))


class Conv1dDepthwiseShared(nn.Module):
    def __init__(self, in_channels, k, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        model = nn.Conv1d(1, k, kernel_size)
        self.weight = model.weight
        self.bias = model.bias

    def forward(self, x):
        weight = self.weight.repeat(self.in_channels, 1, 1)
        bias = self.bias.repeat(self.in_channels)
        return nn.functional.conv1d(
            x, weight=weight, bias=bias, groups=self.in_channels
        )


def test_mixing():
    """
    Validate that when the input comes with shape `[batch_size, num_patches, emb_dim]`,
    both token mixing and channel mixing can be implemented with linear as well as conv.

    Token mixing can be seen as a Conv1d with `num_patches` input channels,
    `token_hid_dim` output channels, and kernel size 1.

    Channel mixing can be seen as a Depthwise Conv1d with `num_patches` input channels,
    `channel_hid_dim * num_patches` output channels, `num_patches` groups and kernel
    size `hid_dim`. Each group of this conv shares the same weight and bias.
    """
    batch_size = 8
    emb_dim = 32
    token_hid_dim = 60
    channel_hid_dim = 60
    num_patches = 25

    input_ = torch.randn(batch_size, num_patches, emb_dim)

    # token mixing
    conv1 = nn.Conv1d(num_patches, token_hid_dim, 1)
    linear1 = nn.Linear(num_patches, token_hid_dim)
    # channel mixing
    conv2 = Conv1dDepthwiseShared(num_patches, channel_hid_dim, emb_dim)
    linear2 = nn.Linear(emb_dim, channel_hid_dim)

    rprint(conv1.weight.shape)  # [token_hid_dim, num_patches, 1]
    rprint(linear1.weight.shape)
    rprint(conv2.weight.shape)  # [channel_hid_dim * num_patches, 1, emb_dim]
    rprint(linear2.weight.shape)

    count_trainable_parameter = lambda m: sum(
        p.numel() for p in m.parameters() if p.requires_grad
    )
    rprint(count_trainable_parameter(linear1))
    rprint(count_trainable_parameter(conv1))
    rprint(count_trainable_parameter(linear2))
    rprint(count_trainable_parameter(conv2))

    linear1.weight = nn.Parameter(conv1.weight.clone().flatten(start_dim=1))
    linear1.bias = nn.Parameter(conv1.bias.clone())
    linear2.weight = nn.Parameter(conv2.weight.clone()[:, 0])
    linear2.bias = nn.Parameter(conv2.bias.clone())

    out_linear1 = linear1(input_.permute(0, 2, 1))
    out_conv1 = conv1(input_)
    rprint(out_linear1.shape)  # [batch_size, emb_dim, token_hid_dim]
    rprint(out_conv1.shape)  # [batch_size, token_hid_dim, emb_dim]
    rprint(torch.allclose(out_linear1.permute(0, 2, 1), out_conv1))

    out_linear2 = linear2(input_)
    out_conv2 = conv2(input_)
    rprint(out_linear2.shape)  # [batch_size, num_patches, token_hid_dim]
    rprint(out_conv2.shape)  # [batch_size, num_patches * token_hid_dim, 1]
    rprint(
        torch.allclose(
            out_linear2,
            out_conv2.view(batch_size, num_patches, token_hid_dim),
            atol=1e-6,
        )
    )    


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
    import pytorch_lightning as pl

    pl.seed_everything(42)

    # test_mlpmixer()
    test_longsequence()
    # test_patch_embedding()
    # test_mixing()
