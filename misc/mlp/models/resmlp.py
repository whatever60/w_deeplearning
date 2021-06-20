import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce


class Affine(nn.Module):
    """This functions as layer norm in MLP-Mixer."""

    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, emb_dim))
        self.b = nn.Parameter(torch.ones(1, 1, emb_dim))

    def forward(self, x):  # [batch_size, seq_length, emb_dim]
        return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):
    """From CaiT"""

    def __init__(self, emb_dim, layer, fn):
        super().__init__()
        if layer <= 18:
            init_eps = 0.1
        elif 18 < layer <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.full((1, 1, emb_dim), init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(emb_dim)
        self.fn = fn

    def forward(self, x):  # [batch_size, seq_length, emb_dim]
        return self.fn(self.affine(x)) * self.scale + x


class ResMLP(nn.Module):
    def __init__(
        self, num_layers, seq_length, emb_dim, num_classes, expansion=4
    ):
        super().__init__()
        self.seq_length = seq_length
        self.emb_dim = emb_dim
        self.expansion = expansion
        self.wrapper = lambda num_layers, module: PreAffinePostLayerScale(
            emb_dim, num_layers + 1, module
        )

        self.model = nn.Sequential(
            *[
                nn.Sequential(self._make_token_mixing(i), self._make_channel_mixing(i))
                for i in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            Affine(emb_dim), Reduce("b c d -> b d", "mean"), nn.Linear(emb_dim, num_classes)
        )

    def _make_token_mixing(self, layer):
        return self.wrapper(layer, nn.Conv1d(self.seq_length, self.seq_length, 1))

    def _make_channel_mixing(self, layer):
        model = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * self.expansion),
            nn.GELU(),
            nn.Linear(self.emb_dim * self.expansion, self.emb_dim),
        )
        return self.wrapper(layer, model)

    def forward(self, x):
        return self.head(self.model(self.embedding(x)))


class ResMLPVision(ResMLP):
    def __init__(self, num_layers, in_channels, input_size, patch_size, emb_dim, expansion, num_classes):
        assert not input_size % patch_size
        seq_length = (input_size // patch_size) ** 2
        embedding = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim, patch_size, patch_size),
            Rearrange("b d h w -> b (h w) d"),
        )
        super().__init__(num_layers, seq_length, emb_dim, num_classes, expansion=expansion)
        self.embedding = embedding


class ResMLPNLP(ResMLP):
    def __init__(self, num_layers, num_tokens, seq_length, emb_dim, expansion, num_classes):
        super().__init__(num_layers, seq_length, emb_dim, num_classes, expansion=expansion)
        self.embedding = nn.Embedding(num_tokens, emb_dim)


def test_resmlp():
    batch_size = 11
    num_layers = 6
    num_classes = 10
    model = ResMLPVision(num_layers, 3, 60, 6, 256, 3, num_classes)
    data = torch.randn(batch_size, 3, 60, 60)
    rprint(model(data).shape)  # [batch_size, num_classes]


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
    import pytorch_lightning as pl

    pl.seed_everything(42)

    test_resmlp()
