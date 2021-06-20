import torch
from torch import nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange, Reduce


class Attention(nn.Module):
    """
    Single head self attention.
    """

    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.scale = in_dim ** -0.5
        self.QKV = nn.Linear(in_dim, hid_dim * 3, bias=False)
        self.out = nn.Linear(hid_dim, out_dim)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, in_dim]
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        sim = torch.einsum("bqd, bkd -> bqk", q, k)
        if mask is not None:
            sim.masked_fill_(mask, -torch.finfo(q.dtype).max)  # why not just '-inf'
        att = sim.softmax(dim=-1)
        out = torch.einsum("bqk, bkd -> bqd", att, v)
        return self.out(out)


class SpacialGatingUnit(nn.Module):
    init_eps = 1e-3

    def __init__(self, d_z, seq_len, act):
        super().__init__()
        self.norm = nn.LayerNorm(d_z // 2)
        self.model = nn.Conv1d(seq_len, seq_len, 1)  # O(n^2)
        self.act = act
        # ---- Initialization ----
        # It is important to initialize weights to small values and bias to 1, so that
        # during the initial training s(·) is close to identity.
        init_eps = self.init_eps / seq_len
        nn.init.uniform_(self.model.weight, -init_eps, init_eps)
        nn.init.constant_(self.model.bias, 1.0)

    def forward(self, z, gate_res=None, mask=None):
        # z: [batch_size, seq_len, d_z]. mask: [seq_len, seq_len, 1]
        seq_len = z.shape[1]
        # split into two parts of equal size along the last dimension (the dimension of word/patch embedding or of channel if you use conv for embedding).
        res, gate = torch.chunk(z, 2, dim=-1)
        gate = self.norm(gate)
        weight, bias = self.model.weight, self.model.bias

        if mask is not None:
            # it's okay for shorter seq_len
            weight, bias = weight[:seq_len, :seq_len], bias[:seq_len]
            weight = weight.masked_fill(mask, 0)

        gate = F.conv1d(gate, weight, bias)  # [batch_size, seq_len, d_z]
        # Here the last dimension of mask must be 1, i.e. each sample in the batch must
        # have the same mask. If you want to enable sample specific mask, expand weight
        # and use `einsum` instead:
        # gate = torch.einsum('qkb, kbd -> qbd', weight, gate) + self.bias[:seq_len, None, None]
        if gate_res is not None:
            gate += gate_res

        return self.act(gate) * res


class gMLPBlock(nn.Module):
    def __init__(self, emb_dim, hid_dim, seq_len, att_dim=None, act=nn.Identity()) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.proj_in = nn.Sequential(nn.Linear(emb_dim, hid_dim), nn.GELU())
        if att_dim is not None:
            self.att = Attention(emb_dim, att_dim, hid_dim // 2)
        else:
            self.att = None
        self.sgu = SpacialGatingUnit(hid_dim, seq_len, act=act)
        self.proj_out = nn.Linear(hid_dim // 2, emb_dim)
        self.size = emb_dim

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, emb_dim]. mask: [batch_size, seq_len, seq_len]
        shortcut = x
        x = self.norm(x)
        gate_res = self.att(x, mask=mask) if self.att is not None else None
        x = self.proj_in(x)
        x = self.sgu(x, gate_res=gate_res, mask=mask)
        x = self.proj_out(x)
        return x + shortcut


class gMLP(nn.Module):
    def __init__(
        self,
        num_layers,
        seq_len,
        emb_dim,
        expansion,
        att_dim,
        prob_survival=1.0,
        act=nn.Identity(),
    ):
        super().__init__()
        self.prob_survival = prob_survival
        self.layers = nn.ModuleList(
            [
                gMLPBlock(
                    emb_dim=emb_dim,
                    hid_dim=emb_dim * expansion,
                    seq_len=seq_len,
                    att_dim=att_dim,
                    act=act,
                )
                for _ in range(num_layers)
            ]
        )

    def dropout(self):
        to_drop = torch.rand(len(self.layers)) > self.prob_survival
        if all(to_drop):
            to_drop[torch.randint(self.layers, size=(1,))] = False
        layers = [layer for layer, drop in zip(self.layers, to_drop) if not drop]
        return layers

    def forward(self, x, mask=None):  # [batch_size, seq_len]
        x = self.embedding(x)
        layers = self.dropout() if self.training else self.layers
        for layer in layers:
            x = layer(x, mask)
        # x = nn.Sequential(*layers)(x)  # is this even possible???
        return self.head(x)


class gMLPVision(gMLP):
    def __init__(
        self,
        num_layers,
        in_channels,
        input_size,
        patch_size,
        emb_dim,
        expansion,
        att_dim,
        num_classes,
        prob_survival=1.0,
        act=nn.Identity(),
    ):
        assert not input_size % patch_size
        num_patches = (input_size // patch_size) ** 2
        super().__init__(
            num_layers, num_patches, emb_dim, expansion, att_dim, prob_survival, act
        )
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim, patch_size, patch_size),
            Rearrange("b d h w -> b (h w) d"),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            Reduce("b c d -> b d", "mean"),  # global average pooling
            nn.Linear(emb_dim, num_classes),
        )


class gMLPNLP(gMLP):
    """Batch first manner, in in concordance with vision."""

    def __init__(
        self,
        num_layers,
        num_tokens,
        seq_len,
        emb_dim,
        expansion,
        att_dim,
        prob_survival=1.0,
        act=nn.Identity(),
    ):
        super().__init__(
            num_layers,
            seq_len,
            emb_dim,
            expansion,
            att_dim,
            prob_survival=prob_survival,
            act=act,
        )
        self.embedding = nn.Embedding(num_tokens, emb_dim)
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_tokens))


def test_gmlp():
    # ---- shared hyperparams ----
    num_layers = 3
    emb_dim = 10
    expansion = 2
    att_dim = 5
    batch_size = 11

    input_size = 64
    patch_size = 16
    in_channels = 3
    num_classes = 10
    model = gMLPVision(
        num_layers,
        in_channels,
        input_size,
        patch_size,
        emb_dim,
        expansion,
        att_dim,
        num_classes,
    )
    data = torch.randn(batch_size, in_channels, input_size, input_size)
    rprint(model(data).shape)  # [batch_size, num_classes]

    num_tokens = 2000
    seq_length = 1024
    model = gMLPNLP(num_layers, num_tokens, seq_length, emb_dim, expansion, att_dim)
    data = torch.randint(num_tokens, size=(batch_size, seq_length))
    rprint(model(data).shape)  # [batch_size, seq_length, num_tokens]


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
    import pytorch_lightning as pl

    pl.seed_everything(42)

    test_gmlp()
