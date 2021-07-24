import torch
from torch import nn
import einops
from einops.layers.torch import Rearrange, Reduce


class SwinTransformer(nn.Module):
    def __init__(self, in_channels, hid_dim, layers, heads, head_dim, window_size, downscaling_factors, relative_pos_emb=True):
        super().__init__()
        assert len(layers) == len(heads) == len(downscaling_factors) == 4
        out_channels = hid_dim
        for i in range(1, 5):
            module = StageModule(in_channels, out_channels, layers[i], downscaling_factors[i], heads[i], head_dim, window_size, relative_pos_emb)
            in_channels = out_channels
            out_channels *= 2
            self.add_module(f'layer{i}', module)
    
    def forward(self, x):
        for i in range(1, 5):
            x = getattr(self, f'layer{i}')(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=downscaling_factor, stride=downscaling_factor),
        #     Rearrange('b c h w -> b h w c')
        # )
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)
        self.downscaling_factor = downscaling_factor
    
    def forward(self, x):  # [batch_size, in_channels, h, w]
        # x = self.model(x)
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x)
        x = x.view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


def test_patch_merging():
    data = torch.randn(5, 3, 224, 224)
    model = PatchMerging(3, 64, 4)
    rprint(model(data).shape)


class StageModule(nn.Module):
    """
    Composed of one `PatchMerging` and several regular `SwinBlock` + shifted `SwinBlock` 
    pair. 
    """
    def __init__(self, in_channels, hid_dim, layers: int, downscaling_factor, num_heads, head_dim, window_size, relative_pos_emb):
        super().__init__()
        assert not layers % 2
        self.patch_partition = PatchMerging(in_channels, hid_dim, downscaling_factor)
        self.layers = layers
        for i in range(1, layers // 2 + 1):
            regular_block = SwinBlock(hid_dim, num_heads, head_dim, head_dim * 4, False, window_size, relative_pos_emb)
            shifted_block = SwinBlock(hid_dim, num_heads, head_dim, head_dim * 4, True, window_size, relative_pos_emb)
            self.add_module(f'regular_block{i}', regular_block)
            self.add_module(f'shifted_block{i}', shifted_block)
    
    def forward(self, x):
        x = self.patch_partition(x)
        for i in range(1, self.layers // 2 + 1):
            x = getattr(self, f'regular_block{i}')(x)
            x = getattr(self, f'shifted_block{i}')(x)

        return x.permute(0, 3, 1, 2)


class Residual(nn.Module):
    """
    Residual wrapper.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_emb):
        super().__init__()
        module = WindowAttention(dim, heads, head_dim, shifted, window_size, relative_pos_emb)
        self.attention_block = Residual(PreNorm(dim, module))
        module = FeedForward(dim, mlp_dim)
        self.mlp_block = Residual(PreNorm(dim, module))
    
    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x
    

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, shifted, window_size, relative_pos_emb):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_emb = relative_pos_emb

        displacement = window_size // 2
        self.cyclic_shift = CyclicShift(-displacement)
        self.cyclic_back_shift = CyclicShift(displacement)
        self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=True, left_right=False), requires_grad=False)  # [49, 49]
        self.left_right = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=False, left_right=True), requires_grad=False)  # [49, 49]

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.rearragne1 = Rearrange('b (w_h w1) (w_w w2) (h d) -> b h (w_h w_w) (w1 w2) d', h=num_heads, w1=window_size, w2=window_size)
        self.rearrange2 = Rearrange('b h (w_h w_w) (w1 w2) d -> b (w_h w1) (w_w w2) (h d)', h=num_heads, w1=window_size, w2=window_size)

        if self.relative_pos_emb:
            self.relative_indices = get_relative_distances(window_size) + window_size
            self.pos_emb = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_emb = nn.Parameter(torch.randn(2 * window_size, 2 * window_size))

        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, x):  # [batch_size, h, w, c, num_heads]
        x = self.cyclic_shift(x)

        b, h, w, _ = x.shape
        window_h = h // self.window_size
        window_w = w // self.window_size

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = self.rearrange(q)
        k = self.rearrange(k)
        v = self.rearragne(v)

        dots = torch.einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_emb:
            dots += self.pos_emb[self.relative_indicesp[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_emb
        
        dots[:, :, -window_w] += self.upper_lower_mask
        dots[:, :, window_w-1::window_w] += self.left_right_mask
        attn = dots.softmax(dim=-1)  # [batch_size, num_heads, c, num_tokens, num_tokens]


def get_relative_distances(window_size):
    """
    This is a smart function. Especially the last line.
    Examples:
        >>> distances = get_relative_distance(2)
        >>> distances.shape
        torch.Size([9, 9, 2])
        >>> distances
        tensor([[[ 0,  0],
                 [ 0,  1],
                 [ 1,  0],
                 [ 1,  1]],

                [[ 0, -1],
                 [ 0,  0],
                 [ 1, -1],
                 [ 1,  0]],

                [[-1,  0],
                 [-1,  1],
                 [ 0,  0],
                 [ 0,  1]],

                [[-1, -1],
                 [-1,  0],
                 [ 0, -1],
                 [ 0,  0]]])
    """
    indices = torch.arange(window_size)
    indices = torch.cartesian_prod(indices, indices)
    return indices[None, :, :] - indices[:, None, :]




class SwinHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(in_dim * 8),
            nn.Linear(in_dim * 8, out_dim)
        )

    def forward(self, x):  # [batch_size, in_dim]
        return self.model(x)


def swin_t(**kwargs):
    return SwinTransformer(hid_dim=96, layers=(2,2,6,2),heads=(3,6,12,24), **kwargs)

# more layers
def swin_s(**kwargs):
    return SwinTransformer(hid_dim=96, layers=(2,2,18,2), heads=(3,6,12,24), **kwargs)

# higher hidden dimension and more heads
def swin_b(**kwargs):
    return SwinTransformer(hid_dim=128, layers=(2,2,18,2), heads=(4,8,16,32), **kwargs)

# higher hidden dimension and more heads
def swin_l(**kwargs):
    return SwinTransformer(hid_dim=192, layers=(2,2,18,2), heads=(6,12,24,48), **kwargs)


if __name__ == '__main__':
    from rich import print as rprint
    from rich.traceback import install
    import pytorch_lightning as pl

    install()
    pl.seed_everything(42)
    
    test_patch_merging()