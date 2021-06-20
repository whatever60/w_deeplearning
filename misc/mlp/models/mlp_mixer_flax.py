import einops
from flax import linen as nn
import jax
from jax import numpy as jnp


class MLPBlock(nn.Module):
    hid_dim: int

    @nn.compact
    def __call__(self, x):  # [batch_size, dim1, dim2]
        y = nn.Dense(self.hid_dim)(x)  # [batch_size, dim1, hid_dim]
        y = nn.gelu(y)
        # [batch_size, dim1, dim2]. Back to original shape
        return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
    tokens_hid_dim: int
    channels_hid_dim: int

    @nn.compact
    def __call__(self, x):  # [batch_size, num_tokens, hid_dim]
        y = nn.LayerNorm()(x)  # LayerNorm along the patch embedding
        y = jnp.swapaxes(y, 1, 2)
        y = MLPBlock(self.tokens_hid_dim, name="token_mixing")(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y

        y = nn.LayerNorm()(x)
        y = MLPBlock(self.channels_hid_dim, name="channel_mixing")(y)
        return x + y


class MLPMixer(nn.Module):
    num_classes: int
    num_blocks: int
    patch_size: int
    hid_dim: int
    tokens_hid_dim: int
    channels_hid_dim: int

    @nn.compact
    def __call__(self, x):  # [batch_size, h, w, c].
        # Note this is TensorFlow layout different from pytorch
        s = self.patch_size
        # [batch_size, num_tokens_x, num_tokens_y, hid_dim]. patch-embedding. conv with sxs kernel and sxs stride.
        x = nn.Conv(self.hid_dim, (s, s), strides=(s, s), name="stem")(x)
        # [batch_size, num_tokens, hid_dim]. Merge height and width to collect all patches into one dimension.
        x = einops.rearrange(x, "n h w c -> n (h w) c")
        for _ in range(self.num_blocks):
            # a series of MixerBlocks with the same token hidden dim and channel hidden dim
            x = MixerBlock(self.tokens_hid_dim, self.channels_hid_dim)(x)
        # one last LayerNorm
        x = nn.LayerNorm(name="pre_head_layer_norm")(x)
        # global average pooling on the token dimension
        x = jnp.mean(x, axis=1)  # [batch_size, hid_dim]
        # classification head
        logit = nn.Dense(
            self.num_classes, name="head", kernel_init=nn.initializers.zeros
        )(x)
        return logit


def test_mlpmixer():
    num_classes = 10
    patch_size = 16
    batch_size = 20
    input_shape = (224, 224, 3)
    model = MLPMixer(num_classes, 2, patch_size, 100, 50, 50)
    key = jax.random.PRNGKey(0)
    out, params = model.init_with_output(key, jnp.ones((batch_size, *input_shape)))
    rprint(out.shape)  # (batch_size, num_classes)


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
    jax.config.update("jax_platform_name", "cpu")

    test_mlpmixer()
