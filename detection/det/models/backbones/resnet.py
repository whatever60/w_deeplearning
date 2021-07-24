from typing import Dict, List, Optional

# import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from ...cv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from ...cv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

# from ..builder import BACKBONES
from ..utils import ResLayer


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg)
        self._sanity_check(dcn, plugins)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)

        # Both channel increasing and downsampling, if any, take place at the first conv of a BasicBlock
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )

        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def _sanity_check(self, dcn, plugins):
        assert dcn is None, NotImplemented
        assert plugins is None, NotImplemented

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        # Why isn't relu included in the `_inner_forward`???
        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample: Optional[nn.Module] = None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn: Optional[dict] = None,
        plugins: Optional[list] = None,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg)
        self._sanity_check(style, dcn, plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin["cfg"]
                for plugin in plugins
                if plugin["position"] == "after_conv1"
            ]
            self.after_conv2_plugins = [
                plugin["cfg"]
                for plugin in plugins
                if plugin["position"] == "after_conv2"
            ]
            self.after_conv3_plugins = [
                plugin["cfg"]
                for plugin in plugins
                if plugin["position"] == "after_conv3"
            ]
        if self.style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        # why not directly assigning norm layer to attribute like conv???
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3
        )
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        self.add_module(self.norm3_name, norm3)

        # lower the number of channels
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
        )

        # downsampling
        fallback_on_stride = True
        if self.with_dcn:
            fallback_on_stride = dcn.pop("fallback_on_stride", False)
        if fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        else:
            assert self.conv_cfg is None, "conv_cfg must be None for DCN"
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        # raise the number of channels
        self.conv3 = build_conv_layer(
            conv_cfg, planes, planes * self.expansion, kernel_size=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins
            )
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins
            )
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins
            )

    def _sanity_check(self, style, dcn, plugins):
        assert style in ["pytorch", "caffe"]
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ["after_conv1", "after_conv2", "after_conv3"]
            assert all(p["position"] in allowed_position for p in plugins)

    def make_block_plugins(self, in_channels, plugins) -> List[str]:
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop("postfix", ""),  # `.get`?
            )
            assert not hasattr(self, name), f"duplicate plugin {name}"
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out  # only the last plugin gets its output returned???

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResNet(BaseModule):
    """

    Args:
        depth:
            One of (18, 34, 50, 101, 152). Depth also determines which block to use.
        in_channels:

        stem_channels:
            If None, it will be the same as `base_channels`. Namely, the output channel
            of stem (and input channel of the first stage) will be `stem_channels`.
        base_channels:
            Number of base channels of res layer. Namely, the output channel number of
            the first stage will be `base_channels * self.block.expansion`
        num_stages:

        strides:
            Stride of the first block of each stage.
        dilations:
            Dilation of each stage.
        out_indices:
            Output from with stages.
        style:

        deep_stem:
            Whether to replace the first 7x7 conv in stem with 3x3 conv.
        avg_down:
            Whether to use average pooling instead of strided conv when downsampling in
            the residual connection.
        frozen_stages:
            Stages to be frozen (stop grad and set to eval mode). -1 means not freezing
            any parameters.
        norm_cfg:

        norm_eval:
            Whether to set norm layers to eval mode, i.e., freeze running stats (mean
            and var).
            Note: effect only on BatchNorm and its variants.
        plugins:
            List of plugins for stages. Each dict contains:
                - cfg (required)
                - position (optional)
                - stages (optional): Stages to apply plugin. If not None, Length should
                    be the same as `num_stages`. If None, the plugin would be applied to
                    all stages.
            Examples:
                >>> plugins = [
                ...     dict(
                ...         cfg=dict(type='xxx', arg1='xxx'),
                ...         stages=(False, True, True, True),
                ...         position='after_conv2'
                ...     ),
                ...     dict(
                ...         cfg=dict(type='yyy'),
                ...         stages=(False, True, True, True),
                ...         position='after_conv3'
                ...     ),
                ...     dict(
                ...         cfg=dict(type='zzz', postfix='1'),
                ...         stages=(True, True, True, True),
                ...         position='after_conv3'
                ...     ),
                ...     dict(
                ...         cfg=dict(type='zzz', postfix='2'),
                ...         stages=(True, True, True, True),
                ...         position='after_conv3'
                ...     )
                ... ]
        with_cp

        zero_init_residual:
            Whether to use zero init for the last norm layer in a block to let them
            behave as identity.
        pretrained

        init_cfg
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg: dict = dict(type="BN", requires_grad=True),
        norm_eval: bool = True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins: Optional[List[Dict]] = None,
        with_cp=False,
        zero_init_residual=True,
        # pretrained=None,
        init_cfg=None,
    ) -> None:
        super().__init__()
        self._sanity_check(
            depth,
            num_stages,
            strides,
            dilations,
            out_indices,
            dcn,
            stage_with_dcn,
            plugins,
        )
        self._save_hyperparameters(
            depth,
            in_channels,
            stem_channels,
            base_channels,
            num_stages,
            strides,
            dilations,
            out_indices,
            style,
            deep_stem,
            avg_down,
            frozen_stages,
            conv_cfg,
            norm_cfg,
            norm_eval,
            dcn,
            stage_with_dcn,
            plugins,
            with_cp,
            zero_init_residual,
        )

        if init_cfg is None:
            self.init_cfg, block_init_cfg = self._default_cfg()

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2 ** i
            res_layer = ResLayer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                avg_down=self.avg_down,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                # ---- kwargs ----
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i+1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self._freeze_stages()
        self.feat_dim = self.block.expansion * base_channels * 2 ** (num_stages - 1)

    def _default_cfg(self):
        block_init_cfg = None
        init_cfg = [
            dict(type="Kaiming", layer="Conv2d"),
            dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"]),
        ]
        if self.zero_init_residual:
            if self.block is BasicBlock:
                block_init_cfg = dict(
                    type="Constant", val=0, override=dict(name="norm2")
                )
            elif self.block is Bottleneck:
                block_init_cfg = dict(
                    type="Constant", val=0, override=dict(name="norm3")
                )
        return init_cfg, block_init_cfg

    def make_stage_plugins(self, plugins, stage_idx) -> List[dict]:
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop("stages", None)
            if stages is None or stages[stage_idx] is True:
                stage_plugins.append(plugins)
        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        # why bother making a method for this?
        return ResLayer(**kwargs)

    def _make_stem_layer(self):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels,
                    self.stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, self.stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    self.stem_channels // 2,
                    self.in_channelsstem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, self.stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    self.stem_channels // 2,
                    self.stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, self.stem_channels)[1],
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                self.in_channels,
                self.stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, self.stem_channels, postfix=1
            )
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x) -> tuple:
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def _freeze_stages(self):
        # what is the difference between directly `freeze` and setting `eval` and `requires_grad_`???
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad_(False)
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad_(False)
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad_(False)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _sanity_check(
        self,
        depth,
        num_stages,
        strides,
        dilations,
        out_indices,
        dcn,
        stage_with_dcn,
        plugins,
    ):
        assert depth in self.arch_settings
        assert num_stages >= 1 and num_stages <= 4
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) <= num_stages
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        if plugins is not None:
            for plugin in plugins:
                stages = plugin.get("stages")
                assert stages is None or len(stages) == num_stages
        # assert not (init_cfg and pretrained)

    def _save_hyperparameters(
        self,
        depth,
        in_channels,
        stem_channels,
        base_channels,
        num_stages,
        strides,
        dilations,
        out_indices,
        style,
        deep_stem,
        avg_down,
        frozen_stages,
        conv_cfg,
        norm_cfg,
        norm_eval,
        dcn,
        stage_with_dcn,
        plugins,
        with_cp,
        zero_init_residual,
    ):
        self.depth = depth
        self.in_channels = in_channels
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.plugins = plugins
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels


class ResNetV1d(ResNet):
    """ResNetV1d variant described in `Bag of Tricks
    Modification:
        - replaces the 7x7 conv in the input stem with three 3x3 convs.
        - in the downsampling block, a 2x2 avg_pool with stride 2 is added before conv,
            whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=True, **kwargs)


def test_resnet():
    model = ResNet(depth=50)
    inputs = torch.randn(10, 3, 224, 224)
    level_outputs = model(inputs)
    for level_out in level_outputs:
        rprint(level_out.shape)
        # [batch_size, 256, 56]
        # [batch_size, 512, 28]
        # [batch_size, 1024, 14]
        # [batch_size, 2048, 7]


if __name__ == "__main__":
    import torch
    import pytorch_lightning as pl
    from rich import print as rprint
    from rich.traceback import install

    install()
    pl.seed_everything(42)
    test_resnet()
