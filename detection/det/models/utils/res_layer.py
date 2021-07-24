from typing import Optional
from ...cv.cnn import build_conv_layer, build_norm_layer
from ...cv.runner import BaseModule, Sequential

from torch import nn


class ResLayer(Sequential):
    """
    Build ResNet style backbone

    Args:
        block:
            Block used to build ResLayer.
        inplanes

        planes

        num_blocks

        stride

        avg_down:
            Use AvgPool instead of stride conv when downsampling in the bottleneck.

        conv_cfg

        norm_cfg

        downsample_first:
            Downsample at the first or last block. True for ResNet, False for Hourglass.
    """

    def __init__(
        self,
        block: nn.Module,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        avg_down: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN"),
        downsample_first: bool = True,
        **kwargs,
    ) -> None:
        # self.block = block

        # The following code serves the same function as the "infamous" `_make_layer` method.
        downsample = get_downsample(
            block, inplanes, planes, stride, avg_down, conv_cfg, norm_cfg
        )

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs,
                    )
                )
        else:
            # for Hourglass
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs,
                    )
                )
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
        super().__init__(*layers)


def get_downsample(block, inplanes, planes, stride, avg_down, conv_cfg, norm_cfg):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = []
        conv_stride = stride
        if avg_down:
            conv_stride = 1
            # why average pooling is specified like this?
            downsample.append(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False,
                )
            )
        downsample.extend(
            [
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, planes * block.expansion)[1],
            ]
        )
        downsample = nn.Sequential(*downsample)
    return downsample


class SimplifiedBasicBlock(BaseModule):
    """
    Simplified version of the original basic residual block, used in SCNet

    Modifications:
        - Norm layer becomes optional.
        - Last ReLU in the forward function is removed.

    """

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
        assert dcn is None, NotImplemented
        assert plugins is None, NotImplemented
        assert not with_cp, NotImplemented

        self.with_norm = norm_cfg is not None
        with_bias = True if norm_cfg is not None else False

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=with_bias,
        )
        if self.with_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=with_bias
        )
        if self.with_norm:
            self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
            self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplanes=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name) if self.with_norm else None

    @property
    def norm2(self):
        return getattr(self, self.norm2_name) if self.with_norm else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.with_norm:
            out = self.relu(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out
