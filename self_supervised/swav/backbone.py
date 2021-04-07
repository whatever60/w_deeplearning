from torch import nn


def build_conv3(in_channels, out_channels, stride, groups, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=False, dilation=padding)


def build_conv1(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample, groups, base_width, dilation, norm_layer):
        super().__init__()
        self.model = nn.Sequential(
            build_conv3(in_channels, out_channels, stride, groups, dilation),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            build_conv3(in_channels, out_channels, stride, groups, dilation),
            norm_layer(out_channels)
        )
        self.final_act = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        connection = x
        out = self.model(x)
        if self.downsample is not None:
            connection = self.downsample(x)
        out += connection
        return nn.functional.relu(out)




GROUPS = 1
BASE_WIDTH = 64
DILATION = 1

