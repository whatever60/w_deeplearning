'''EfficientNet b0~b7
Aladdin Persson's implementation
https://www.youtube.com/watch?v=fR_0o25kigM
Tricks:
- Depthwise convolution
- Inverted residual block
- Swish
- Squeeze and excitation
- Stochastic depth
'''


import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from torchvision import transforms as T
from torchvision.datasets import CIFAR10


# for MBConv
base_model = [
    # [expand_ratio, channels, repeats, stride, kernel_size]
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

# depth = alpha ** phi, width = beta ** phi, resolution = gamma ** phi
# alpha = 1.2, beta = 1.1, gamma = 1.15 for b0
phi_values = dict(
    # (phi, input_resolution, drop_rate)
    b0=(0, 224, 0.2),
    b1=(0.5, 240, 0.2),  # 240 = 224 * 1.15 **  0.5
    b2=(1, 260, 0.3),
    b3=(2, 300, 0.3),
    b4=(3, 380, 0.4),
    b5=(4, 456, 0.4),
    b6=(5, 528, 0.5),
    b7=(6, 600, 0.5)
)
# dropout rate increases as model increases.


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,  # for depth-wise conv, 1 for normal conv, `in_channel` for depth-wise conv
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, **bn_config),
            nn.SiLU()  # Sigmoid Weighted Linear Unit
        )
    
    def forward(self, x):
        return self.block(x)


# compute "attention" score for each channel 
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),  # Excitation
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )  # how do we prioritize the value in each channel
    
    def forward(self, x):
        return x * self.se(x)


# mobile inverted bottleneck MBConv
class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,  # for depth-wise conv
        reduction=4,  # for squeeze excitation
        survical_prob=0.8,  # for stochastic depth
    ):
        super().__init__()
        self.survival_prob = survical_prob
        self.use_residual = in_channels == out_channels and stride == 1  # so that the dimension of input and output will match.
        self.expand = expand_ratio != 1
        hidden_dim = in_channels * expand_ratio
        reduced_dim = math.ceil(in_channels / reduction)

        if expand_ratio != 1:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, **bn_config)
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return x * binary_tensor / self.survival_prob

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    alpha = 1.2
    beta = 1.1

    def __init__(self, version, channels, num_classes):
        phi, _, dropout_rate = phi_values[version]
        super().__init__()
        depth_factor = self.alpha ** phi
        width_factor = self.beta ** phi
        last_channels = math.ceil(1280 * width_factor)

        self.features = self.create_features(channels, width_factor, depth_factor, last_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    def create_features(self, channels, width_factor, depth_factor, last_channels):
        in_channels = int(32 * width_factor)
        features = [CNNBlock(channels, in_channels, 3, stride=2, padding=1)]

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = int(channels * width_factor)
            layer_num = math.ceil(repeats * depth_factor)

            # if there is downsample between stages and layer number is greater than 1, downsample at the first layer of this stage
            for i in range(layer_num):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1 if i else stride,
                        padding=kernel_size // 2,  # to ensure same conv
                        expand_ratio=expand_ratio
                    )
                )
                in_channels = out_channels

        return nn.Sequential(
            *features,
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.shape[0], -1))


class Net(pl.LightningModule):
    def __init__(
        self,
        version,
        channels,
        num_classes,
        optim_config
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = EfficientNet(version, channels, num_classes)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            **self.hparams.optim_config
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return dict(loss=loss)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        return dict(val_loss=loss, val_acc=acc)


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, dims=(28, 28), batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dims = (3, *dims)
        self.num_classes = 10

        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(dims),
            T.Normalize(mean, std)
        ])

    def prepare_data(self):
        '''download, tokenize, etc
        This is called from one single GPU, so do not use it to assign state 
        '''
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        '''This is called from every GPU, so it is ok to set state here,
        '''
        if stage == 'fit' or stage is None:
            mnist_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [45000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=4
        )


if __name__ == '__main__':
    dm = CIFAR10DataModule('/home/tiankang/wusuowei/data')
    optim_config = dict(lr=1.25e-2, weight_decay=1e-4, momentum=0.9, nesterov=True)
    bn_config = dict(momentum=0.99, eps=1e-5)
    model = Net('b0', dm.size()[0], dm.num_classes, optim_config)
    trainer = pl.Trainer(max_epochs=10, gpus=[3], progress_bar_refresh_rate=20)
    trainer.fit(model, dm)
