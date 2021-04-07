from functools import singledispatchmethod
from typing import Iterable

import torch
from torch import nn
import pytorch_lightning as pl


"""
There will be three detection branches, for predicting small, medium and big objects.
Grid setting is the save as v2 (13 * 13), but in each grid there are 9 prior boxes, 3 for each detection branch.
Number of classes is the same as v2 (80)

How to read this architecture config:
- Tuple: (channels, kernel_size, stride).
Note: Every conv layer is a same convolution.
- List: ['B', number_of_repeats], here 'B' indicates a residual block.
- 'S' is for scale prediction block and computing the YOLO loss.
Note: this is an improvement over YOLOv2.
- 'U' is for upsampling the feature map and concatenating with a previous layer.
"""
# input shape at inference: (416, 416)
CONFIG = [  # 3 x 416
    (32, 3, 1),  # 32 x 416
    (64, 3, 2),  # 64 x 208
    ['B', 1],  # 64 x 208
    (128, 3, 2),  # 128 x 104
    ['B', 2],  # 128 x 104
    (256, 3, 2),  # 256 x 52
    ['B', 8],  # 256 x 52  -->
    (512, 3, 2),  # 512 x 26
    ['B', 8],  # 512 x 26  -->
    (1024, 3, 2),  # 1024 x 13
    ['B', 4],  # 13
    # ------- Darknet above, DBL below -------
    (512, 1, 1),  # 512 x 13
    (1024, 3, 1),  # 1024 x 13
    'S',  # (PRIOR_BOX_NUM x NUM_CLASSES 13 x 13) , 8 times downsampling, for predicting small objects
    (256, 1, 1),  # 256 x 13  
    'U',  # (256 x 26 x 26)  <-- (512 x 26 x 26)
    (256, 1, 1),  # (256 x 26 x 26)
    (512, 3, 1),  # (256 x 26 x 26)
    'S',  # (26, 26), 16 times downsampling, for predicting medium objects
    (128, 1, 1),
    'U',  # <--
    (128, 1, 1),
    (256, 3, 1),
    'S',  # (52, 52), 32 times downsampling, for predicting big objects
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn_act=True):
        super().__init__()
        if bn_act:
            model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            )
        else:
            model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.model = model

    def forward(self, x):
        return self.model(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, num_repeats, use_residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(nn.Sequential(
                CNNBlock(channels, channels // 2, kernel_size=1, stride=1, padding=0),
                CNNBlock(channels // 2, channels, kernel_size=3, stride=1, padding=1)
            ))
        self.num_repeats = num_repeats
        self.use_residual = use_residual

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x


class ScalePrediction(nn.Module):
    PRIOR_BOX_NUM = 3
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * self.PRIOR_BOX_NUM, kernel_size=1, stride=1, padding=0, bn_act=False)
        )
        self.num_classes = num_classes

    def forward(self, x):
        pred = self.pred(x)
        pred = pred.reshape(x.shape[0], self.PRIOR_BOX_NUM, self.num_classes + 5, x.shape[2], x.shape[3])
        return pred.permute(0, 1, 3, 4, 2)


class YOLOv3(nn.Module):
    def __init__(self, config, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model = nn.ModuleList()
        for i in config:
            layer = self.create_conv_layer(i)
            if isinstance(layer, Iterable):
                self.model.extend(layer)
            else:
                self.model.append(layer)
    
    def forward(self, x):
        outputs = []
        connections = []
        for layer in self.model:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
            else:
                x = layer(x)
                if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                    connections.append(x)
                elif isinstance(layer, nn.Upsample):
                    x = torch.cat([x, connections.pop()], dim=1)
        return outputs

    @singledispatchmethod
    def create_conv_layer(self, i):
        raise NotImplementedError
    
    @create_conv_layer.register
    def _(self, i: tuple):
        out_channels, kernel_size, stride = i
        padding = 1 if kernel_size == 3 else 0
        model =  CNNBlock(
            self.in_channels,
            *i,
            padding
        )
        self.in_channels = out_channels
        return model
    
    @create_conv_layer.register
    def _(self, i: list):
        num_repeats = i[1]
        return ResidualBlock(self.in_channels, num_repeats=num_repeats)

    @create_conv_layer.register
    def _(self, i: str):
        if i == 'S':
            # the reason we do not make these `ResidualBlock` and `CNNBlock` part of `ScalePrediction` is to make feedforward easier.
            model = nn.Sequential(
                ResidualBlock(self.in_channels, use_residual=False, num_repeats=1),
                CNNBlock(self.in_channels, self.in_channels // 2, kernel_size=1, stride=1, padding=0),
                ScalePrediction(self.in_channels // 2, num_classes=self.num_classes)
            )
            self.in_channels //= 2
            return model
        elif i == 'U':
            self.in_channels *= 3  # because of concatenation makes channel number larger.
            return nn.Upsample(scale_factor=2)
        else:
            raise NotImplementedError
            
    def test(self):
        x = torch.randn((10, 3, 416, 416))
        out = self(x)
        print([i.shape for i in out])


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()  # we do not have multiple labels here
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No object loss
        no_object_loss = self.bce(predictions[..., 0:1][noobj], target[..., 0:1][noobj])

        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)  # 3 x 2
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj], ious * target[..., 0:1]))
        
        # box coordinate loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(target[..., 3:5] / anchors + 1e-16)  # for better gradient flow
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # class loss
        class_loss = self.entropy(predictions[..., 5:][obj], target[..., 5][obj].long())

        return self.lambda_box * box_loss + self.lambda_obj * object_loss + self.lambda_noobj * no_object_loss + self.lambda_class * class_loss
        


if __name__ == '__main__':
    NUM_CLASSES = 20
    IN_CHANNELS = 3
    model = YOLOv3(CONFIG, in_channels=3, num_classes=20)
    model.test()
