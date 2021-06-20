from typing import List
import torch
from torch import nn


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True)
            ShiftedReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class ShiftedReLU(nn.Module):
    def __init__(self, shift=0.5, inplace=True):
        super().__init__()
        self.model = nn.ReLU(inplace=inplace)
        self.shift = shift
    
    def forward(self, x):
        return self.model(x - self.shift)


class DeepClusterV2(nn.Module):
    def __init__(self, backbone, hid_dim, out_dim, num_prototypes, normalize=True):
        super().__init__()
        self.backbone = backbone
        in_dim = backbone.out_dim
        self.normalize = normalize

        # projection head
        if not out_dim:
            self.projection_head = nn.Identity()
        if not hid_dim:
            self.projection_head = nn.Linear(in_dim, out_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(in_dim, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hid_dim, out_dim),
            )
        # prototype layer
        if isinstance(num_prototypes, int):
            num_prototypes = [num_prototypes]
        self.prototypes = nn.ModuleList(
            [nn.Linear(out_dim, k, bias=False) for k in num_prototypes]
        )

    def forward(self, inputs):
        # [[batch_size, 3, size_0, size_0], ..., [batch_size, 3, size_n, size_n], ...]
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        idx_crops = self.get_crop_idx([input_.shape[-1] for input_ in inputs])
        features = []
        start_idx = 0
        for i, end_idx in enumerate(idx_crops):
            features.append(self.backbone(torch.cat(inputs[start_idx:end_idx]), i))
            start_idx = end_idx
        embeddings = self.projection_head(torch.cat(features))
        embeddings = nn.functional.normalize(embeddings, dim=1, p=2)
        # [batch_size * sum(num_crops), out_dim], [batch_size * sum(num_crops), num_prototype] * len(num_prototypes)
        return embeddings, [prototype(embeddings) for prototype in self.prototypes]

    @staticmethod
    def get_crop_idx(sizes: list):
        """
        The inputs to DeepCluster/SwAV/... is a list of images with different size. For
        example, the images are of size [224, 224, 96, 96, 96, 96] (note that images of
        the same size are consecutive). This function returns the index pointers to each
        size (but without the first 0).

        >>> get_crop_idx([224, 224, 96, 96, 96, 96])
        tensor([2, 6])
        >>> get_crop_idx([32, 32])
        tensor([2])
        """
        return torch.cumsum(
            torch.unique_consecutive(
                torch.tensor(sizes),
                return_counts=True,
            )[1],
            0,
        )


def weight_init(m: nn.Module, zero_init_residual=False):

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    # if zero_init_residual:
    #     if isinstance(m, Bottleneck):
    #         nn.init.constant_(m.bn3.weight, 0)
    #     elif isinstance(m, BasicBlock):
    #         nn.init.constant_(m.bn2.weight, 0)

class MultiscaleMLP(nn.Module):
    def __init__(self, in_dims: List, dims: List):
        super().__init__()
        out_dim = dims[0]
        self.init1 = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim in in_dims])
        self.init2 = nn.Sequential(nn.BatchNorm1d(out_dim), ShiftedReLU(inplace=True))
        in_dim = out_dim
        layers = []
        for out_dim in dims[1:]:
            layers.append(LinearBlock(in_dim, out_dim))
            in_dim = out_dim
        self.out_dim = out_dim
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, i):  # [batch_size * num_crop, in_dims[i]]
        x = self.init2(self.init1[i](x))
        return self.model(x)


def test_deepclusterv2():
    model = DeepClusterV2(MultiscaleMLP([1000, 500], [128, 50]), 2048, 128, [300, 300, 300])
    batch_size = 10
    size_crops = [1000, 1000, 500, 500, 500, 500]
    inputs = [torch.randn(batch_size, size) for size in size_crops]
    outputs = model(inputs)
    rprint(len(outputs))
    rprint(outputs[0].shape)
    rprint(len(outputs[1]))
    rprint([output.shape for output in outputs[1]])


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
    import pytorch_lightning as pl

    pl.seed_everything(42)

    test_deepclusterv2()
