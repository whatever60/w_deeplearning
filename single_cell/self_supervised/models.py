import math
import torch
from torch import nn
from torch.nn.functional import normalize


class NPIDHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return normalize(self.model(x))


class MLP(nn.Module):
    def __init__(self, in_dim, dims=(800, 400, 200)):
        super().__init__()
        layers = []
        for out_dim in dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        self.out_dim = out_dim
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NPID(nn.Module):
    def __init__(self, in_dim, dims, feature_dim, memory_length, nce_m):
        super().__init__()
        self.backbone = MLP(in_dim, dims)
        self.head = NPIDHead(self.backbone.out_dim, feature_dim)
        stdv = 1 / math.sqrt(feature_dim / 3)
        self.register_buffer(
            "memory", torch.rand(memory_length, feature_dim) * 2 * stdv - stdv
        )
        self.new_memory = self.new_indices = None
        self.nce_m = nce_m

    def forward(self, x, indices=None):
        repres = self.head(self.backbone(x))
        if indices is not None:
            self.new_memory = repres.detach()
            self.new_indices = indices
        return repres

    @torch.no_grad()
    def update_memory(self) -> None:
        self.memory[self.new_indices] = normalize(
            self.memory[self.new_indices] * self.nce_m
            + self.new_memory * (1 - self.nce_m)
        )
        self.new_memory = self.new_indices = None
