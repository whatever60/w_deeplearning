# from dataclasses import dataclass, field
# from abc import ABCMeta, abstractmethod
import torch
from torch import nn
# @dataclass
# class C:
#     x: int
#     y: int = field(repr=False)
#     z: int = field(repr=False, default=10)
#     t: int = 20


# # @dataclass
# class C2(C):
#     x: int

# class A():
#     # a = 1
#     def __init__(self):
#         # ...
#         pass

#     @property
#     def a(self):
#         pass

#     @property
#     @abstractmethod
#     def b(self):
#         pass


# class B(A):
#     a = 1
#     b = 1
#     # def b(self):
#     #     pass

class DISC(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(10, 1)
    def forward(self, x):
        return self.model(x).reshape(-1)


class GEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 10)
    def forward(self, x):
        return self.model(x)


disc = DISC()
gen = GEN() 

opt_g = torch.optim.Adam(gen.parameters())
opt_d = torch.optim.Adam(disc.parameters())
in_z = torch.randn(10, 1)
criterion = nn.BCELoss()

fake = gen(in_z)
disc_fake = disc(fake.detach())
d_loss = criterion(disc_fake, torch.zeros_like(disc_fake))
opt_d.zero_grad()
d_loss.backward()
opt_d.step()

disc_fake = disc(fake)
g_loss = criterion(disc_fake, torch.ones_like(disc_fake))
opt_g.zero_grad()
g_loss.backward()
opt_g.step()
