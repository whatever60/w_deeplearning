import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, emb_dim, commitment_cost):
        super().__init__()
        self.num_embddings = num_embeddings
        self.emb_dim = emb_dim
        self.commitment_cost = commitment_cost  # beta

        self.emb = nn.Parameter(torch.zeros(num_embeddings, emb_dim))

    def init_weights(self):
        thres = 1 / self.num_embddings
        nn.init.uniform_(self.emb, -thres, thres)
    
    def forward(self, x):  # [b, emb_dim, h, w]
        x = x.permute(0, 2, 3, 1).reshape(-1, self.emb_dim)  # [n, emb_dim]
        # [n, num_embeddings]
        distances = torch.cdist(x.unsqueeze(0), self.emb.unsqueeze(0)).squeeze(0)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.emb.index_select(0, encoding_indices)  # [n, emb_dim]
