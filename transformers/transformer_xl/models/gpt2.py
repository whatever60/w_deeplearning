import torch
from torch import nn

class Block(nn.Module):
    """No dropout, Layernorm first
    ====
    Layernorm
    Multi-head attnntion
    add
    ----
    Layernorm
    Feedforward
    add
    ====
    """
    forward_expansion = 4
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * self.forward_expansion),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        self.norm0 = nn.LayerNorm(emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
    
    def forward(self, x):  # [seq_length, batch_size, emb_dim]
        seq_length = x.shape[0]
        attn_mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool()
        x = self.norm0(x)
        # [seq_length, batch_size, emb_dim]
        a, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + a
        x = self.norm1(x)
        m = self.mlp(x)
        return x + m

    
class GPT2(nn.Module):
    def __init__(self, vocab_size, num_layers, emb_dim, num_heads, num_positions):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedding = nn.Embedding(num_positions, emb_dim)
        self.layers = nn.ModuleList([
            Block(emb_dim, num_heads) for _ in range(num_layers)
        ])
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
    
    def forward(self, x, return_att=False):  # [seq_length, batch_size]
        # `return_att` is here for API consistency.
        seq_length = x.shape[0]
        w_emb = self.word_embedding(x)
        p_emb = self.positional_embedding(torch.arange(seq_length, device=x.device)).unsqueeze_(dim=1)
        h = w_emb + p_emb
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
    

def test_gpt2():
    vocab_size = 100
    num_layers = 4
    emb_dim = 16
    num_heads = 2
    num_positions = 20
    batch_size = 6
    model = GPT2(vocab_size, num_layers, emb_dim, num_heads, num_positions)
    input_idxs = torch.randint(vocab_size, (num_positions, batch_size))
    rprint(model(input_idxs).shape)  # [20, 6, 100]


if __name__ == '__main__':
    from rich import print as rprint
    from rich.traceback import install
    install()
    test_gpt2()
