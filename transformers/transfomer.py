import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, head_dim):
        super().__init__()
        assert (
            not emb_dim % head_dim
        ), "Embedding dimension needs to be divisible by number of heads"
        self.num_heads = emb_dim // head_dim
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.W_q = nn.Linear(head_dim, head_dim, bias=False)
        self.W_k = nn.Linear(head_dim, head_dim, bias=False)
        self.W_v = nn.Linear(head_dim, head_dim, bias=False)
        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = v.shape[0]
        # [batch_size, query_length, num_heads, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        # [batch_size, key_length, num_heads, head_dim]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        # [batch_size, value_length, num_heads, head_dim]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)  # Share weights among different heads and input embeddings.
        attention = torch.einsum("nqhd, nkhd -> nqkh", q, k)
        if mask is not None:
            attention = attention.masked_fill(~mask, -float("inf"))
        # Normalize across the source sentence.
        attention = torch.softmax(attention / (self.emb_dim ** 0.5), dim=3)
        # key and value always have the same length.
        return self.fc(
            torch.einsum("nqvh, nvhd -> nqhd", attention, v).reshape(
                batch_size, -1, self.emb_dim
            )
        )  # cannot use view here


class TransformerBlock(nn.Module):
    """
    ====
    Multi-head attention
    Add
    Layernorm
    Dropout
    ----
    Feed-forward
    Add
    Layernorm
    Dropout
    ====
    """

    forward_expansion = 4

    def __init__(self, emb_dim, head_dim, p):
        super().__init__()
        self.multi_head_attention = SelfAttention(emb_dim, head_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, self.forward_expansion * emb_dim),
            nn.ReLU(),
            nn.Linear(self.forward_expansion * emb_dim, emb_dim),
        )
        self.norm0 = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(p))
        self.norm1 = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(p))

    def forward(self, q, k, v, mask=None):
        q = self.norm0(self.multi_head_attention(q, k, v, mask) + q)
        return self.norm1(self.feed_forward(q) + q)


class Encoder(nn.Module):
    def __init__(
        self, vocab_size_src, num_layers, emb_dim, head_dim, p, seq_length
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size_src, emb_dim)
        # if you use nn.Embedding instead of like sinusoid, seq_length will be fixed for
        # the entire model.
        self.positional_embedding = nn.Embedding(seq_length, emb_dim)
        self.dropout = nn.Dropout(p)
        self.layers = nn.ModuleList(
            [TransformerBlock(emb_dim, head_dim, p) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze_(0)
        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    """
    Masked multi-head attention
    Add
    Layernorm
    TransformerBlock
    """
    def __init__(
        self,
        emb_dim,
        head_dim,
        p
    ):
        super().__init__()
        self.self_attention = SelfAttention(emb_dim, head_dim)
        self.norm = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(p))
        self.transformer_block = TransformerBlock(emb_dim, head_dim, p)

    def forward(self, q, v, k, src_mask, trg_mask):
        # trg_mask is a must. src_mask is used when padding is applied.
        attention = self.self_attention(q, q, q, trg_mask)
        q = self.norm(attention + q)
        return self.transformer_block(q, k, v, src_mask)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size_trg,
        num_layers,
        emb_dim,
        head_dim,
        p,
        seq_length
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size_trg, emb_dim)
        self.positional_embedding = nn.Embedding(seq_length, emb_dim)
        self.dropout = nn.Dropout(p)
        self.layers = nn.ModuleList(
            [DecoderBlock(emb_dim, head_dim, p) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(emb_dim, vocab_size_trg)

    def forward(self, q, enc_out, src_mask, trg_mask):
        positions = torch.arange(0, q.shape[1]).unsqueeze_(0)
        out = self.dropout(self.word_embedding(q) + self.positional_embedding(positions))
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)
        return self.fc(out)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size_src: int,
        vocab_size_trg: int,
        src_pad_idx: int,
        trg_pad_idx: int,
        num_layers: int,
        emb_dim: int,
        head_dim: int,
        p: int,
        seq_length: int  # all hyper-parameters are int.
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size_src,
            num_layers,
            emb_dim,
            head_dim,
            p,
            seq_length
        )
        self.decoder = Decoder(
            vocab_size_trg,
            num_layers,
            emb_dim,
            head_dim,
            p,
            seq_length
        )  # we use the exact same set of parameters here, for simplicity.
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
    
    def make_src_mask(self, src):  # [batch_size, src_length]
        batch_size, seq_length = src.shape
        # [batch_size, 1, src_length, 1]. Mask on the key dimension. The dimension order 
        # of mask must correspond to the actual implementation of self-attention.
        # 0 means mask. 
        return (src != self.src_pad_idx).view(batch_size, 1, seq_length, 1)
    
    def make_trg_mask(self, trg):
        batch_size, seq_length = trg.shape
        mask_decoder = torch.tril(torch.ones(seq_length, seq_length)).view(1, seq_length, seq_length, 1)
        # [batch_size, trg_length, trg_length, 1]
        mask_padding = (trg != self.trg_pad_idx).view(batch_size, 1, seq_length, 1)
        return (mask_decoder * mask_padding).bool()

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        return self.decoder(trg, enc_src, src_mask, trg_mask)
        

def test():
    from rich import print as rprint
    from rich.traceback import install
    install()

    model = Transformer(
        vocab_size_src=10,
        vocab_size_trg=10,
        src_pad_idx=0,
        trg_pad_idx=0,
        num_layers=6,
        emb_dim=512,
        head_dim=64,
        p=0,
        seq_length=100
    )
    
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]])
    rprint(x.shape)  # [2, 9]
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])
    rprint(trg.shape)  # [2, 8]
    out = model(x, trg)
    rprint(out.shape)  # [2, 8, 10]


if __name__ == '__main__':
    test()
