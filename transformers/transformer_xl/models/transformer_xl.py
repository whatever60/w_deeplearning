import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, pos_dim):
        # In TransformerXL, pos_dim must be the same as emb_dim, because positional
        # embedding will be added with word embedding instead of concatenated.
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0, pos_dim, 2) / pos_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        angle = torch.einsum("i, j -> ij", positions, self.inv_freq)
        # [seq_length, 1, pos_dim]
        return torch.cat([angle.sin(), angle.cos()], dim=1).unsqueeze(dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, head_dim, num_heads, p=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.Wq = nn.Linear(emb_dim, head_dim * num_heads, bias=False)
        self.Wkv = nn.Linear(
            emb_dim, head_dim * num_heads * 2, bias=False
        )  # get both keys and values at once for efficiency
        self.Wp = nn.Linear(
            emb_dim, head_dim * num_heads, bias=False
        )  # for positional embeddings

        self.scale = 1 / (head_dim ** 5)
        self.dropout = nn.Dropout(p)
        self.norm = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(head_dim * num_heads, emb_dim, bias=False)

    def _rel_shift(self, x):
        """
        Compute relative positional embedding for each key-query pair
        """
        zero_pad = torch.zeros_like(x[:, 0:1])
        return (
            torch.cat([zero_pad, x], dim=1)
            .view(x.shape[1] + 1, x.shape[0], *x.shape[2:])[1:]
            .view_as(x)
        )

    def forward(
        self,
        emb_new,  # [cur_seq_length, batch_size, emb_dim]
        emb_old,  # [prev_seq_length, batch_size, emb_dim]
        emb_pos,  # [cur_seq_length + prev_seq_length, emb_dim]
        u_,  # [num_heads, head_dim]
        v_,  # [num_heads, head_dim]
        mask=None,
    ):
        # In essence, emb_new is just query, emb_mem is just key and value.
        batch_size = emb_new.shape[1]
        # concatenate recurrent memory across sequence length.
        emb = torch.cat([emb_old, emb_new], dim=0)
        k, v = torch.chunk(self.Wkv(emb), 2, dim=-1)
        # cannot use reshape here
        k = k.reshape(-1, batch_size, self.num_heads, self.head_dim)
        v = v.reshape_as(k)
        q = self.Wq(emb_new).view(-1, batch_size, self.num_heads, self.head_dim)
        p = self.Wp(emb_pos).view(-1, self.num_heads, self.head_dim)
        # i for the total length of memory and current sequences.
        attention_content = torch.einsum("qbhd, ibhd -> qibh", q + u_, k)
        attention_position = self._rel_shift(
            torch.einsum("qbhd, ihd -> qibh", q + v_, p)
        )
        # final attention is the summation of these
        attention = attention_content + attention_position
        if mask is not None:
            attention = attention.masked_fill(mask, -float("inf"))
        attention = torch.softmax(attention * self.scale, dim=1)
        attention = self.dropout(attention)
        # cannot use view here
        output = torch.einsum("qibh, ibhd -> qbhd", attention, v).flatten(start_dim=2)
        return self.fc(output)  # project back and add residual and norm


def test_self_attention():
    # embedding dimension is 32, 4 attention heads, 17 dimensions per head
    model = MultiHeadAttention(32, 17, 4)
    # current sequence length is 7, batch size is 3
    emb_new = torch.randn(7, 3, 32)
    emb_old = torch.randn(6, 3, 32)  # old sequence length is 6
    emb_pos = torch.randn(6 + 7, 32)
    u_, v_ = torch.randn(4, 17), torch.randn(4, 17)
    rprint(model(emb_new, emb_old, emb_pos, u_, v_).shape)  # [7, 3, 32]


class DecoderBlock(nn.Module):
    forward_expansion = 4

    def __init__(self, emb_dim, head_dim, num_heads, p_att, p_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(emb_dim, head_dim, num_heads, p=p_att)
        self.dropout = nn.Dropout(p_ff)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim * self.forward_expansion)),
            nn.ReLU(inplace=True),
            nn.Linear(int(emb_dim * self.forward_expansion), emb_dim),
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, emb_new, emb_mem, emb_pos, u, v, mask=None):
        attention_output = self.norm(
            self.dropout(
                self.self_attention(emb_new, emb_mem, emb_pos, u, v, mask=mask)
            )
            + emb_new
        )
        return self.norm(
            self.dropout(self.feed_forward(attention_output)) + attention_output
        )


class StandardWordEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        div_val=1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.scale = emb_dim ** 0.5

    def forward(self, token_ids):
        return self.embedding(token_ids) * self.scale


class TransformerXL(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        emb_dim,
        head_dim,
        num_heads,
        p_ff=0.1,
        p_att=0.0,
        memory_length=512,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.word_embedding = StandardWordEmbedding(vocab_size, emb_dim)
        self.positional_embedding = PositionalEmbedding(emb_dim)
        self.dropout = nn.Dropout(p_ff)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(emb_dim, head_dim, num_heads, p_att, p_ff)
                for _ in range(num_layers)
            ]
        )
        # share weight between the embedding layer and the final projection head
        self.projection_head = nn.Linear(emb_dim, vocab_size)
        self.projection_head.weight = self.word_embedding.embedding.weight
        self.u = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.v = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.memory = None
        self.memory_length = memory_length

    @torch.no_grad()
    def update_memory(self, hidden_states) -> None:
        """New tensor will be created as memory each iteration, so we don't need to care
        about memory changing before backprop.
        """
        new_memory = []
        end_idx = self.memory[0].shape[0] + hidden_states[0].shape[0]
        # self.memory_length can change, so we re-calculate indices each iteration
        begin_idx = max(0, end_idx - self.memory_length)
        for m, h in zip(self.memory, hidden_states):
            new_memory.append(torch.cat([m, h], dim=0)[begin_idx:end_idx].detach())
        self.memory = new_memory

    def forward(self, input_idxs):
        # input_idxs: [seq_length, batch_size]
        if self.memory is None:  # init memory
            self.memory = [
                torch.empty(0, device=input_idxs.device)
                for _ in range(self.num_layers + 1)
            ]
        seq_length = input_idxs.shape[0]
        memory_length = self.memory[0].shape[0]
        # [seq_length, batch_size, emb_dim]
        emb_word = self.dropout(self.word_embedding(input_idxs))
        # Because seq_length and memory_length can be changed on the fly (although I
        # don't know why we would do this), pos_idxs is not a constant and thus we have
        # to generate it once per step.
        pos_idxs = torch.arange(
            memory_length + seq_length - 1, -1, -1, device=emb_word.device
        )
        # [seq_length, 1, emb_dim]. Dropout respectively because emb_word and emb_pos
        # will pass throught different linear layers before added together.
        emb_pos = self.dropout(self.positional_embedding(pos_idxs))

        hidden_states = [emb_word]
        layer_out = emb_word
        mask = (
            torch.triu(
                torch.ones(
                    seq_length, memory_length + seq_length, device=emb_word.device
                ),
                diagonal=1 + memory_length,
            )
            .bool()
            .unsqueeze_(-1)
            .unsqueeze_(-1)
        )
        for m, layer in zip(self.memory, self.layers):
            layer_out = layer(layer_out, m, emb_pos, self.u, self.v, mask=mask)
            hidden_states.append(layer_out)

        logits = self.projection_head(self.dropout(layer_out))
        self.update_memory(hidden_states)
        return logits


def test_transformer():
    model = TransformerXL(
        vocab_size=1000,
        num_layers=4,
        emb_dim=32,
        head_dim=17,
        num_heads=3,
        memory_length=7,
    )
    input_idxs = torch.randint(1000, (5, 9))  # 5 tokens per sentence, batch size 9
    rprint(model(input_idxs).shape)  # [5, 9, 1000]
    rprint(len(model.memory))  # [5]
    model(input_idxs)
    rprint(model.memory[0].shape)  # [7, 9, 32]


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
