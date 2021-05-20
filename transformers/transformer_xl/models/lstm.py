import torch
from torch import nn


class NaiveLSTM(nn.Module):
    """
    A naive one layer unidirectional LSTM, in length first format.
    """

    def __init__(self, emb_dim, hid_dim, num_layers):
        # `num_layers` for uniform interface
        super().__init__()
        self.hid_dim = hid_dim
        self.W_all = nn.Linear(emb_dim + hid_dim, hid_dim * 4)
        self.batch_dim = 1
        self.seq_dim = 1 - self.batch_dim

    def one_step(self, xt, ht, ct):
        xt = torch.cat([xt, ht], dim=self.batch_dim)
        inters = self.W_all(xt).chunk(4, dim=1)
        it = torch.sigmoid(inters[0])
        ft = torch.sigmoid(inters[1])
        gt = torch.tanh(inters[2])
        ot = torch.sigmoid(inters[3])
        ct = ft * ct + it * gt
        ht = ot * torch.tanh(ct)  # [batch_size, emb_dim]
        return ht, ct

    def mogrify(self, xt, ht):
        return xt, ht

    def forward(self, x):  # [seq_length, batch_size, emb_dim]
        output = []
        ht = torch.zeros(x.shape[self.batch_dim], self.hid_dim, device=x.device)
        ct = torch.zeros(x.shape[self.batch_dim], self.hid_dim, device=x.device)
        for t in range(x.shape[self.seq_dim]):
            xt = x.select(self.seq_dim, t)
            xt, ht = self.mogrify(xt, ht)
            ht, ct = self.one_step(xt, ht, ct)
            output.append(ht.unsqueeze(dim=self.seq_dim))
        return torch.cat(output, dim=self.seq_dim), (ht, ct)


class MogLSTM(NaiveLSTM):
    def __init__(self, emb_dim, hid_dim, num_layers, mog_iters):
        super().__init__(emb_dim, hid_dim, num_layers)
        self.Q = nn.Linear(hid_dim, emb_dim, bias=False)
        self.R = nn.Linear(emb_dim, hid_dim, bias=False)
        self.mog_iters = mog_iters

    def mogrify(self, xt, ht):
        for i in range(1, self.mog_iters + 1):
            if i % 2:  # odd
                xt = 2 * torch.sigmoid(self.Q(ht)) * xt
            else:  # even
                ht = 2 * torch.sigmoid(self.R(xt)) * ht
        return xt, ht


class LSTMLM(nn.Module):
    def __init__(
        self, backbone_name, emb_dim, hid_dim, vocab_size, mog_iters, num_layers=1
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        if backbone_name == "naive_lstm":
            self.backbone = NaiveLSTM(emb_dim, hid_dim, num_layers)
        elif backbone_name == "builtin_lstm":
            self.backbone = nn.LSTM(emb_dim, hid_dim, num_layers)
        elif backbone_name == "mog_lstm":
            self.backbone = MogLSTM(emb_dim, hid_dim, num_layers, mog_iters)
        else:
            raise NotImplementedError("The backbone you ask hasn't been implemented.")
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, idxs):  # [seq_length, batch_size]
        x = self.word_embedding(idxs)
        x, _ = self.backbone(x)  # [seq_length, batch_size, emb_dim]
        return self.fc(x)


def test_lstm():
    backbone_name = "mog_lstm"
    emb_dim = 32
    hid_dim = 16
    batch_size = 5
    vocab_size = 100
    mog_iters = 5
    seq_length = 10
    model = LSTMLM(backbone_name, emb_dim, hid_dim, vocab_size, mog_iters)
    input_idxs = torch.randint(vocab_size, (seq_length, batch_size))
    logits = model(input_idxs)
    rprint(logits.shape)  # [10, 5, 100]


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
    test_lstm()
