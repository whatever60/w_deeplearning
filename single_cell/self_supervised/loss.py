import math
import torch
from torch import nn
from torch.autograd import Function


class NCECriterion(nn.Module):
    eps = 1e-7

    def __init__(self, nce_k, nce_t):
        super().__init__()
        self.nce_k = nce_k
        self.nce_t = nce_t
        # self.multinomial = AliasMethod(torch.ones(num_data))

    def forward(self, x, targets, memory):
        x = self.nce_average(x, targets, memory)
        memory_length = memory.shape[0]
        # `targets` is here to provide consistent API with nn.CrossEntropyLoss
        K = x.shape[1] - 1  # number of negative samples
        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
        lnPmt = x[:, 0] / (x[:, 0] + K / memory_length + self.eps)
        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        lnPon = (K / memory_length) / (x[:, 1:] + K / memory_length + self.eps)
        # equation 6 in ref. A
        lnPmtsum = lnPmt.log().mean()
        lnPonsum = lnPon.log().sum(dim=1).mean()
        loss = -(lnPmtsum + lnPonsum)
        # loss = -(lnPmtsum + lnPonsum) / batchSize
        return loss

    def nce_average(self, repres, pos_indices, memory):
        # repres: [batch, out_dim]
        memory_length = memory.shape[0]
        indices = torch.multinomial(
            torch.ones(memory_length, device=memory.device),
            repres.shape[0] * self.nce_k,
            replacement=True
        ).view(repres.shape[0], self.nce_k)
        indices[:, 0] = pos_indices
        # [batch, nce_k, out_dim]
        weight = memory[indices].view(repres.shape[0], self.nce_k, -1)
        # similarities [batch, nce_k]
        sims = (torch.einsum("bki, bi -> bk", weight, repres) / self.nce_t).exp()
        sims = (
            sims / sims.sum(dim=1, keepdim=True) * (self.nce_k / memory.shape[0])
        )  # each sample add up to K / N
        return sims


class AliasMethod(nn.Module):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        super().__init__()
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def to(self, device):
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)

    def draw(self, N):
        """
        Draw N samples from multinomial
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj


class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super().__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer("params", torch.tensor([T, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory", torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, memory):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return self.criterion(out, y)


class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T)  # batchSize * N

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, grad_output):
        x, memory, y, params = self.saved_tensors
        batchSize = grad_output.size(0)
        T = params[0].item()
        momentum = params[1].item()

        # add temperature
        grad_output.data.div_(T)

        # gradient of linear
        grad_input = torch.mm(grad_output.data, memory)
        grad_input.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None


def test():
    criterion = NCECriterion(40000)
    preds = torch.rand(10, 400)
    loss = criterion(preds, 0)
    print(loss)


if __name__ == "__main__":
    from rich import print

    test()
