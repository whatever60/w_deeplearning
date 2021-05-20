import math

import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.autograd import Function


class AliasMethod:
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):

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


class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer("params", torch.tensor([T, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory", torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out



@torch.no_grad()
def knn(
    num_classes,
    memory,
    input_data,
    memory_labels,
    input_labels=None,
    n_neighbors=40,
    nce_t=0.1,
    relax=3,
):
    # empirical
    num_subknn = math.ceil(input_data.numel() * n_neighbors / (10_000 * 128 * 200))
    results = []
    for data, labels in zip(
        input_data.chunk(num_subknn, dim=0), input_labels.chunk(num_subknn)
    ):
        results.append(
            _knn(
                num_classes,
                memory,
                data,
                memory_labels,
                labels,
                n_neighbors,
                nce_t,
                relax,
            )
        )
    if input_labels is None:
        return torch.cat(results)
    else:
        top_1_acc = top_relax_acc = 0
        for result in results:
            top_1_acc += result[0] * result[2]
            top_relax_acc += result[1] * result[2]
        return top_1_acc / input_labels.shape[0], top_relax_acc / input_labels.shape[0]


@torch.no_grad()
def _knn(
    num_classes,
    memory,
    input_data,
    memory_labels,
    input_labels=None,
    n_neighbors=40,
    nce_t=0.1,
    relax=3,
):
    """
    Perform K-nearest neighbor classification.
    Args:
        num_classes:
            int
            Number of classes.
        memory:
            torch.Tensor
            shape: [memory_length, feature_dim]
            Memory bank, neighbors of `input_data`.
        input_data:
            torch.Tensor
            shape: [input_length, feature_dim]
            Data you want to perform classification on.
        memory_labels:
            torch.Tensor
            shape: [memory_length, ]
            Ground truth label of `memory`.
        input_labels:
            None | torch.Tensor
            shape: [input_length, ]
            Ground truth label of `input_data`. When set to `None`, use `input_labels` to calculate accuracy.
        n_neighbors:
            int
            Numebr of nearest neighbors, i.e. use K data points in the memory to estimate the label of `input_data`.
        nce_t:
            float.
            Temperature used in logits calculation.
        relax:
            int.
            Calculate top `relax` accuracy besides top 1 accuracy.

    Return:
        When `input_labels` is `None`, return top 1 and top `relax` accuracy.
        tuple[float, float, int]

        Otherwise, return label prediction.
        torch.Tensor
        shape: [input_data, ]
    """
    dist = input_data @ memory.T  # [input_length, memory_length]
    # both [input_length, n_neighbors]
    similarity, indices = dist.topk(n_neighbors, dim=1, largest=True, sorted=True)
    similarity = (similarity / nce_t).exp()
    # Is there better way of doing such indexing?
    candidates = memory_labels[indices.view(-1)].view(-1, n_neighbors)
    # [input_length, n_neighbors, num_classes]
    retrieval_one_hot = one_hot(candidates, num_classes=num_classes)
    # [input_length, n_neighbors, 1] * [input_length, n_neighbors, num_classes] -> [input_length, num_classes]
    # this step adds up similarity of the same class.
    logits = (similarity.unsqueeze(dim=-1) * retrieval_one_hot).sum(dim=1)
    # [input_length, relax]
    _, preds = logits.topk(relax, dim=1, largest=True, sorted=True)

    if input_labels is None:
        return preds[:, 0]
    else:
        result = preds == input_labels.unsqueeze(dim=-1)
        top_1_acc = result[:, 0].float().mean()
        top_relax_acc = result.any(dim=1).float().mean()
        return top_1_acc, top_relax_acc, input_labels.shape[0]


def draw_chord()


def test_knn():
    num_classes = 100
    memory_length = 100_000
    input_length = 20_000
    feature_dim = 200
    n_neighbors = 300
    memory = torch.randn(memory_length, feature_dim)
    input_data = torch.randn(input_length, feature_dim)
    memory_labels = torch.randint(num_classes, (memory_length, ))
    input_labels = torch.randint(num_classes, (input_length, ))
    # about (0.01, 0.03)
    rprint(knn(num_classes, memory, input_data, memory_labels, input_labels, n_neighbors))


if __name__ == '__main__':
    from rich import print as rprint
    from rich.traceback import install
    install()
    test_knn()

