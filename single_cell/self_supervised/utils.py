import math

import torch
from torch.nn.functional import one_hot


@torch.no_grad()
def knn(
    num_classes,
    memory,
    input_data,
    memory_labels,
    n_neighbors,
    nce_t,
    relax=3,
):
    # empirical
    num_subknn = math.ceil(input_data.numel() * n_neighbors / (10_000 * 128 * 200))
    results = []
    for data in input_data.chunk(num_subknn, dim=0):
        results.append(
            _knn(num_classes, memory, data, memory_labels, n_neighbors, nce_t, relax)
        )
    return torch.cat(results, dim=0)


@torch.no_grad()
def _knn(
    num_classes,
    memory,
    input_data,
    memory_labels,
    n_neighbors,
    nce_t,
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
    return preds


def test_knn():
    num_classes = 100
    memory_length = 100_000
    input_length = 20_000
    feature_dim = 200
    n_neighbors = 300
    relax = 4
    memory = torch.randn(memory_length, feature_dim)
    input_data = torch.randn(input_length, feature_dim)
    memory_labels = torch.randint(num_classes, (memory_length,))
    input_labels = torch.randint(num_classes, (input_length,))
    preds = knn(
        num_classes, memory, input_data, memory_labels, n_neighbors, relax=relax
    )

    result = preds == input_labels.unsqueeze(dim=-1)
    top_1_acc = result[:, 0].float().mean()
    top_relax_acc = result.any(dim=1).float().mean()
    rprint(preds.shape)  # [input_length, relax]
    rprint(top_1_acc, top_relax_acc)  # about (1 / num_classes, relax / num_classes)


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
    test_knn()
