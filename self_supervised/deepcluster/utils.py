from typing import List
import numpy as np
from scipy import sparse as ss
import torch
from torch import nn
from torch import distributed as dist

rank = 0
world_size = 2
feat_dim = 128
crops_for_assign = [0, 1]


@torch.no_grad()
def cluster_memory(
    model, local_memory_index, local_memory_embeddings, size_dataset, num_kmeans_iters
):
    j = 0
    num_prototypes = 10
    assignments = -100 * torch.ones(len(num_prototypes), size_dataset)
    for i, K in enumerate(num_prototypes):
        # distributed k-means

        # init centroids with elements from memory bank of rank 0
        centroids = torch.empty(K, feat_dim).cuda(non_blocking=True)
        if rank == 0:
            random_idx = torch.randperm(len(local_memory_embeddings[j]))[:K]
            assert len(random_idx) >= K
            centroids = local_memory_embeddings[j][random_idx]
        dist.broadcast(centroids, 0)

        for n_iter in range(num_kmeans_iters + 1):
            # E
            dot_products = local_memory_embeddings[j] @ centroids.t()
            _, local_assignments = dot_products.max(dim=1)
            if n_iter == num_kmeans_iters:
                break

            # M
            where_helper = get_indices_sparse(local_assignments.cpu().numpy())
            counts = torch.zeros(K).cuda(non_blocking=True).int()
            emb_sums = torch.zeros(K, feat_dim).cuda(non_blocking=True)
            for k, idxs in enumerate(where_helper):
                if len(idxs) > 0:
                    emb_sums[k] = local_memory_embeddings[j][idxs].sum(dim=0)
                    counts[k] = len(idxs)
            dist.all_reduce(emb_sums)
            dist.all_reduce(counts)
            mask = counts > 0
            centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)
            centroids = nn.functional.normalize(centroids, dim=1, p=2)
        model.module.prototypes[i].weight.copy_(centroids)

        # gather the assignments
        assignments_all = torch.empty(
            world_size,
            local_assignments.shape[0],
            dtype=local_assignments.dtype,
            device=local_assignments.device,
        )
        assignments_all = list(assignments_all.unbind(0))
        dist_process = dist.all_gather(
            assignments_all, local_assignments, async_op=True
        )
        dist_process.wait()
        assignments_all = torch.cat(assignments_all).cpu()

        # gather the indexes
        indexes_all = torch.empty(
            world_size,
            local_memory_index.shape[0],
            dtype=local_memory_index.dtype,
            device=local_memory_index.device,
        )
        indexes_all = list(indexes_all.unbind(0))
        dist_process = dist.all_gather(indexes_all, local_memory_index, async_op=True)
        dist_process.wait()
        indexes_all = torch.cat(indexes_all).cpu()

        # log assignments
        assignments[i][indexes_all] = assignments_all

        # next memory bank to use
        j = (j + 1) % len(crops_for_assign)
    return assignments


def get_indices_sparse(data, num_classes) -> List[np.ndarray]:
    """This is a smart function.
    Say you have 50 smaples that fall into 10 classes. Then the return value will be a
    list with 10 elements. The i-th element of this list is an orderd 1d array of
    indices of samples that belong to the i-th class.
    Args:
        data: np.ndarray[int]
            shape: [sample_number, ]
    Return:
        indices: list[tuple[np.ndarray]]
            Length equals the number of classes.
    Examples:
        >>> get_indices_sparse(np.array([1, 0, 2, 2, 0]))
        [array([1, 4]), array([0]), array([2, 3])]
    """
    cols = np.arange(data.size)
    M = ss.csr_matrix((cols, (data.ravel(), cols)), shape=(num_classes, data.size))
    return [np.unravel_index(row.data, data.shape)[0] for row in M]
