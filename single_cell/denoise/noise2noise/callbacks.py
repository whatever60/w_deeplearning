import anndata
import numpy as np
from scipy.stats import pearsonr
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from datamodules import NaiveDataset


class RefPearson(pl.Callback):
    """
    Compare aggregated reconstructed poor-quality RNA-seq profile with aggregated
    fair-quality RNA-seq profile using Pearson correlation.
    """

    def __init__(
        self, data_path_pred, data_path_ref, gene_path_pred, gene_path_ref
    ) -> None:
        super().__init__()
        self.gene_list_pred = (
            pd.read_csv(gene_path_pred, header=None, squeeze=True)
            .str.lower()
            .str.strip()
            .values
        )
        self.gene_list_ref = (
            pd.read_csv(gene_path_ref, header=None, squeeze=True)
            .str.lower()
            .str.strip()
            .values
        )
        self.gene_index_pred, self.gene_index_ref, self.gene_list = _get_index(
            self.gene_list_pred, self.gene_list_ref
        )
        # print(len(self.gene_list))  # 9435
        self.agg_ref = (
            anndata.read_h5ad(data_path_ref)
            .X.mean(axis=0)
            .A.flatten()[self.gene_index_ref]
        )
        self.agg_pred = (
            anndata.read_h5ad(data_path_pred)
            .X.mean(axis=0)
            .A.flatten()[self.gene_index_pred]
        )
        self.init_pearsonr = pearsonr(self.agg_pred, self.agg_ref)[0]
        print("init_pearsonr", self.init_pearsonr)
        self.dataset = NaiveDataset(data_path_pred)

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.eval()
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=128,
            shuffle=False,
            pin_memory=False,
            num_workers=4,
        )
        agg_pred = torch.zeros([])
        num_nonzero = 0
        for data in dataloader:
            data = data.to(pl_module.device)
            pred = pl_module(data)
            # `torch.zeros([]) + torch.arange(10)` works. But `+=` doesn't work. Interesting
            agg_pred = agg_pred + pred.sum(dim=0)
            num_nonzero += (pred != 0).float().sum()

        density = num_nonzero / (len(self.dataset) * agg_pred.shape[0])
        agg_pred = (agg_pred / len(self.dataset)).cpu().numpy()[self.gene_index_pred]
        pl_module.log_dict(
            dict(global_density=density, agg_corr=pearsonr(agg_pred, self.agg_ref)[0])
        )


def _get_index(a: np.ndarray, b: np.ndarray):
    """
    Modified from:
    - https://www.followthesheep.com/?p=1366
    """
    a1 = np.argsort(a)
    b1 = np.argsort(b)

    sort_left_a = a[a1].searchsorted(b[b1], side="left")
    sort_right_a = a[a1].searchsorted(b[b1], side="right")

    sort_left_b = b[b1].searchsorted(a[a1], side="left")
    sort_right_b = b[b1].searchsorted(a[a1], side="right")

    # # which values are in b but not in a?
    # idx_b=(sort_right_a-sort_left_a == 0).nonzero()[0]
    # # which values are in b but not in a?
    # idx_a=(sort_right_b-sort_left_b == 0).nonzero()[0]

    # which values of b are also in a?
    idx_b = (sort_right_a - sort_left_a > 0).nonzero()[0]
    idx_b_1 = b1[idx_b]
    # which values of a are also in b?
    idx_a = (sort_right_b - sort_left_b > 0).nonzero()[0]
    idx_a_1 = a1[idx_a]

    common_a, common_b = a[idx_a_1], b[idx_b_1]
    common_unique_a, idx_a_2 = np.unique(common_a, return_index=True)
    common_unique_b, idx_b_2 = np.unique(common_b, return_index=True)

    assert (common_unique_a == common_unique_b).all()

    return idx_a_1[idx_a_2], idx_b_1[idx_b_2], common_unique_a


def test_get_index():
    a = np.array([0, 1, 0, 3, 2, 1, 3])
    b = np.array([1, 2, 0, 3, 0, 2])
    i1, i2, common = _get_index(a, b)
    print(a[i1], b[i2], common)


if __name__ == "__main__":
    test_get_index()
