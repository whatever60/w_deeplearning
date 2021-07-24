import numpy as np
import anndata
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def gen_dataset(data_dir, binomial_p=0.85, cache_dir="./data"):
    # It takes time to import this, so do lazy import.
    from pipeline import read_mtx, qc, normalize

    X = read_mtx(data_dir)
    X, _, _ = qc(X, gene_min_cells=50, gene_min_counts=100, logic="mine")
    X1, X2 = pseudo_replicate(X, binomial_p)

    X1, X2 = normalize(X1, apply_qc=False), normalize(X2, apply_qc=False)
    X1_train, X1_val, X2_train, X2_val = train_test_split(X1, X2, test_size=0.2)
    anndata.AnnData(X1_train).write_h5ad(f"{cache_dir}/{binomial_p}/x1_train.h5ad")
    anndata.AnnData(X1_val).write_h5ad(f"{cache_dir}/{binomial_p}/x1_val.h5ad")
    anndata.AnnData(X2_train).write_h5ad(f"{cache_dir}/{binomial_p}/x2_train.h5ad")
    anndata.AnnData(X2_val).write_h5ad(f"{cache_dir}/{binomial_p}/x2_val.h5ad")


def pseudo_replicate(X_sparse, p):
    replicate_data = np.array([np.random.binomial(i, p, 2) for i in X_sparse.data])
    replicate1 = X_sparse.copy()
    replicate1.data = replicate_data[:, 0]
    replicate2 = X_sparse.copy()
    replicate2.data = replicate_data[:, 1]
    return replicate1, replicate2


class scRNADataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data = anndata.read_h5ad(data_dir).X

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index].A.flatten())


class RNADenoisingDataset(Dataset):
    def __init__(self, data_dir1, data_dir2, mask_p) -> None:
        super().__init__()
        self.X1 = anndata.read_h5ad(data_dir1).X
        self.X2 = anndata.read_h5ad(data_dir2).X
        assert self.X1.shape == self.X2.shape
        self.length, self.dim = self.X1.shape[0], self.X1.shape[1]
        self.mask_p = mask_p
        self.density = ((self.X1 != 0).sum() + (self.X2 != 0).sum()) / (
            self.length * self.dim * 2
        )
        print(self.density)

    def __len__(self):
        return self.length * 2

    def __getitem__(self, index):
        if index < self.length:
            input_ = self.X1[index]
            target = self.X2[index]
        else:
            input_ = self.X1[index - self.length]
            target = self.X2[index - self.length]
        input_ = torch.from_numpy(input_.A.flatten())
        target = torch.from_numpy(target.A.flatten())
        # 1 means mask
        mask = torch.from_numpy(np.random.binomial(1, self.mask_p, self.dim)).bool()
        input_ = input_.masked_fill(mask, 0)
        return input_, target, mask


class RNADenoisingDataModule(pl.LightningDataModule):
    name = "SNARE_seq"
    dims = (13183,)
    length_train = 8247 * 2
    length_val = 2062 * 2

    def __init__(self, cache_dir, batch_size, binomial_p, mask_p):
        super().__init__()
        self.X1_train_dir = f"{cache_dir}/{binomial_p}/x1_train.h5ad"
        self.X1_val_dir = f"{cache_dir}/{binomial_p}/x1_val.h5ad"
        self.X2_train_dir = f"{cache_dir}/{binomial_p}/x2_train.h5ad"
        self.X2_val_dir = f"{cache_dir}/{binomial_p}/x2_val.h5ad"
        self.batch_size = batch_size
        self.mask_p = mask_p

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = RNADenoisingDataset(
                self.X1_train_dir, self.X2_train_dir, self.mask_p
            )
            self.val_dataset = RNADenoisingDataset(
                self.X1_val_dir, self.X2_val_dir, self.mask_p
            )
        elif stage == "test":
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=6,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=6,
        )


if __name__ == "__main__":
    from rich.traceback import install

    install()
    DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/babel/snareseq_GSE126074/GSE126074_AdBrainCortex_SNAREseq_cDNA.counts.mtx.gz"
    gen_dataset(DATA_DIR, binomial_p=0.85)
