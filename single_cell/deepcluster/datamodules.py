from typing import List
import numpy as np
from scipy import sparse as ss
import anndata
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class MultiscaleDropout:
    def __init__(
        self,
        num_crops: List[int],
        idx_crops: List[np.ndarray],
        min_dropout_p: float,
        max_dropout_p: float,
        p_noise: float,
        min_noise_scale: float,
        max_noise_scale: float,
    ) -> None:
        assert len(num_crops) == len(idx_crops)
        to_tensor = lambda x: torch.from_numpy(x)
        trans = []
        for i in range(len(num_crops)):
            # indexing = lambda x: x[idx_crops[i]]
            trans.extend(
                [
                    Compose(
                        [
                            Indexing(idx_crops[i]),
                            RandomGaussianBlur(p_noise, min_noise_scale, max_noise_scale),
                            Dropout(min_dropout_p, max_dropout_p),
                            to_tensor,
                        ]
                    )
                ]
                * num_crops[i]
            )
        self.trans = trans

    def __call__(self, x: np.ndarray) -> List[np.ndarray]:
        return list(map(lambda trans: trans(x), self.trans))


class Indexing:
    def __init__(self, idx) -> None:
        self.idx = idx
    
    def __call__(self, x: np.ndarray):
        return x[self.idx]


class Compose:
    def __init__(self, transforms: List) -> None:
        self.transforms = transforms
    
    def __call__(self, x: np.ndarray):
        for transform in self.transforms:
            x = transform(x)
        return x


class Dropout:
    def __init__(self, min_dropout_p: float, max_dropout_p: float) -> None:
        # 0, 0.1
        assert 0 <= min_dropout_p <= max_dropout_p < 1
        self.min_dropout_p = min_dropout_p
        self.max_dropout_p = max_dropout_p

    def __call__(self, x: np.ndarray):
        assert len(x.shape) == 1
        dropout_p = np.random.uniform(self.min_dropout_p, self.max_dropout_p)
        mask = np.random.binomial(1, dropout_p, x.shape).astype(bool)  # 1 means dropout
        x[mask] = 0
        return x


class RandomGaussianBlur:
    def __init__(
        self, p, min_scale, max_scale, clip=False, min_clip=0, max_clip=1
    ) -> None:
        # 0.05, 0.15
        assert 0 < min_scale <= max_scale
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale
        if clip:
            self.clipping = lambda x: x.clip(min=min_clip, max=max_clip)
        else:
            self.clipping = lambda x: x

    def __call__(self, x: np.ndarray):
        if np.random.rand() <= self.p:
            scale = np.random.uniform(self.min_scale, self.max_scale)
            return self.clipping(x + np.random.normal(scale=scale))
        else:
            return x


class scRNADataset(Dataset):
    def __init__(self, data_path: str, label_path: str, transform):
        super().__init__()
        self.data = anndata.read_h5ad(data_path).X
        self.labels = np.loadtxt(label_path, dtype=int)
        assert self.data.shape[0] == len(self.labels)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def _sparse_to_array(sp_matrix: ss.spmatrix):
        """
        Convert a scipy sparse matrix with only one line to a 1d numpy array

        Args:
            sp_matrix: scipy.sparse.spmatrix. Must be of shape [1, x].

        Returns:
            np.ndarray: 1d numpy array.
        """
        return sp_matrix.A.flatten()

    def __getitem__(self, index):
        return (
            self.transform(self._sparse_to_array(self.data[index])),
            self.labels[index],
            index,
        )


class scRNAMultiscaleDataModule(pl.LightningDataModule):
    name = "Tabula_Muris"
    num_classes = 108
    dim = 22804
    length_train = 118926
    length_val = 0

    def __init__(self, data_dir, batch_size, transform_train):
        super().__init__()
        self.train_data_dir = f"{data_dir}/data.h5ad"
        self.train_label_dir = f"{data_dir}/label.txt"
        self.batch_size = batch_size
        self.transform_train = transform_train

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = scRNADataset(
                self.train_data_dir, self.train_label_dir, self.transform_train
            )
        elif stage == "test":
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
        )

    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         pin_memory=True,
    #         num_workers=6,
    #     )


class SNAREDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.data = h5py.File(data_path)["data"]
        print(self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index].A.flatten()), -1, index


class SNAREDataModule(pl.LightningDataModule):
    name = "SNARE_seq"
    dims = (13183,)
    length_train = 10309

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir + "train/data.h5ad"
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = SNAREDataset(self.data_dir)
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


def test_datamodule():
    from rich import print as rprint
    from rich.traceback import install
    install()
    pl.seed_everything(42)

    num_crops = [2, 3]
    dim = 10_000
    idx_crops = [
        np.sort(np.random.choice(dim, dim, replace=False)),
        np.sort(np.random.choice(dim, dim // 2, replace=False)),
    ]

    transforms = MultiscaleDropout(
        num_crops,
        idx_crops,
        0.0,
        0.1,
        0.5,
        0.05,
        0.15,
    )

    data_dir = "./data"
    batch_size = 3
    module = scRNAMultiscaleDataModule(
        data_dir, batch_size=batch_size, transform_train=transforms
    )
    module.setup(stage="fit")
    loader = module.train_dataloader()
    batch = next(iter(loader))
    rprint(len(batch))  # batch_size
    rprint(len(batch[0]))  # sum(num_crops)
    rprint([i.shape for i in batch[0]])
    rprint(batch[1], batch[2])


if __name__ == "__main__":
    test_datamodule()
