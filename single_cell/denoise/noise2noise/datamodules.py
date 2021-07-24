import numpy as np
from scipy import sparse as ss
import anndata
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class scRNADataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data = anndata.read_h5ad(data_dir).X

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index].A.flatten())


class RNANoise2NoiseFixedDataset(Dataset):
    def __init__(self, data_path1, data_path2) -> None:
        super().__init__()
        self.X1 = anndata.read_h5ad(data_path1).X
        self.X2 = anndata.read_h5ad(data_path2).X
        assert self.X1.shape == self.X2.shape
        self.length, self.dim = self.X1.shape[0], self.X1.shape[1]
        self.density = ((self.X1 != 0).sum() + (self.X2 != 0).sum()) / (
            self.length * self.dim * 2
        )
        print(self.density)
        self.transform = Compose([Sparse2Array(), ToTensor()])

    def __len__(self):
        return self.length * 2

    def __getitem__(self, index):
        if index < self.length:
            input_ = self.X1[index]
            target = self.X2[index]
        else:
            input_ = self.X1[index - self.length]
            target = self.X2[index - self.length]
        input_ = self.transform(input_)
        target = self.transform(target)
        # 1 means mask
        return input_, target


class RNANoise2NoiseDataset(Dataset):
    def __init__(
        self,
        data_path,
        min_dropout_p,
        max_dropout_p,
        p_noise,
        min_noise_scale,
        max_noise_scale,
    ) -> None:
        super().__init__()
        self.X = anndata.read_h5ad(data_path).X
        self.length, self.dim = self.X.shape[0], self.X.shape[1]
        self.density = (self.X != 0).mean()
        print(self.density)
        self.transform = Compose(
            [
                Sparse2Array(),
                # RandomGaussianBlur(
                #     p_noise, min_noise_scale, max_noise_scale, clip=True
                # ),
                Dropout(min_dropout_p, max_dropout_p),
                ToTensor(),
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.X[index]
        input_ = self.transform(data)
        target = self.transform(data)
        return input_, target


class NaiveDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.X = anndata.read_h5ad(data_path).X
        self.transform = Compose([Sparse2Array(), ToTensor()])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.transform(self.X[index])


class Sparse2Array:
    """
    Convert a scipy sparse matrix with only one line to a 1d numpy array

    Args:
        sp_matrix: scipy.sparse.spmatrix. Must be of shape [1, x].

    Returns:
        np.ndarray: 1d numpy array.
    """

    def __call__(self, sp_matrix: ss.spmatrix):
        return sp_matrix.A[0]


class ToTensor:
    def __call__(self, x: np.ndarray):
        return torch.from_numpy(x)


class Compose:
    def __init__(self, transforms) -> None:
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
        # 0.02, 0.05
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
            noise = np.random.normal(scale=scale, size=x.shape[0]).astype("float32")
            return self.clipping(x + noise)
        else:
            return x


class SNARESeqDataModule(pl.LightningDataModule):
    name = "SNARE_seq"
    dims = (9767,)
    length_train = 5081 - 800
    length_val = 800

    def __init__(
        self,
        data_path_train,
        data_path_val,
        batch_size,
        min_dropout_p,
        max_dropout_p,
        p_noise,
        min_noise_scale,
        max_noise_scale,
    ):
        super().__init__()
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val
        self.batch_size = batch_size
        self.min_dropout_p = min_dropout_p
        self.max_dropout_p = max_dropout_p
        self.p_noise = p_noise
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = RNANoise2NoiseDataset(
                self.data_path_train,
                self.min_dropout_p,
                self.max_dropout_p,
                p_noise=self.p_noise,
                min_noise_scale=self.min_noise_scale,
                max_noise_scale=self.max_noise_scale,
            )
            self.val_dataset = RNANoise2NoiseDataset(
                self.data_path_val,
                self.min_dropout_p,
                self.max_dropout_p,
                p_noise=self.p_noise,
                min_noise_scale=self.min_noise_scale,
                max_noise_scale=self.max_noise_scale,
            )
        elif stage == "test":
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=8,
        )


class SNARESeqFixedDataModule(pl.LightningDataModule):
    name = "SNARE_seq"
    dims = (9767,)
    length_train = (5081 - 800) * 2
    length_val = 800 * 2

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.X1_train_dir = f"{data_dir}/train/data1.h5ad"
        self.X2_train_dir = f"{data_dir}/train/data2.h5ad"
        self.X1_val_dir = f"{data_dir}/val/data1.h5ad"
        self.X2_val_dir = f"{data_dir}/val/data2.h5ad"
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = RNANoise2NoiseFixedDataset(
                self.X1_train_dir, self.X2_train_dir
            )
            self.val_dataset = RNANoise2NoiseFixedDataset(self.X1_val_dir, self.X2_val_dir)
        elif stage == "test":
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=6,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=6,
        )
