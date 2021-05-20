import numpy as np
from scipy import sparse as ss
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class scRNADataset(Dataset):
    def __init__(self, data: ss.spmatrix, labels: np.ndarray):
        super().__init__()
        self.data = data
        self.labels = labels
        assert self.data.shape[0] == len(self.labels)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index].A.flatten()), self.labels[index], index


class scRNADataModule(pl.LightningDataModule):
    name = "Tabula_Muris"
    num_classes = 108
    dims = (22804,)
    length_train = 103_926
    length_val = 15000

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_train = ss.load_npz(data_dir + "train/data.npz")
        self.label_train = np.load(data_dir + "train/labels.npy")
        self.data_val = ss.load_npz(data_dir + "val/data.npz")
        self.label_val = np.load(data_dir + "val/labels.npy")
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = scRNADataset(self.data_train, self.label_train)
            self.val_dataset = scRNADataset(self.data_val, self.label_val)
        elif stage == "test":
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )


def test_datamodule():
    from rich import print as rprint
    from rich.traceback import install

    install()
    data_dir = "/home/tiankang/wusuowei/data/single_cell/MARS/Tabula_Muris/"
    module = scRNADataModule(data_dir, batch_size=200)
    module.setup(stage="fit")
    loader = module.train_dataloader()
    batch = next(iter(loader))
    rprint(len(batch))
    rprint(batch[0].shape)


if __name__ == "__main__":
    test_datamodule()
