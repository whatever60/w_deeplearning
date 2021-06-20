import random
import math
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class WikiText2Dataset(Dataset):
    def __init__(self, data_list) -> None:
        """
        Args:
            data_list: List. A list of raw text. Each element is an independent entry, so the order of this list doesn't matter.
        """
        super().__init__()
        self.data_list = data_list

    def form_dataset(self, batch_size, bptt, shuffle=True):
        if shuffle:
            data_list = self.data_list[:]
            random.shuffle(data_list)
        else:
            data_list = self.data_list
        data = [id_ for ids in data_list for id_ in ids]
        # data = self.data_list
        num_steps = len(data) // batch_size

        self.batch_size = batch_size
        self.bptt = bptt
        self.num_steps = num_steps
        self._sample_per_line = math.ceil((self.num_steps - 1) / self.bptt)
        self._len = self._sample_per_line * batch_size
        self.data = (
            torch.tensor(data[: num_steps * batch_size])
            .view(batch_size, num_steps)
            .t()
            .contiguous()
        )
        return self

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        start_idx_row = index // self.batch_size * self.bptt
        end_idx_row = min(start_idx_row + self.bptt, self.num_steps - 1)
        idx_col = index % self.batch_size
        return (
            self.data[start_idx_row:end_idx_row, idx_col],
            self.data[start_idx_row + 1 : end_idx_row + 1, idx_col],
        )


def test_dataset():
    dataset = WikiText2Dataset(torch.arange(1000).reshape(10, 100).tolist())
    dataset.form_dataset(8, 4, shuffle=True)
    rprint(dataset[0])
    dataset.form_dataset(8, 4, shuffle=False)
    rprint(dataset[0])


class WikiText2DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, bptt_train, bptt_val):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.bptt_train = bptt_train
        self.bptt_val = bptt_val
        with open(data_dir, "rb") as f:
            (
                self.vocab_size,
                self.data_list_train,
                self.data_list_val,
                self.data_list_test,
            ) = pickle.load(f)

    def setup(self, stage=None):
        if stage == "fit":
            self.dataset_train = WikiText2Dataset(self.data_list_train)
            self.dataset_val = WikiText2Dataset(self.data_list_val).form_dataset(
                self.batch_size, self.bptt_val, shuffle=False
            )
        elif stage == "test":
            self.dataset_test = WikiText2Dataset(self.data_list_test).form_dataset(
                self.batch_size, self.bptt_val, shuffle=False
            )

    def train_dataloader(self):
        self.dataset_train.form_dataset(
            batch_size=self.batch_size, bptt=self.bptt_train
        )
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=2
        )


if __name__ == "__main__":
    from tokenizers import Tokenizer
    from rich import print as rprint
    from rich.traceback import install
    install()

    DATA_DIR = "./data/tokenized.pkl"
    BATCH_SIZE = 128
    BPTT_TRAIN = 64
    BPTT_VAL = 64
    datamodule = WikiText2DataModule(DATA_DIR, BATCH_SIZE, BPTT_TRAIN, BPTT_VAL)
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()
    tokenizer = Tokenizer.from_file("./tokenizer/tokenizer.json")
    a = next(iter(loader))
    rprint(a[0].shape)
    rprint(tokenizer.decode(a[0][1].tolist()))
