from abc import abstractmethod
from typing import Any
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
import pytorch_lightning as pl



@dataclass
class VisionModule(pl.LightningDataModule):
    '''
    Args:
        data_dir:
        val_split: Percent (float) or number (int) of samples to use for the validation split
        num_workers:
        normalize:
        batch_size:
        shuffle: If true shuffles the train data every epoch
        pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                    returning them
        drop_last: If true drops the last incomplete batch
    '''
    dataset_cls: Dataset
    dims: tuple
    data_dir: str = './'
    val_split: float = 1 / 6
    num_workers: int = 8
    batch_size: int = 32
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False
    img_size: tuple = None
    train_transforms: Any = T.ToTensor()
    val_transforms: Any = T.ToTensor()
    test_transforms: Any = T.ToTensor()

    def __post_init__(self):
        dims = self.dims  # a little bit tricky here.
        super().__init__()
        self.dims = dims

    def prepare_data(self) -> None:
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage = None) -> None:
        if stage == 'fit' or stage is None:
            dataset = self.dataset_cls(self.data_dir, train=True)
            
            self.dataset_train, self.dataset_val = random_split(
                dataset, 
                [len(dataset) - int(len(dataset) * self.val_split), int(len(dataset) * self.val_split)],
            )

            self.dataset_train.dataset.transform = self.train_transforms
            self.dataset_val.dataset.transform = self.val_transforms

        if stage == 'test' or stage is None:
            self.dataset_test = self.dataset_cls(self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)
    
    def val_dataloader(self) -> DataLoader:
        """ The val dataloader """
        return self._data_loader(self.dataset_val)
    
    def test_dataloader(self) -> DataLoader:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
