import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST
import pytorch_lightning as pl


def split_dataset(dataset, val_length, transform_train, transform_val):
    mask = np.ones(len(dataset), dtype=bool)
    indices = np.arange(len(dataset))
    mask[
        np.random.choice(len(dataset), val_length)
    ] = False  # True for training data, False for validation data
    return SplitDataset(dataset, indices[mask], transform_train), SplitDataset(
        dataset, indices[~mask], transform_val
    )


class SplitDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray, transform) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), label


class CIFAR10DataModule(pl.LightningDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
        Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
        :width: 400
        :alt: CIFAR-10
    Specs:
        - 10 classes (1 per class)
        - Each image is (3 x 32 x 32)
    Standard CIFAR10, train, val, test splits and transforms
    Transforms::
        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
    Example::
        from pl_bolts.datamodules import CIFAR10DataModule
        dm = CIFAR10DataModule(PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    Or you can set your own transforms
    Example::
        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
    """

    name = "cifar10"
    num_classes = 10
    size = (32, 32)
    dims = (3, 32, 32)
    means = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    stds = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    length_train = 44_000
    length_val = 6_000
    length_test = 10_000

    def __init__(
        self,
        data_dir,
        batch_size: int,
        transform_train,
        transform_val,
        transform_test=None,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_test = transform_test

    def prepare_data(self) -> None:
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit":
            dataset = CIFAR10(
                self.data_dir,
                train=True,
                download=False,
            )
            self.dataset_train, self.dataset_val = split_dataset(
                dataset,
                self.length_val,
                transform_train=self.transform_train,
                transform_val=self.transform_val,
            )

        if stage == "test":
            self.dataset_test = (
                CIFAR10(
                    self.data_dir,
                    train=False,
                    download=False,
                    transform=self.transform_test,
                ),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Loads the test split of STL10
        Args:
            batch_size: the batch size
            transforms: the transforms
        """
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )


class MNISTDataModule(pl.LightningDataModule):
    """
    .. figure:: https://miro.medium.com/max/744/1*AO2rIhzRYzFVQlFLx9DM9A.png
        :width: 400
        :alt: MNIST

    Specs:
        - 10 classes (1 per digit)
        - Each image is (1 x 28 x 28)

    Standard MNIST, train, val, test splits and transforms

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])

    Example::

        from pl_bolts.datamodules import MNISTDataModule

        dm = MNISTDataModule('.')
        model = LitModel()

        Trainer().fit(model, datamodule=dm)
    """

    name = "mnist"
    num_classes = 10
    size = (28, 28)
    dims = (1, 28, 28)
    means = (0.5,)
    stds = (0.5,)
    length_train = 50_000
    length_val = 10_000
    length_test = 10_000

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        transform_train,
        transform_val,
        transform_test=None,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_test = transform_test

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit":
            val_split = 1 / 6
            self.dataset_train, _ = random_split(
                MNIST(
                    self.data_dir,
                    train=True,
                    download=False,
                    transform=self.transform_train,
                ),
                [self.length_train, self.length_val],
            )
            _, self.dataset_val = random_split(
                MNIST(
                    self.data_dir,
                    train=True,
                    download=False,
                    transform=self.transform_val,
                ),
                [self.length_train, self.length_val],
            )
        if stage == "test":
            self.dataset_test = (
                MNIST(
                    self.data_dir,
                    train=False,
                    download=False,
                    transform=self.transform_test,
                ),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Loads the test split of STL10
        Args:
            batch_size: the batch size
            transforms: the transforms
        """
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )
