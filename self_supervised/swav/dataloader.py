"""
Adapted from:
    - https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/datamodules/stl10_datamodule.py
    - https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/datamodules/cifar10_datamodule.py
    - https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/datasets/concat_dataset.py
    - https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/transforms.py
"""
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10, STL10
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl


class EuroSATSSLDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.img_list = os.listdir(data_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        return self.transform(Image.open(os.path.join(self.data_dir, self.img_list[index]))), -1


class EuroSATDataModule(pl.LightningDataModule):
    name = 'eurosat'
    num_classes = 10
    size = (64, 64)
    dims = (3, 64, 64)
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    length_unlabeled = 21350
    length_labeled = 250
    length_test = 5400

    def __init__(
        self,
        data_dir_train,
        data_dir_val,
        batch_size,
        transform_train,
        transform_val,
    ):
        super().__init__()
        self.data_dir_train = data_dir_train
        self.data_dir_val = data_dir_val
        self.batch_size = batch_size
        self.transform_train = transform_train
        self.transform_val = transform_val
        
    def setup(self, stage=None):
        if stage == 'fit':
            self.dataset_train = EuroSATSSLDataset(self.data_dir_train, self.transform_train)
            self.dataset_val = ImageFolder(self.data_dir_val)
        if stage == 'test':
            raise NotImplementedError
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        ) 

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(dataset[i % len(dataset)] for dataset in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)


class STL10DataModule(pl.LightningDataModule):  # pragma: no cover
    """
    .. figure:: https://samyzaf.com/ML/cifar10/cifar1.jpg
        :width: 400
        :alt: STL-10
    Specs:
        - 10 classes (1 per type)
        - Each image is (3 x 96 x 96)
    Standard STL-10, train, val, test splits and transforms.
    STL-10 has support for doing validation splits on the labeled or unlabeled splits
    Transforms::
        mnist_transforms = T.Compose([
            T.ToTensor(),
            transforms.Normalize(
                mean=(0.43, 0.42, 0.39),
                std=(0.27, 0.26, 0.27)
            )
        ])
    Example::
        from pl_bolts.datamodules import STL10DataModule
        dm = STL10DataModule(PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    name = "stl10"
    num_classes = 10
    size = (96, 96)
    dims = (3, 96, 96)
    means = (0.43, 0.42, 0.39)
    stds = (0.27, 0.26, 0.27)
    length_unlabeled = 100_000
    length_train = 5_000
    length_test = 8_000

    def __init__(
        self,
        data_dir,
        # val_split: float,  # 0.05 for mode = 'unlabeled', 0.1 for mode = 'labeled', 0.1 for mixed
        batch_size: int,
        transform_train,
        transform_val,
        transform_test=None,
        mode='mixed',
        # seed: int = 42,
        # shuffle: bool = False,
        # pin_memory: bool = False,
        # drop_last: bool = False,
        # *args: Any,
        # **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to save/load the data
            unlabeled_train_val_split: how many images from the unlabeled training split to use for validation
            train_val_split: how many images from the labeled training split to use for validation
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__()

        self.data_dir = data_dir
        # self.val_split = val_split
        self.batch_size = batch_size
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_test = transform_test
        self.mode = mode

    def prepare_data(self) -> None:
        """
        Downloads the unlabeled, train and test split.
        """
        STL10(self.data_dir, split="unlabeled", download=True)
        STL10(self.data_dir, split="train", download=True)
        STL10(self.data_dir, split="test", download=True)

    def setup(self, stage=None):
        if stage == "fit":
            if self.mode == "unlabeled":
                val_split = 0.05
                dataset_train, _ = random_split(
                    STL10(self.data_dir, split="unlabeled", download=False, transform=self.transform_train),
                    [
                        int(self.length_unlabeled * (1 - val_split)),
                        int(self.length_unlabeled * val_split),
                    ],
                )
                _, dataset_val = random_split(
                    STL10(self.data_dir, split="unlabeled", download=False, transform=self.transform_val),
                    [
                        int(self.length_unlabeled * (1 - val_split)),
                        int(self.length_unlabeled * val_split),
                    ],
                )
                dataset_train.transform = self.transform_train
                dataset_val.transform = self.transform_val
            elif self.mode == "labeled":
                val_split = 0.1
                dataset_train, _ = random_split(
                    STL10(self.data_dir, split="train", download=False),
                    [
                        int(self.length_train * (1 - val_split)),
                        int(self.length_train * val_split),
                    ],
                )
                _, dataset_val = random_split(
                    STL10(self.data_dir, split="unlabeled", download=False, transform=self.transform_val),
                    [
                        int(self.length_unlabeled * (1 - val_split)),
                        int(self.length_unlabeled * val_split),
                    ],
                )
            else:
                val_split = 0.1
                dataset_train0, _ = random_split(
                    STL10(self.data_dir, split="unlabeled", download=False, transform=self.transform_train),
                    [
                        int(self.length_unlabeled * (1 - val_split)),
                        int(self.length_unlabeled * val_split),
                    ],
                )
                _, dataset_val0 = random_split(
                    STL10(self.data_dir, split="unlabeled", download=False, transform=self.transform_val),
                    [
                        int(self.length_unlabeled * (1 - val_split)),
                        int(self.length_unlabeled * val_split),
                    ],
                )

                dataset_train1, _ = random_split(
                    STL10(self.data_dir, split="train", download=False, transform=self.transform_train),
                    [
                        int(self.length_train * (1 - val_split)),
                        int(self.length_train * val_split),
                    ],
                )
                _, dataset_val1 = random_split(
                    STL10(self.data_dir, split="train", download=False, transform=self.transform_val),
                    [
                        int(self.length_train * (1 - val_split)),
                        int(self.length_train * val_split),
                    ],
                )
                dataset_train = ConcatDataset(dataset_train0, dataset_train1)
                dataset_val = ConcatDataset(dataset_val0, dataset_val1)
            self.dataset_train = dataset_train
            self.dataset_val = dataset_val
        if stage == 'test':
            dataset_test = STL10(self.data_dir, split="test", download=False),
            dataset_test.transform = self.transform_test
            self.dataset_test = dataset_test

    
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
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    # def _default_transforms(self) -> Callable:
    #     data_transforms = T.Compose([T.ToTensor(), stl10_normalization()])
    #     return data_transforms


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
    dims = (3, 32, 32)
    means = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    stds = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    length_train = 50_000
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
        if stage == 'fit':
            val_split = 0.2
            self.dataset_train, _ = random_split(
                CIFAR10(self.data_dir, train=True, download=False, transform=self.transform_train),
                [
                    int(self.length_train * (1 - val_split)),
                    int(self.length_train * val_split),
                ],
            )
            _, self.dataset_val = random_split(
                CIFAR10(self.data_dir, train=True, download=False, transform=self.transform_val),
                [
                    int(self.length_train * (1 - val_split)),
                    int(self.length_train * val_split),
                ],
            )
        if stage == 'test':
            self.dataset_test = CIFAR10(self.data_dir, train=False, download=False, transform=self.transform_test),

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


class SwAVTransform:
    def __init__(
        self,
        means,
        stds,
        size_crops,
        num_crops,
        min_scale_crops,
        max_scale_crops,
        use_gaussian_blur,
        jitter_strength,
        train,
    ) -> None:
        global_size = size_crops[0]
        transforms = []
        color_jitter = T.RandomApply(
            [T.ColorJitter(
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.2 * jitter_strength,
            )],
            p=0.8,
        )
        color_transform = [color_jitter, T.RandomGrayscale(p=0.2)]

        if use_gaussian_blur:
            kernel_size = 0.1 * global_size // 2 * 2 + 1
            gaussian_blur = T.RandomApply([T.GaussianBlur(kernel_size, (0.1, 2))], p=0.5)
            color_transform.append(gaussian_blur)

        for size, num, min_, max_ in zip(
            size_crops, num_crops, min_scale_crops, max_scale_crops
        ):
            for _ in range(num):
                # Add this same transform `num` times.
                transforms.append(
                    T.Compose(
                        [T.ToTensor(),
                        T.RandomResizedCrop(size, scale=(min_, max_)),
                        T.RandomHorizontalFlip(p=0.5),
                        *color_transform,
                        T.Normalize(means, stds)]
                    )
                )
        if train:  # no color transform
            transforms.append(T.Compose(
                [
                    T.ToTensor(),
                    T.RandomResizedCrop(global_size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(means, stds),
                    
                ]
            ))
        else:
            transforms.append(T.Compose(
                [
                    T.ToTensor(),
                    T.Resize(int(global_size + 0.1 * global_size)),
                    T.CenterCrop(global_size),
                    T.Normalize(means, stds),
                ]
            ))

        self.transforms = transforms

    def __call__(self, image):
        return list(map(lambda t: t(image), self.transforms))


class SwAVFinetuneTransform:
    """
    Finetune doesn't need that much augmentation.
    """
    def __init__(
        self,
        means,
        stds,
        size,
        jitter_strength,
        train,
    ) -> None:
        self.color_jitter = T.ColorJitter(
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength,
        )

        if train:
            self.transform = T.Compose([
                T.ToTensor(),
                T.RandomResizedCrop(size=size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([self.color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.Normalize(means, stds)
            ])
        else:
            self.transform = [
                T.ToTensor(),
                T.Resize(int(size + 0.1 * size)),
                T.CenterCrop(size),
                T.Normalize(means, stds)
            ]

    def __call__(self, sample):
        return self.transform(sample)


def get_swav_transform(
        means,
        stds,
        size_crops,
        num_crops,
        min_scale_crops,
        max_scale_crops,
        use_gaussian_blur,
        jitter_strength,
):
    transforms = [
        SwAVTransform(
            means,
            stds,
            size_crops,
            num_crops,
            min_scale_crops,
            max_scale_crops,
            use_gaussian_blur,
            jitter_strength,
            train=True
        ),
        SwAVTransform(
            means,
            stds,
            size_crops,
            num_crops,
            min_scale_crops,
            max_scale_crops,
            use_gaussian_blur,
            jitter_strength,
            train=False
        )
    ]
    return transforms
