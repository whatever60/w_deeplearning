import numpy as np
from PIL import ImageFilter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms as T
import pytorch_lightning as pl


def split_dataset(dataset, val_length, transform_train, transform_val):
    mask = np.ones(len(dataset), dtype=bool)
    indices = np.arange(len(dataset))
    # True for training data, False for validation data
    mask[np.random.choice(len(dataset), val_length)] = False
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
        image, label, index = self.dataset[self.indices[index]]
        return self.transform(image), label, index


class MultiCropDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label, index


class MultiCropDataModule(pl.LightningDataModule):
    name = "imagenette"
    num_classes = 10
    size = (224, 224)
    dims = (3, 224, 224)
    # means = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    # stds = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    means = [0.485, 0.456, 0.406]  # from Imagenet
    stds = [0.229, 0.224, 0.225]
    length_train = 9025  # 9469
    length_val = 3929  # 3925
    length_test = 0

    def __init__(
        self,
        data_dir,
        batch_size: int,
        transform_train,
        transform_val,
        transform_test=None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_test = transform_test

    def setup(self, stage=None):
        if stage == "fit":
            self.dataset_train = MultiCropDataset(
                ImageFolder(self.data_dir), self.transform_train
            )

        if stage == "test":
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )


def get_aug(
    means,
    stds,
    # ---- the following 4 params must be of the same length ----
    num_crops,
    size_crops,
    min_scale_crops,
    max_scale_crops,
):
    assert len(size_crops) == len(num_crops)
    assert len(min_scale_crops) == len(num_crops)
    assert len(max_scale_crops) == len(num_crops)
    color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
    trans = []
    for i in range(len(size_crops)):
        randomresizedcrop = T.RandomResizedCrop(
            size_crops[i],
            scale=(min_scale_crops[i], max_scale_crops[i]),
        )
        trans.extend(
            [
                T.Compose(
                    [
                        randomresizedcrop,
                        T.RandomHorizontalFlip(p=0.5),
                        T.Compose(color_transform),
                        T.ToTensor(),
                        T.Normalize(mean=means, std=stds),
                    ]
                )
            ]
            * num_crops[i]
        )

    def transform(image):
        return list(map(lambda trans: trans(image), trans))

    return transform


class PILRandomGaussianBlur:
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if np.random.rand() <= self.prob:

            return img.filter(
                ImageFilter.GaussianBlur(
                    radius=np.random.uniform(self.radius_min, self.radius_max)
                )
            )
        else:
            return img


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
