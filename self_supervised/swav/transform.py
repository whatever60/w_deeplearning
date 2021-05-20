"""
Adapted from:
    - https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/transforms.py
"""

from torchvision import transforms as T


class SwAVTransform:
    def __init__(
        self,
        means,
        stds,
        size_crops,
        num_crops,
        min_scale_crop,
        max_scale_crop,
        use_gaussian_blur,
        jitter_strength,
        train,
    ) -> None:
        transforms = []
        color_jitter = T.RandomApply(
            T.ColorJitter(
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.2 * jitter_strength,
            ),
            p=0.8,
        )
        color_transform = [color_jitter, T.RandomGrayscale(p=0.2)]

        if use_gaussian_blur:
            kernel_size = 0.1 * size_crops[0] // 2 * 2 + 1
            gaussian_blur = T.RandomApply(T.GaussianBlur(kernel_size, (0.1, 2)), p=0.5)
            color_transform.append(gaussian_blur)

        for size, num, min_, max_ in zip(
            size_crops, num_crops, min_scale_crop, max_scale_crop
        ):
            for _ in range(num):
                # Add this same transform `num` times.
                transforms.append(
                    T.Compose(
                        T.ToTensor(),
                        T.RandomResizedCrop(size, scale=(min_, max_)),
                        T.RandomHorizontalFlip(p=0.5),
                        *self.color_transform,
                        T.Normalize(means, stds)
                    )
                )

        if train:    
            transforms.append(T.Compose(  # no color transform
                [
                    T.ToTensor(),
                    T.RandomSizedCrop(size_crops[0]),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(means, stds),
                ]
            ))
        else:
            transforms.append(T.Compose(
                [
                    transforms.Resize(int(size_crops[0] + 0.1 * size_crops[0])),
                    transforms.CenterCrop(size_crops[0]),
                    self.final_transform,
                ]
            ))

        self.transforms = transforms

    def __call__(self, image):
        return list(map(lambda t: t(image), self.transforms))


class SwAVFinetuneTransform(object):
    """
    Finetune doesn't need that much augmentation.
    """
    def __init__(
        self,
        means,
        stds,
        input_height,
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
                T.RandomResizedCrop(size=input_height),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([self.color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.Normalize(means, stds)
            ])
        else:
            self.transform = [
                T.ToTensor(),
                T.Resize(int(input_height + 0.1 * input_height)),
                T.CenterCrop(input_height),
                T.Normalize(means, stds)
            ]

    def __call__(self, sample):
        return self.transform(sample)
