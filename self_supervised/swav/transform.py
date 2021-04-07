from torchvision import transforms as T


class SwAVTransformTrain:
    def __init__(
        self,
        means,
        stds,
        size_crops,
        num_crops,
        min_scale_crop,
        max_scale_crop,
        use_gaussian_blur,
        jitter_strength
    ) -> None:
        transforms = []
        color_jitter = T.RandomApply(
            T.ColorJitter(
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.2 * jitter_strength
            ),
            p=0.8
        )
        color_transform = [color_jitter, T.RandomGrayscale(p=0.2)]
        
        if use_gaussian_blur:
            kernel_size = 0.1 * size_crops[0] // 2 * 2 + 1
            gaussian_blur = T.RandomApply(T.GaussianBlur(kernel_size, (0.1, 2)), p=0.5)
            color_transform.append(gaussian_blur)
        
        for size, num, min_, max_ in zip(size_crops, num_crops, min_scale_crop, max_scale_crop):
            for _ in range(num):
                transforms.append(T.Compose(
                    T.ToTensor(),
                    T.RandomResizedCrop(size, scale=(min_, max_)),
                    T.RandomHorizontalFlip(p=0.5),
                    *self.color_transform,
                    T.Normalize(means, stds)
                ))

        online_train_transform = T.Compose([  # no color transform
            T.ToTensor(),
            T.RandomSizedCrop(size_crops[0]),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(means, stds)
        ])
        transforms.append(online_train_transform)
        self.transforms = transforms

    def __call__(self, image):
        return list(map(lambda t: t(image), self.transforms))
        