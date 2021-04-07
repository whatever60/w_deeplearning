import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MapDataset(Dataset):
    def __init__(self, root_dir, image_size, train) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(root_dir)
        self.train = train  # this determine how to augment input image

        self.transform = A.Compose(
            [
                A.Resize(width=image_size, height=image_size),
                A.HorizontalFlip(p=0.5)
            ],
            additional_targets={'image0': 'image'}
        )

        self.transform_input_only = A.Compose(
            [
                A.ColorJitter(p=0.1),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2()
            ]
        )

        self.transform_target_only = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2()
            ]
        )
    
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.list_files[index])
        image = np.asarray(Image.open(img_path))
        input_ = image[:, :image.shape[1] // 2]
        target = image[:, image.shape[1] // 2:]
        
        transformed = self.transform(image=input_, image0=target)
        input_, target = transformed['image'], transformed['image0']
        
        if self.train:
            input_ = self.transform_input_only(image=input_)['image']
        else:
            input_ = self.transform_target_only(image=input_)['image']
        target = self.transform_target_only(image=target)['image']
        return input_, target


def test():
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    dataset = MapDataset('/home/tiankang/wusuowei/data/maps/train')
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        save_image(x, 'x.jpg')
        save_image(y, 'y.jpg')
        break
    

if __name__ == '__main__':
    test()
