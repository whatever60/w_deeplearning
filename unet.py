import os

import numpy as np
from PIL import Image

import torch
from torch import is_deterministic, nn
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # print(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=(64, 128, 256, 512)
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        for feature in features[::-1]:
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def path1(self, x, skip_connections):
        for i in range(len(skip_connections)):
            x = self.decoder[i * 2](x)
            skip_connection = skip_connections[i]

            if x.shape != skip_connection.shape:  # because of rounding in Maxpool, the skip_connection might be larger than x
                x = TF.resize(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i * 2 + 1](x)
        return x

    def path2(self, x, skip_connections):
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            # print(idx)
            skip_connection = skip_connections[idx//2]
            # print(id(skip_connection))
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](x)

        return x
    
    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.path1(x, skip_connections)
        return self.final_conv(x)


class CARVANADataset(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))
        image = np.array(Image.open(data_path))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float)  # 
        mask[mask == 255.0] = 1.0
        # if self.trainsform is not None:
        augmentation = self.transform(image=image, mask=mask)
        image, mask = augmentation['image'], augmentation['mask']
        return image, mask


class CARVANADataModule(pl.LightningDataModule):
    def __init__(self, data_dir, mask_dir, val_data_dir, val_mask_dir, image_size, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.val_data_dir = val_data_dir
        self.val_mask_dir = val_mask_dir
        self.batch_size = batch_size
    
        self.transform_train = A.Compose([
            A.Resize(*image_size),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
        self.transform_val = A.Compose([
            A.Resize(*image_size),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            CARVANADataset(self.data_dir, self.mask_dir, self.transform_train),
            self.batch_size,
            True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            CARVANADataset(self.val_data_dir, self.val_mask_dir, self.transform_val),
            self.batch_size,
            False,
            num_workers=4
        )


class Net(pl.LightningModule):
    def __init__(self, in_channels, out_channels, lr, val_loader):
        super().__init__()
        self.val_loader = val_loader
        # self.sample = sample.to(self.device)
        self.save_hyperparameters()
        self.model = UNET(in_channels, out_channels)
        self.loss_fn = nn.BCEWithLogitsLoss()  # only for binary classification

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log('train_loss', loss)
        return dict(loss=loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log('val_loss', loss)

        pred = (torch.sigmoid(output) > 0.5).float()
        acc = (pred == y).sum() / torch.numel(pred)
        self.log('val_acc', acc)

        dice = (2 * pred * y).sum() / ((pred + y).sum() + 1e-8)
        self.log('val_dice', dice)
        # numerator： how many white pixels are predicted correctly.
        # denominator: how many white pixels in total does the prediction and y have. 
        # only for binary classification
    
    # def validation_epoch_end(self, outputs):
    #     sample_output = make_grid((torch.sigmoid(self(self.sample)) > 0.5).float())
    #     tensorboard = self.logger.experiment
    #     tensorboard.add_image(sample_output)
    def validation_epoch_end(self, output):
        for x, y in self.val_loader:
            x = x.to(self.device)
            preds = (torch.sigmoid(self(x)) > 0.5).float()
            y = y.unsqueeze(1)
            tensorboard = self.logger.experiment
            tensorboard.add_image('sample_pred', make_grid(preds), self.current_epoch)
            tensorboard.add_image('sample_truth', make_grid(y), self.current_epoch)
            break
    

if __name__ == '__main__':
    seed_everything(2021)
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 5
    IMAGE_SIZE = 160, 240
    IN_CHANNELS = 3
    OUT_CHANNELS = 1
    IMG_DIR = '/home/tiankang/wusuowei/data/kaggle/carvana/train'
    MASK_DIR = '/home/tiankang/wusuowei/data/kaggle/carvana/train_masks'
    VAL_IMG_DIR = '/home/tiankang/wusuowei/data/kaggle/carvana/val'
    VAL_MASK_DIR = '/home/tiankang/wusuowei/data/kaggle/carvana/val_masks'

    dm = CARVANADataModule(
        data_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        val_data_dir=VAL_IMG_DIR,
        val_mask_dir=VAL_MASK_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    model = Net(IN_CHANNELS, OUT_CHANNELS, LEARNING_RATE, dm.val_dataloader())
    trainer = pl.Trainer(max_epochs=EPOCHS, gpus=[7], deterministic=True)
    trainer.fit(model, dm)
    # model.load_from_checkpoint('/home/tiankang/wusuowei/deeplearning/lightning_logs/version_0/checkpoints/epoch=2-step=431.ckpt')
    # model.sample(
    #     '/home/tiankang/wusuowei/data/kaggle/carvana/val/0cdf5b5d0ce1_08.jpg',
    #     '/home/tiankang/wusuowei/deeplearning/temp/0cdf5b5d0ce1_08.jpg'
    # )
