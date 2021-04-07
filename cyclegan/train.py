import os
import random

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import Disc, Gen


class Net(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        features,
        num_residuals,
        num_features,
        lr_d,
        lr_g,
        betas_d,
        betas_g,
        lambda_cycle,
        lambda_identity,
        horse_dir,
        zebra_dir,
        batch_size,
    ):
        super().__init__()
        self.horse_dir = horse_dir
        self.zebra_dir = zebra_dir
        self.batch_size = batch_size
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.disc_h = Disc(in_channels, features)  # Discriminate horses
        self.disc_z = Disc(in_channels, features)  # Discriminate zebras
        self.gen_h = Gen(in_channels, num_residuals, num_features)
        self.gen_z = Gen(in_channels, num_residuals, num_features)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt_disc, opt_gen = self.optimizers()
        horse, zebra = batch

        # Train two discriminators
        # generate fake horse from zebra
        fake_horse = self.gen_h(8)
        d_h_real = self.disc_h(horse)
        d_h_fake = self.disc_h(fake_horse.detach())
        d_h_real_loss = self.mse_loss(d_h_real, torch.ones_like(d_h_real))
        d_h_fake_loss = self.mse_loss(d_h_fake, torch.zeros_like(d_h_fake))

        # generate fake zebra from horse
        fake_zebra = self.gen_z(horse)
        d_z_real = self.disc_z(zebra)
        d_z_fake = self.disc_z(fake_zebra.detach())
        d_z_real_loss = self.mse_loss(d_z_real, torch.ones_like(d_z_real))
        d_z_fake_loss = self.mse_loss(d_z_fake, torch.zeros_like(d_z_fake))

        d_loss = (d_h_real_loss + d_h_fake_loss + d_z_real_loss + d_z_fake_loss) / 2
        opt_disc.zero_grad()
        self.manual_backward(d_loss)
        opt_disc.step()
        self.log("disc_loss", d_loss, prog_bar=True, on_step=True, on_epoch=True)

        # Train two generators
        # Adversarial loss
        d_h_fake = self.disc_h(fake_horse)
        d_z_fake = self.disc_z(fake_zebra)
        g_h_loss = self.mse_loss(d_h_fake, torch.ones_like(d_h_fake))
        g_z_loss = self.mse_loss(d_z_fake, torch.ones_like(d_z_fake))

        # Cycle loss
        cycle_horse = self.gen_h(fake_zebra)
        cycle_zebra = self.gen_z(fake_horse)
        cycle_h_loss = self.l1_loss(horse, cycle_horse)
        cycle_z_loss = self.l1_loss(zebra, cycle_zebra)

        # Identity loss
        # identity_horse = self.gen_h(horse)
        # identity_zebra = self.gen_z(zebra)
        # i_h_loss = self.l1_loss(horse, identity_horse)
        # i_z_loss = self.l1_loss(zebra, identity_zebra)

        g_loss = (
            (g_h_loss + g_z_loss)
            + (cycle_h_loss + cycle_z_loss) * self.hparams.lambda_cycle
            # + (i_h_loss + i_z_loss) * self.hparams.lambda_identity
        )
        opt_gen.zero_grad()
        self.manual_backward(g_loss)
        opt_gen.step()
        self.log('gen_loss', g_loss, prog_bar=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        horse, zebra = batch
        if batch_idx > 0:
            return None
        tensorflow = self.logger.experiment
        if self.current_epoch == 0:
            tensorflow.add_image(
                "horses and zebras",
                make_grid(torch.cat((horse, zebra), dim=0) * 0.5 + 0.5),
            )
        tensorflow.add_image(
            "generated horses",
            make_grid(self.gen_h(zebra) * 0.5 + 0.5),
            self.current_epoch,
        )
        tensorflow.add_image(
            "generated zebras",
            make_grid(self.gen_z(horse) * 0.5 + 0.5),
            self.current_epoch,
        )

    def configure_optimizers(self):
        opt_disc = optim.Adam(
            list(self.disc_h.parameters()) + list(self.disc_z.parameters()),
            lr=self.hparams.lr_d,
            betas=self.hparams.betas_d,
        )
        opt_gen = optim.Adam(
            list(self.gen_h.parameters()) + list(self.gen_z.parameters()),
            lr=self.hparams.lr_g,
            betas=self.hparams.betas_g,
        )
        sche_dics = optim.lr_scheduler.LambdaLR(opt_disc, lambda epoch: 1 if epoch < 100 else (200 - epoch) / 100)
        sche_gen = optim.lr_scheduler.LambdaLR(opt_gen, lambda epoch: 1 if epoch < 100 else (200 - epoch) / 100)
        return [opt_disc, opt_gen], [sche_dics, sche_gen]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            HorseZebraDataset(self.horse_dir, self.zebra_dir),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            HorseZebraDataset(self.horse_dir, self.zebra_dir),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )


class HorseZebraDataset(Dataset):
    def __init__(self, horse_dir, zebra_dir) -> None:
        super().__init__()
        self.horse_images = [os.path.join(horse_dir, i) for i in os.listdir(horse_dir)]
        self.zebra_images = [os.path.join(zebra_dir, i) for i in os.listdir(zebra_dir)]

        less_images = (
            self.horse_images
            if len(self.horse_images) < len(self.zebra_images)
            else self.zebra_images
        )
        less_images.extend(
            random.sample(
                less_images, abs(len(self.horse_images) - len(self.zebra_images))
            )
        )

        self.transform = A.Compose(
            [
                A.Resize(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255
                ),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )

    def __len__(self):
        return len(self.horse_images)

    def __getitem__(self, index):
        horse_image_path = self.horse_images[index]
        zebra_image_path = self.zebra_images[index]
        horse_image = np.array(Image.open(horse_image_path).convert("RGB"))
        zebra_image = np.array(Image.open(zebra_image_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=horse_image, image0=zebra_image)
            horse_image = augmentations["image"]
            zebra_image = augmentations["image0"]
        return horse_image, zebra_image


if __name__ == "__main__":
    pl.seed_everything(2021)
    HORSE_DIR = "/home/tiankang/wusuowei/data/kaggle/horse2zebra/trainA"
    ZEBRA_DIR = "/home/tiankang/wusuowei/data/kaggle/horse2zebra/trainB"
    IN_CHANNELS = 3
    FEATURES = (64, 128, 256, 512)
    NUM_RESIDUALS = 9
    NUM_FEATURES = 64
    BATCH_SIZE = 1
    LAMBDA_IDENTITY = 0.0
    LAMBDA_CYCLE = 10
    NUM_EPOCH = 200

    LR = 2e-4
    BETAS_D = (0.5, 0.999)
    BETAS_G = (0.5, 0.999)


    # CHECKPOINT = '/home/tiankang/wusuowei/deeplearning/cyclegan/lightning_logs/version_1/checkpoints/epoch=14-step=18746.ckpt'
    CHECKPOINT = None
    
    cyclegan = Net(
        IN_CHANNELS,
        FEATURES,
        NUM_RESIDUALS,
        NUM_FEATURES,
        LR,
        LR,
        BETAS_D,
        BETAS_G,
        LAMBDA_CYCLE,
        LAMBDA_IDENTITY,
        HORSE_DIR,
        ZEBRA_DIR,
        BATCH_SIZE,
    )
    if not CHECKPOINT:
        trainer = pl.Trainer(
            max_epochs=NUM_EPOCH,
            gpus=[9],
            automatic_optimization=False,
            deterministic=True,
            precision=16,
        )
    else:
        trainer = pl.Trainer(resume_from_checkpoint=CHECKPOINT)
    trainer.fit(cyclegan)
