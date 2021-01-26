'''
https://www.youtube.com/watch?v=pG0QZ7OddX4

'''

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler
from torchvision.transforms.transforms import Compose

from datamodule import VisionModule


class Critic(nn.Module):
    def __init__(
        self,
        channels,
        feature_c
    ):
        super().__init__()
        # input image: 64 * 64
        self.critic = nn.Sequential(
            nn.Conv2d(channels, feature_c, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self.gen_block(feature_c * 1, feature_c * 2),  # 16 * 16
            self.gen_block(feature_c * 2, feature_c * 4),  # 8 * 8
            self.gen_block(feature_c * 4, feature_c * 8),  # 4 * 4
            nn.Conv2d(feature_c * 8, 1, 4, 2, 0),  # 1 * 1
        )

    @staticmethod
    def gen_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.critic(x).reshape(-1)

    
class Generator(nn.Module):
    def __init__(self, latent_dim, channels, feature_g=64):
        # input noise: 1 * 1
        super().__init__()
        self.gen = nn.Sequential(
            self.gen_block(latent_dim, feature_g * 16, 4, 1, 0),  # 4 * 4
            self.gen_block(feature_g * 16, feature_g * 8, 4, 2, 1),  # 8 * 8
            self.gen_block(feature_g * 8, feature_g * 4, 4, 2, 1),  # 16 * 16
            self.gen_block(feature_g * 4, feature_g * 2, 4, 2, 1),  # 32 * 32
            nn.ConvTranspose2d(feature_g * 2, channels, 4, 2, 1),  # 64 * 64
            nn.Tanh()
        )

    @staticmethod
    def gen_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class WGAN(pl.LightningModule):
    def __init__(
        self,
        channels: int,
        feature_g: int,
        feature_c: int,
        latent_dim: int,
        lr: float,
        critic_iteration: int,
        weight_clip: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.gen = Generator(
            latent_dim=self.hparams.latent_dim,
            channels=self.hparams.channels,
            feature_g=self.hparams.feature_g
        )
        
        self.critic = Critic(
            channels=self.hparams.channels,
            feature_c=self.hparams.feature_c
        )
        initialize_weights(self.gen)
        initialize_weights(self.critic)
    
    def forward(self, z):
        z = z.reshape(-1, self.hparams.latent_dim, 1, 1)
        return self.gen(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        
        opt_c, opt_g = self.optimizers(use_pl_optimizer=True)
        for _ in range(self.hparams.critic_iteration):
            # Train critic
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)
            imgs_fake = self(z)
            critic_real = self.critic(imgs)
            critic_fake = self.critic(imgs_fake.detach())
            c_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))
            self.manual_backward(c_loss, opt_c)
            opt_c.step()
            self.log('loss/critic', c_loss, on_epoch=True, prog_bar=True)
            for p in self.critic.parameters():
                # clipping the weight to be within a small range so that the critic is one lipschitz continuous
                p.data.clamp_(-self.hparams.weight_clip, self.hparams.weight_clip)
        else:
            # Train generator: minimize -E[critic(imgs_fake)]
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)
            imgs_fake = self(z)
            critic_fake = self.critic(imgs_fake)
            g_loss = -torch.mean(critic_fake)
            self.manual_backward(g_loss, opt_g)
            opt_g.step()
            self.log('loss/gen', g_loss, on_epoch=True, prog_bar=True)
            return g_loss
        # if optimizer_idx < self.hparams.critic_iteration:
        #     # Train critic
        #     z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)
        #     imgs_fake = self(z)
        #     critic_real = self.critic(imgs)                                   
        #     critic_fake = self.critic(imgs_fake.detach())
        #     c_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))
        #     self.log('loss/critic', c_loss, on_epoch=True, prog_bar=True)
        #     return c_loss
        
        # if optimizer_idx == self.hparams.critic_iteration:
        #     # Train generator: minimize -E[critic(imgs_fake)]
        #     z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)
        #     imgs_fake = self(z)
        #     critic_fake = self.critic(imgs_fake)
        #     g_loss = -torch.mean(critic_fake)
        #     self.log('loss/gen', g_loss, on_epoch=True, prog_bar=True)
        #     return g_loss
    
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, *args, **kwargs):
    #     optimizer.step(closure=closure)

    #     if optimizer_idx < self.hparams.critic_iteration:
    #         for p in self.critic.parameters():
    #             # clipping the weight to be within a small range so that the critic is one lipschitz continuous
    #             p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.RMSprop(self.gen.parameters(), lr=lr)
        opt_c = torch.optim.RMSprop(self.critic.parameters(), lr=lr)
        # return [opt_c] * self.hparams.critic_iteration + [opt_g], []
        return [opt_c, opt_g], []


class Celeb(pl.LightningDataModule):
    def __init__(self, root, batch_size):
        super().__init__()
        self.dims = (3, 64, 64)
        self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.root = root
        self.batch_size = batch_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(IMAGE_SIZE),
            T.Normalize(self.mean, self.std)
        ])

    def setup(self, stage=None) -> None:
        if stage == 'fit' or stage is None:
            self.dataset = datasets.ImageFolder(root=self.root, transform=self.transform)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7, 8, 9"

    ROOT = '/home/tiankang/wusuowei/dataset'
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 64
    IMAGE_SIZE = (64, 64)
    LATENT_DIM = 100
    EPOCH = 10
    MEAN, STD = (0.5,), (0.5,)
    FEATURE_CRITIC = 64
    FEATURE_GEN = 64
    CRITIC_ITERATION = 5
    WEIGHT_CLIP = 0.01

    transform = T.Compose([
        T.ToTensor(),
        T.Resize(IMAGE_SIZE),
        T.Normalize(MEAN, STD)
    ])
    
    dm = VisionModule(
        dataset_cls=datasets.MNIST,
        dims=(1, 28, 28),
        data_dir=ROOT,
        val_split=0,
        batch_size=BATCH_SIZE,
        img_size=IMAGE_SIZE,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform
    )
    # dm = Celeb(ROOT, BATCH_SIZE)

    model = WGAN(
        channels=dm.dims[0],
        feature_g=FEATURE_CRITIC,
        feature_c=FEATURE_GEN,
        latent_dim=LATENT_DIM,
        lr=LEARNING_RATE,
        critic_iteration=CRITIC_ITERATION,
        weight_clip=WEIGHT_CLIP,
    )

    callbacks = [
        TensorboardGenerativeModelImageSampler(num_samples=64),
        # Generates images and logs to tensorboard. Your model must implement the forward function for generation
        LatentDimInterpolator(interpolate_epoch_interval=5),
        # Interpolates the latent space for a model by setting all dims to zero and stepping through the first two dims increasing one unit at a time.
    ]

    trainer = pl.Trainer(
        gpus=2,
        max_epochs=EPOCH,
        callbacks=callbacks,
        accelerator='ddp',
    )
    trainer.fit(model, dm)
