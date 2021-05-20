'''
https://www.youtube.com/watch?v=pG0QZ7OddX4

'''

import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T
import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler

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


class WGAN_GP(pl.LightningModule):
    def __init__(
        self,
        channels: int,
        feature_g: int,
        feature_c: int,
        latent_dim: int,
        lr: float,
        beta1: float,
        beta2: float,
        critic_iteration: int,
        lambda_gp: float
    ):
        super().__init__()
        self.save_hyperparameters()

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
        
        if optimizer_idx < self.hparams.critic_iteration:
            # Train critic
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)
            imgs_fake = self(z).detach()
            imgs_fake.requires_grad = True  # this line is necessary for calculating gradient in `self.gradient_penalty`
            gp = self.gradient_penalty(imgs, imgs_fake)
            critic_real = self.critic(imgs)
            critic_fake = self.critic(imgs_fake)
            c_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.hparams.lambda_gp * gp
            self.log('loss/critic', c_loss, on_epoch=True, prog_bar=True)
            return c_loss
        if optimizer_idx == self.hparams.critic_iteration:
            # Train generator: minimize -E[critic(imgs_fake)]
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)
            imgs_fake = self(z)
            critic_fake = self.critic(imgs_fake)
            g_loss = -torch.mean(critic_fake)
            self.log('loss/gen', g_loss, on_epoch=True, prog_bar=True)
            return g_loss
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, beta2))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(beta1, beta2))
        return [opt_c] * self.hparams.critic_iteration + [opt_g], []
    
    def toggle_optimizer(self, optimizer, optimizer_idx: int):
        pass

    def gradient_penalty(self, imgs, imgs_fake):
        epsilon = torch.rand(len(imgs), device=self.device).view(-1, 1, 1, 1)
        interpolated_imgs = epsilon * imgs + (1 - epsilon) * imgs_fake

        mixed_scores = self.critic(interpolated_imgs)
        # compute the gradient of the mixed scores with respect ot the interpolated images.
        gradient = torch.autograd.grad(
            inputs=interpolated_imgs,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0].view(len(imgs), -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty


class Celeb(pl.LightningDataModule):
    def __init__(self, root, batch_size, val_split):
        super().__init__()
        self.dims = (3, 64, 64)
        self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.root = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(IMAGE_SIZE),
            T.Normalize(self.mean, self.std)
        ])

    def setup(self, stage=None) -> None:
        if stage == 'fit' or stage is None:
            dataset = datasets.ImageFolder(root=self.root, transform=self.transform)
            self.dataset, _ = random_split(
                dataset, 
                [len(dataset) - int(len(dataset) * self.val_split), int(len(dataset) * self.val_split)],
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6, 7, 8, 9"

    ROOT = '/home/tiankang/wusuowei/dataset/celeb'
    LEARNING_RATE = 1e-4
    BETA1 = 0
    BETA2 = 0.9
    BATCH_SIZE = 64
    IMAGE_SIZE = (64, 64)
    LATENT_DIM = 100
    FEATURE_CRITIC = 64
    FEATURE_GEN = 64
    CRITIC_ITERATION = 5
    MEAN, STD = (0.5,), (0.5,)
    EPOCH = 5
    LAMBDA_GP = 10
    
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(IMAGE_SIZE),
        T.Normalize(MEAN, STD)
    ])
    
    dm = VisionModule(
        dataset_cls=datasets.MNIST,
        dims=(1, 28, 28),
        data_dir=ROOT,
        val_split=9/10,
        batch_size=BATCH_SIZE,
        img_size=IMAGE_SIZE,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform
    )
    dm = Celeb(ROOT, BATCH_SIZE, 9/10)

    model = WGAN_GP(
        channels=dm.dims[0],
        feature_c=FEATURE_CRITIC,
        feature_g=FEATURE_GEN,
        latent_dim=LATENT_DIM,
        lr=LEARNING_RATE,
        critic_iteration=CRITIC_ITERATION,
        beta1=BETA1,
        beta2=BETA2,
        lambda_gp=LAMBDA_GP,
    )

    callbacks = [
        TensorboardGenerativeModelImageSampler(num_samples=64),
        # Generates images and logs to tensorboard. Your model must implement the forward function for generation
        LatentDimInterpolator(interpolate_epoch_interval=5),
        # Interpolates the latent space for a model by setting all dims to zero and stepping through the first two dims increasing one unit at a time.
    ]

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCH,
        callbacks=callbacks,
        accelerator='ddp',
    )
    trainer.fit(model, dm)
