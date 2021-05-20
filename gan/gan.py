'''
https://www.youtube.com/watch?v=IZtv9s_Wx9I
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/b0507b3fee1fbd8125777e66d831cb664ead4660/ML/Pytorch/GANs/2.%20DCGAN/train.py
https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/03-basic-gan.ipynb
https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/gans/dcgan/dcgan_module.py#L21-L173
https://zhuanlan.zhihu.com/p/43843694

Generally speaking there are 3 forward/backward strategies.

The first is to use backward(retain_graph=True). This way we only need one feed-forward.
But we do backward twice on both generator and discriminator.
If we do not want to calculate gradient for generator when doing backward on discriminator,
we need to use detach and do feed-forward again before backprop on generator. In this way 
one more feed-forward on discriminator is needed but one backprop on generator is spared.
Typically generator is larger than discriminator and calculating gradient is more expansive,
so it is well worth it.

The third way is update generator first, which is even more computationally efficient, but
it makes more sense to backprop on discriminator first.
'''

import os

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms as T
import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler

from datamodule import VisionModule


class Discriminator(nn.Module):
    def __init__(
        self,
        channels,
        feature_d
    ):
        super().__init__()
        # input image: 64 * 64
        self.disc = nn.Sequential(
            nn.Conv2d(channels, feature_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self.gen_block(feature_d * 1, feature_d * 2),  # 16 * 16
            self.gen_block(feature_d * 2, feature_d * 4),  # 8 * 8
            self.gen_block(feature_d * 4, feature_d * 8),  # 4 * 4
            nn.Conv2d(feature_d * 8, 1, 4, 2, 0),  # 1 * 1
            nn.Sigmoid()
        )

    @staticmethod
    def gen_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x).reshape(-1)

    
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


class DCGAN(pl.LightningModule):
    def __init__(
        self,
        channels: int = 3,
        feature_g: int = 64,
        feature_d: int = 64,
        latent_dim: int = 100,
        lr: float = 2e-4,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gen = Generator(
            latent_dim=self.hparams.latent_dim,
            channels=self.hparams.channels,
            feature_g=self.hparams.feature_g
        )
        
        self.disc = Discriminator(
            channels=self.hparams.channels,
            feature_d=self.hparams.feature_d
        )
        initialize_weights(self.gen)
        initialize_weights(self.disc)

        self.criterion = nn.BCELoss()
    
    def forward(self, z):
        z = z.reshape(-1, self.hparams.latent_dim, 1, 1)
        return self.gen(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        opt_d, opt_g = self.optimizers(use_pl_optimizer=True)

        # Train discriminator
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)
        disc_real = self.disc(imgs)
        disc_fake = self.disc(generated_imgs.detach())
        real_loss = self.criterion(disc_real, torch.ones_like(disc_real))
        fake_loss = self.criterion(disc_fake, torch.zeros_like(disc_fake))
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss, opt_d)
        opt_d.step()
        # Train generator
        disc_fake = self.disc(generated_imgs)
        g_loss = self.criterion(disc_fake, torch.ones_like(disc_fake))
        self.manual_backward(g_loss, opt_g)
        opt_g.step()

        self.log('loss/disc', d_loss, on_epoch=True, prog_bar=True)
        self.log('loss/gen', g_loss, on_epoch=True, prog_bar=True)


        # return d_loss
    
        # # z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)
        # self.generated_imgs = self(self.z)
        # disc_fake = self.disc(self.generated_imgs)
        # # we need to forward again to build computational graph.
        # g_loss = self.criterion(disc_fake, torch.ones_like(disc_fake))
        # return g_loss
        
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))
        return [opt_d, opt_g], []


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "8, 9"

    ROOT = '/home/tiankang/wusuowei/dataset'
    LEARNING_RATE = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    BATCH_SIZE = 128
    IMAGE_SIZE = (64, 64)
    MEAN, STD = (0.5,), (0.5,)
    LATENT_DIM = 100
    EPOCH = 5
    FEATURE_DISC = 64
    FEATURE_GEN = 64

    transform = T.Compose([
        T.ToTensor(),
        T.Resize(IMAGE_SIZE),
        T.Normalize(MEAN, STD)
    ])
    
    dm = VisionModule(
        dataset_cls=datasets.MNIST,
        dims=(1, 28, 28),
        data_dir=ROOT,
        batch_size=BATCH_SIZE,
        img_size=IMAGE_SIZE,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform
    )

    model = DCGAN(
        channels=dm.dims[0],
        feature_g=FEATURE_DISC,
        feature_d=FEATURE_GEN,
        latent_dim=LATENT_DIM,
        lr=LEARNING_RATE,
        b1=BETA1,
        b2=BETA2,
    )

    callbacks = [
        TensorboardGenerativeModelImageSampler(num_samples=16),
        # Generates images and logs to tensorboard. Your model must implement the forward function for generation
        LatentDimInterpolator(interpolate_epoch_interval=5),
        # Interpolates the latent space for a model by setting all dims to zero and stepping through the first two dims increasing one unit at a time.
    ]

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCH,
        callbacks=callbacks,
        accelerator='ddp',
        automatic_optimization=False
    )
    trainer.fit(model, dm)
