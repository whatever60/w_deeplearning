import torch
from torch import optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from model import Gen, Disc


class GanCallback(Callback):
    global_step = 123

    def on_epoch_end(self, trainer, pl_module):
        z = torch.randn((16, pl_module.hparams.z_dim), device=pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            fake_images = pl_module(z) * 0.5 + 0.5  # to [0, 1]
            pl_module.train()
        trainer.logger.experiment.add_image(
            "fake_images", make_grid(fake_images, nrow=4), global_step=self.global_step
        )
        self.global_step += 1

def sample(path, steps, alpha):
    pl_module = Net.load_from_checkpoint(path)
    pl_module.steps = steps
    pl_module.alpha = alpha
    z = torch.randn((16, pl_module.hparams.z_dim), device=pl_module.device)
    with torch.no_grad():
        pl_module.eval()
        fake_images = pl_module(z) * 0.5 + 0.5  # to [0, 1]
    save_image(fake_images, 'temp.jpg', nrow=4)



class Net(pl.LightningModule):
    def __init__(
        self, lr, betas, data_dir, factors, z_dim, in_channels, lambda_gp, lambda_drift
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.data_dir = data_dir

        self.critic = Disc(in_channels, factors)
        self.gen = Gen(z_dim, in_channels, factors)
        # self.alpha = 1e-5
        self.batch_size = None

    def forward(self, z):
        return self.gen(z)

    def configure_optimizers(self):
        opt_critic = optim.Adam(
            self.critic.parameters(), lr=self.hparams.lr, betas=self.hparams.betas
        )
        opt_gen = optim.Adam(
            self.gen.parameters(), lr=self.hparams.lr, betas=self.hparams.betas
        )
        return opt_critic, opt_gen

    def train_dataloader(self) -> DataLoader:
        self.delta_alpha = self.batch_size / (len(self.dataset) * self.epoch * 0.5)
        # Set increment of alpha per batch, after batch_size has been tuned.
        return DataLoader(
            self.dataset, self.batch_size, shuffle=True, pin_memory=True, num_workers=8
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt_critic, opt_gen = self.optimizers()
        real, _ = batch

        noise = torch.randn((real.shape[0], self.hparams.z_dim), device=self.device)
        fake = self.gen(noise)
        critic_real = self.critic(real)
        critic_fake = self.critic(fake.detach())
        gp = self.gradient_penalty(real, fake.detach())
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + self.hparams.lambda_gp * gp
            + self.hparams.lambda_drift * torch.mean(critic_real ** 2)
        )
        opt_critic.zero_grad()
        self.manual_backward(loss_critic)
        opt_critic.step()

        gen_fake = self.critic(fake)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        opt_gen.step()

        self.alpha += self.delta_alpha
        self.alpha = min(self.alpha, 1)

        self.log_dict(dict(gen_loss=loss_gen, critic_loss=loss_critic))
        # In official implementation, the above forward and backward should be repeated 4
        # times with the same batch of images, but we will just leave it the normal way.

    def gradient_penalty(self, real, fake):
        batch_size, c, h, w = real.shape
        beta = torch.rand((batch_size, 1, 1, 1), device=self.device).repeat(1, c, h, w)
        interpolated_image = real * beta + fake * (1 - beta)
        interpolated_image.requires_grad_(True)
        mixed_score = self.critic(interpolated_image)

        gradient = torch.autograd.grad(
            inputs=interpolated_image,
            outputs=mixed_score,
            grad_outputs=torch.ones_like(mixed_score),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def update_prog_status(self, epoch, steps, alpha, batch_size):
        size = 2 ** steps * 4
        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.batch_size = batch_size
        self.dataset = ImageFolder(root=self.data_dir, transform=transform)
        self.epoch = epoch
        self.steps = steps
        self.alpha = alpha

    @property
    def alpha(self):
        assert self.critic.alpha == self.gen.alpha
        return self.critic.alpha

    @alpha.setter
    def alpha(self, alpha):
        self.critic.alpha = self.gen.alpha = alpha

    @property
    def steps(self):
        assert self.critic.step == self.gen.step
        return self.critic.step

    @steps.setter
    def steps(self, step):
        self.critic.steps = self.gen.steps = step


if __name__ == "__main__":
    pl.seed_everything(2021)

    # ------ For optimizers ------
    LR = 1e-3
    BETAS = 0.0, 0.99

    # ------ For loader ------
    DATA_DIR = "/home/tiankang/wusuowei/data/kaggle/anime_faces_256"
    BATCH_SIZES = [64, 64, 32, 32, 32, 32, 16]
    # For resolution [4, 8, 16, 32, 64, 128, 256]
    PROGRESSIVE_EPOCHS = [10, 20, 20, 20, 20, 20, 20]
    # Note: the number of epochs should change depending on the size of your dataset.
    # In the paper, they feed in 800k images as they increase alpha to 1, and another 800k
    # with alpha = 1, except for the initial step (resolution = 4), where there is no
    # fade-in and they just feed in 800k images once.
    # Another implementation detail is that they did 4 "batch repeat" each step, where
    # they feed forward the same batch 4 times and update 4 times, before they went on to the next batch.

    # ------ For the network ------
    START_STEPS = 0
    FACTORS = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8]  # , 1 / 16, 1 / 32]
    Z_DIM = 256  # 512 in original paper
    IN_CHANNELS = 256  # 512 in original paper

    # ------ For loss ------
    LAMBDA_GP = 10
    LAMBDA_DRIFT = 0.001

    CHECKPOINT = '/home/tiankang/wusuowei/deeplearning/progan/lightning_logs/version_6/checkpoints/epoch=3-step=18722.ckpt'

    callback = GanCallback()
    progan = Net(
        LR, BETAS, DATA_DIR, FACTORS, Z_DIM, IN_CHANNELS, LAMBDA_GP, LAMBDA_DRIFT
    )
    if CHECKPOINT is None:
        
        for steps, epoch in enumerate(PROGRESSIVE_EPOCHS, START_STEPS):
            progan.update_prog_status(epoch, steps, alpha=1e-5)
            trainer = pl.Trainer(
                max_epochs=epoch,
                gpus=[8],
                callbacks=callback,
                deterministic=True,
                precision=16,
            )
            trainer.fit(progan)
    else:
        progan.update_prog_status(epoch=17, steps=6, alpha=0.3, batch_size=16)
        trainer = pl.Trainer(
            resume_from_checkpoint=CHECKPOINT,
            max_epochs=17,
            gpus=[8],
            callbacks=callback,
            deterministic=True,
            precision=16,
            # auto_scale_batch_size="binsearch",
        )
        trainer.fit(progan)