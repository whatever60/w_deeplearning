import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl

from model import Gen, Disc
from dataset import MapDataset


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Net(pl.LightningModule):
    def __init__(self, in_channels, lr, betas, l1_lambda, batch_size, image_size):
        super().__init__()
        self.save_hyperparameters()
        self.prepare_sample_data()

        self.gen = Gen(in_channels, features=64)
        self.disc = Disc(in_channels, features=(64, 128, 256, 512))
        self.gen = self.gen.apply(_weights_init)
        self.disc = self.disc.apply(_weights_init)
        self.L1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def prepare_sample_data(self):
        val_data = DataLoader(MapDataset('/home/tiankang/wusuowei/data/maps/val', self.hparams.image_size, train=False), batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)
        for input_, target in val_data:
            self.input_ = input_
            self.target = target
            break


    def forward(self, x):
        return self.gen(x)

    def configure_optimizers(self):
        opt_disc = optim.Adam(self.disc.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        opt_gen = optim.Adam(self.gen.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        return opt_disc, opt_gen

    def training_step(self, batch, batch_idx, optimizer_idx):
        input_, target = batch
        if optimizer_idx == 0:
            # train disc
            # self.fake_target = self.gen(input_)
            # fake_target = self.fake_target.detach()
            fake_target = self.gen(input_).detach()
            d_fake = self.disc(input_, fake_target)
            d_real = self.disc(input_, target)
            fake_loss = self.bce_loss(d_fake, torch.zeros_like(d_fake))
            real_loss = self.bce_loss(d_real, torch.ones_like(d_real))
            loss = (fake_loss + real_loss) / 2
            self.log('disc_loss', loss)
            return loss
        elif optimizer_idx == 1:
            # train generator
            fake_target = self.gen(input_)
            # fake_target = self.fake_target
            d_fake = self.disc(input_, fake_target)
            fake_loss = self.bce_loss(d_fake, torch.ones_like(d_fake))
            reconstruction_loss = self.L1_loss(fake_target, target)
            loss = fake_loss + reconstruction_loss * self.hparams.l1_lambda
            self.log('gen_loss', loss)
            return loss

    def train_dataloader(self) -> DataLoader:
        return DataLoader(MapDataset('/home/tiankang/wusuowei/data/maps/train', self.hparams.image_size, train=True), batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

    # def val_dataloader(self):
    #     return DataLoader(MapDataset('/home/tiankang/wusuowei/data/maps/val', self.hparams.image_size), batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)

    # def toggle_optimizer(self, optimizer, optimizer_idx):
    #     param_requires_grad_state = {}
    #     for opt in self.optimizers(use_pl_optimizer=False):
    #         for group in opt.param_groups:
    #             for param in group['params']:
    #                 # If a param already appear in param_requires_grad_state, continue
    #                 if param in param_requires_grad_state:
    #                     continue
    #                 param_requires_grad_state[param] = param.requires_grad
    #     self._param_requires_grad_state = param_requires_grad_state

    def training_epoch_end(self, output):
        self.eval()
        input_ = self.input_.to(self.device)
        target = self.target.to(self.device)
        tensorboard = self.logger.experiment
        if self.current_epoch == 0:
            tensorboard.add_image('sample_input', make_grid(input_), self.current_epoch)
            tensorboard.add_image('sample_target', make_grid(target), self.current_epoch)
        fake_target = self.gen(input_)
        tensorboard.add_image('sample_fake', make_grid(fake_target), self.current_epoch)


    def test(self):
        print(self(torch.randn(10, 3, 256, 256)).shape)

    def sample(self):
        self.eval()
        for input_, target in self.val_dataloader():
            save_image(input_, 'x.jpg')
            save_image(target, 'y.jpg')
            save_image(self.gen(input_), 'z.jpg')
            break


if __name__ == '__main__':
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 32
    IMAGE_SIZE = 256
    IN_CHANNELS = 3
    L1_LAMBDA = 100
    BETAS = (0.5, 0.999)

    EPOCHS = 500

    trainer = pl.Trainer(gpus=[3], max_epochs=EPOCHS, precision=16)
    pix2pix = Net(IN_CHANNELS, LEARNING_RATE, BETAS, L1_LAMBDA, BATCH_SIZE, IMAGE_SIZE)
    trainer.fit(pix2pix)
    

