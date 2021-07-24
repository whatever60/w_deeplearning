from torch import nn
from torch import optim
import pytorch_lightning as pl

from models import UNET
from datamodule import RNADenoisingDataModule


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight)
        # nn.init.zeros_(m.weight)
        if m.bias is not None:
            # nn.init.eye_(m.bias)
            nn.init.zeros_(m.bias)
        # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
    # if isinstance(m, nn.BatchNorm2d):
    #     torch.nn.init.normal_(m.weight, 0.0, 0.02)
    #     torch.nn.init.constant_(m.bias, 0)


class Net(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        features,
        lr,
        mask_weight,
        batch_size,  # just for hyperparameter logging
        binomial_p,
        mask_p,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNET(in_dim, in_dim, features)
        self.criterion_recon = nn.MSELoss()
        self.criterion_mask = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        input_, target, mask = batch  # 1 means mask in the input.
        pred = self(input_)
        loss_reconstruction = self.criterion_recon(pred, target)
        loss_mask = self.criterion_mask(
            pred.masked_fill(~mask, 0), target.masked_fill(~mask, 0)
        )
        loss = 0 * loss_reconstruction + self.hparams.mask_weight * loss_mask
        return pred, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.shared_step(batch)
        self.log("trian_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self.shared_step(batch)
        self.log_dict(dict(val_loss=loss, density=(pred != 0).float().mean()))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        # return optimizer
        return dict(
            optimizer=optimizer,
            lr_scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5),
            interval="epoch",
        )


if __name__ == "__main__":
    # version 14
    # version 15: L1Loss -> MSELoss
    # version 16: Normalizer: MINMAX -> l1 norm
    # version 17: MSELoss -> L1Loss, as a proof of concept
    # version 18: Continue training of version 17
    # version 19: Learning rate: 2e-4 -> 1e-3
    # version 20: StepLR (20, 0.5)
    # version 21: Scale after normalization
    # version 22: Continue training of version 21
    # version 23: Add Tanh as the last activation
    # version 24: Continue training of version 23
    # version 25: L1Loss -> MSELoss
    # version 26: Mutual prediction dataset
    # version 27: add qc
    # version 28: raw count to PCA

    # version 30: New init, replace LinearBasicBlock, simpler preprocess
    # version 31: Forgot to init bias in version 30, higher batch size (128)
    # version 32: Remove (x - 0.5) / 0.5 shift, replace tanh with sigmoid
    # version 33: Batch size back to 64, learning rate doubled (2e-3)
    # version 34: Continue training of version 33.
    # version 35: Change MSELoss to L1Loss
    # version 36: Add sparsity monitor
    # version 37: Remove sigmoid
    # version 38: Increase learning rate to 1e-2
    # version 39: Learning rate back to 2e-3, add ReLU as the final activation
    # version 40: Replace Adam with SGD
    # version 41: Huge learning rate (1)
    # version 42: Learning rate 10
    # version 43: Learning rate 1e-2, loss multiplied by 1e4
    # version 44: Remove init.
    # version 45: Adam instead of SGD, ReLU instead of LeakyReLU
    # version 46: Profile multiplied by 10 after normalization
    # version 47: Uniform init.
    # version 48: L1 first, smaller model.
    # version 49: QC, learning rate 1e-3, replace ReLU with ELU
    # version 50: Replace ELU with ReLU
    # version 51: Add shift back, replace final ReLU with Tanh
    # version 52: Remove shift, activation ShiftedReLU, final ReLU
    # version 53: Add exp when computing loss, replace L1Loss with MSELoss
    # Something is wrong with version number at this point.
    # version 54: Final RectifiedTanh

    # version 55: Everything restart from here
    # version 56: Less epochs: 50.
    # 57: Less epochs: 10.
    # 58: Epochs: 30.
    # ==== Add mask ====
    # 59: Epochs: 50. masked_p = 0.05, mask_weight = 0.2.
    # 60: mask_weight = 0.05
    # 61: mask_weight = 0.01
    # 62: Epoch: 100.
    # 63: No reconstruction loss
    # 64: mask_p = 0.15. Epochs: 50.
    # ==== Above mask experiments are wrong ====
    # 65: mask_p = 0.15. Epochs: 50.
    # 66: mask_p = 0.15. Epochs: 30.
    # 67: Dual task, mask_weight = 0.1, 30 epochs.
    # 68: N2V task. mask_p = 0.15. Epochs: 100.

    from rich.traceback import install

    install()
    pl.seed_everything(42)

    CACHE_DIR = "./data"
    BATCH_SIZE = 64
    BINOMIAL_P = 0.85
    MASK_P = 0.15

    IN_DIM = 13183
    FEATURES = [128, 32]
    LR = 1e-3
    MASK_WEIGHT = 1
    EPOCHS = 100
    dataloader = RNADenoisingDataModule(
        CACHE_DIR,
        BATCH_SIZE,
        BINOMIAL_P,
        MASK_P,
    )

    CHECKPOINT = ""  # "/home/tiankang/wusuowei/deeplearning/single_cell/babel/denoise/lightning_logs/version_52/checkpoints/epoch=29-step=7739.ckpt"
    TRAIN = True
    model = Net(
        IN_DIM,
        FEATURES,
        LR,
        MASK_WEIGHT,
        BATCH_SIZE,
        BINOMIAL_P,
        MASK_P,
    )
    if not CHECKPOINT:
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            gpus=[5],
            # precision=16,
            deterministic=True,
            # fast_dev_run=True,
        )
    else:
        trainer = pl.Trainer(
            resume_from_checkpoint=CHECKPOINT,
            max_epochs=200,
            gpus=[9],
            # precision=16,
            deterministic=True,
        )
    if TRAIN:
        trainer.fit(model, dataloader)
    else:
        trainer.test(model, dataloader)
