from torch import nn
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models import UNet
from datamodules import SNARESeqFixedDataModule
from callbacks import RefPearson


def weight_init(m):
    if isinstance(m, nn.Linear):
        # nn.init.uniform_(m.weight)
        # nn.init.zeros_(m.weight)
        if m.bias is not None:
            # nn.init.eye_(m.bias)
            nn.init.zeros_(m.bias)


class Net(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        features,
        lr,
        batch_size,  # just for hyperparameter logging
        binomial_p,
        # min_dropout_p,
        # max_dropout_p,
        # p_noise,
        # min_noise_scale,
        # max_noise_scale,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_dim, in_dim, features)
        self.criterion_recon = nn.MSELoss()
        # self.model.apply(weight_init)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        input_, target = batch  # 1 means mask in the input.
        pred = self(input_)
        loss_reconstruction = self.criterion_recon(pred, target)
        loss = loss_reconstruction
        return pred, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self.shared_step(batch)
        self.log_dict(dict(val_loss=loss, density=(pred != 0).float().mean()))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
        # return dict(
        #     optimizer=optimizer,
        #     lr_scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5),
        #     interval="epoch",
        # )


if __name__ == "__main__":
    from rich.traceback import install

    install()
    pl.seed_everything(42)

    # DATA_PATH_TRAIN = (
    #     "/home/tiankang/wusuowei/data/single_cell/snare_seq/processed/train/data.h5ad"
    # )
    # DATA_PATH_VAL = (
    #     "/home/tiankang/wusuowei/data/single_cell/snare_seq/processed/val/data.h5ad"
    # )
    DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/snare_seq/processed"
    BATCH_SIZE = 128
    BINOMIAL_P = 0.85
    # MIN_DROPOUT_P = 0.5
    # MAX_DROPOUT_P = 0.5
    # P_NOISE = 0.5
    # MIN_NOISE_SCALE = 0.02
    # MAX_NOISE_SCALE = 0.05

    module = SNARESeqFixedDataModule(
        DATA_DIR,
        BATCH_SIZE,
        # MIN_DROPOUT_P,
        # MAX_DROPOUT_P,
        # P_NOISE,
        # MIN_NOISE_SCALE,
        # MAX_NOISE_SCALE,
    )

    FEATURES = [64, 32]
    LR = 1e-4

    model = Net(
        in_dim=module.dims[0],
        features=FEATURES,
        lr=LR,
        batch_size=BATCH_SIZE,
        binomial_p=BINOMIAL_P,
        # min_dropout_p=MIN_DROPOUT_P,
        # max_dropout_p=MAX_DROPOUT_P,
        # p_noise=P_NOISE,
        # min_noise_scale=MIN_NOISE_SCALE,
        # max_noise_scale=MAX_NOISE_SCALE,
    )

    DATA_PATH_PRED = (
        "/home/tiankang/wusuowei/data/single_cell/snare_seq/processed/data.h5ad"
    )
    DATA_PATH_REF = (
        "/home/tiankang/wusuowei/data/single_cell/split_seq/processed/data.h5ad"
    )
    GENE_PATH_PRED = (
        "/home/tiankang/wusuowei/data/single_cell/snare_seq/processed/gene.csv"
    )
    GENE_PATH_REF = (
        "/home/tiankang/wusuowei/data/single_cell/split_seq/processed/gene.csv"
    )

    callbacks = [
        # EarlyStopping(monitor="global_density", patience=10, mode="max", verbose=False),
        # ModelCheckpoint(save_top_k=-1, period=10),
        RefPearson(DATA_PATH_PRED, DATA_PATH_REF, GENE_PATH_PRED, GENE_PATH_REF),
    ]

    CHECKPOINT = ""
    if not CHECKPOINT:
        trainer = pl.Trainer(
            # fast_dev_run=True,
            gpus=[1],
            deterministic=True,
            max_epochs=2000,
            callbacks=callbacks,
        )
        trainer.fit(model, module)
