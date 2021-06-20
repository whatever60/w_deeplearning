import torch
from torch import nn
from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torchmetrics import Accuracy

import models
import datamodules


class Net(pl.LightningModule):
    def __init__(
        self,
        backbone,
        lr,
        weight_decay,
        batch_size,
        **model_params,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = backbone(
            **model_params
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, x):  # [batch_size, in_channels, input_size]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)
        self.log("loss_train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        acc = self.metric(preds.argmax(dim=1), labels)
        loss = self.criterion(preds, labels)
        self.log_dict(dict(acc_val=acc, loss_val=loss))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = dict(
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, patience=5
            ),
            monitor="loss_val",
        )
        return [optimizer], [lr_scheduler]


def get_aug(size, means, stds):
    aug_train = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size),
            # T.RandomResizedCrop(size),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.Normalize(means, stds),
        ]
    )

    aug_val = T.Compose([T.ToTensor(), T.Normalize(means, stds)])
    return aug_train, aug_val


if __name__ == "__main__":
    from rich.traceback import install

    install()

    pl.seed_everything(42)

    DATA_DIR = "~/wusuowei/data/"
    BATCH_SIZE = 128
    DATAMODULE = datamodules.CIFAR10DataModule

    datamodule = DATAMODULE(
        DATA_DIR,
        BATCH_SIZE,
        *get_aug(DATAMODULE.size, DATAMODULE.means, DATAMODULE.stds),
    )

    NUM_LAYERS = 12
    PATCH_SIZE = 8
    EMB_DIM = 768
    EXPANSION = 4
    # ATT_DIM = None
    # TOKENS_HID_DIM = 384
    # CHANNELS_HID_DIM = 1024

    DROPOUT = 0.2
    LR = 5e-3
    WEIGHT_DECAY = 1e-4

    MAX_EPOCHS = 100

    model = Net(
        backbone=models.ResMLPVision,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        # ---- model specific parameters ----
        num_layers=NUM_LAYERS,
        in_channels=DATAMODULE.dims[0],
        input_size=DATAMODULE.size[0],
        patch_size=PATCH_SIZE,
        emb_dim=EMB_DIM,
        expansion=EXPANSION,
        # att_dim=ATT_DIM,
        # tokens_hid_dim=TOKENS_HID_DIM,
        # channels_hid_dim=CHANNELS_HID_DIM,
        num_classes=DATAMODULE.num_classes,
    )

    CHECKPOINT = ""

    if not CHECKPOINT:
        callbacks = [
            EarlyStopping(
                monitor="acc_val",
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode="max",
            ),
            ModelCheckpoint(monitor="acc_val", save_last=True, save_top_k=2),
            LearningRateMonitor(logging_interval='epoch')
        ]

        trainer = pl.Trainer(
            # fast_dev_run=True,
            gpus=[2],
            deterministic=True,
            max_epochs=MAX_EPOCHS,
            callbacks=callbacks,
        )

        trainer.fit(model, datamodule)
