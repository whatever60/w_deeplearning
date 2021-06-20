import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models import NPID
from datamodule import scRNADataModule, SNAREDataModule
from loss import NCECriterion, LinearAverage
from callbacks import KNNOnlineEvaluator

import warnings

warnings.filterwarnings("ignore")


class Net(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        dims,
        feature_dim,
        lr,
        nce_k,  # number of negative samples
        nce_t,  # temperature
        nce_m,  # momentum for memory update
        length_train,  # number of training data, also number of classes.
        batch_size,  # just for hyper-parameter logging.
        n_neighbors,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = NPID(in_dim, dims, feature_dim, length_train, nce_m)
        if nce_k > 0:
            # use NCE Loss
            self.criterion = NCECriterion(nce_k, nce_t)
        else:
            self.criterion = LinearAverage(feature_dim, length_train, nce_t, nce_m)

    def forward(self, imgs, indices=None):
        return self.model(imgs, indices)

    def training_step(self, batch, batch_idx):
        imgs, _, indices = batch
        repres = self(imgs, indices)
        loss = self.criterion(repres, indices, self.model.memory)
        self.log("train_loss", loss)
        return loss

    def on_after_backward(self) -> None:
        self.model.update_memory()

    # def validation_step(self, batch, batch_idx):
    #     imgs, labels_val, indices = batch
    #     repres = self(imgs)
    #     return dict(repres=repres, labels_val=labels_val, indices=indices)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[80, 120, 160], gamma=0.1
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    # version 4: start
    # version 5: higher knn N: 150
    # version 6: Lower knn N: 50
    # version 7: Even lower knn N: 20. Higher batch size: 256
    # version 8: Higher knn N: 40
    # version 9: Smaller batch size 64
    # version 10: Adam. LR 1e-3.
    # version 11: LR 5e-3
    # version 12: LR 5e-4
    # version 13: LR 3e-4
    # version 14: LR 1e-4
    # version 16: Larger nce_K: 4096 * 2
    # version 17: Very large batch size: 1024. Larger lr: 3e-4
    # version 18: Batch size 64.
    # version 19: Continue of version 16
    # version 20: Larger batch size: 2048. Larger lr: 3e-4
    # version 21: Batch size 32
    # version 22: Trash

    # ====== Everything restart from here on ======
    # version 23: Adam. LR 1e-4
    # version 24: Adam. LR 3e-4 √
    # version 26: Adam. LR 1e-3
    # version 27: Adam. LR 5e-4
    # version 28: Adam. LR 5e-5
    # version 29: SGD. LR 5e-4
    # version 30: SGD. LR 1e-3
    # version 31: SGD. LR 3e-3
    # version 32: SGD. LR 5e-3
    # version 33: SGD. LR 1e-2

    # version 39: Adam. LR 3e-4. Batch size 32
    # version 40: Adam. LR 3e-4. Batch size 32, built-in multinomial sampler.
    # version 41: Adam. LR 3e-4. Batch size 128
    # version 42: Adam. LR 3e-4. Batch size 512
    # version 43: Adam. LR 3e-4. Batch size 2048
    # version 45: Adam. LR 3e-4. Batch size 4096 √
    # version 46: Same as version 45, but two gpus are used, so total batch size 8192
    # version 47: Adam. LR 3e-4. Batch size 8192
    # version 48: Adam. LR 3e-4. Batch size 9000
    # version 49: Adam. LR 3e-4. Batch size 9000. Two GPUs
    # version 50: Replicate of version 46
    
    # version 52: NCE_K 4096 * 2. Adam. LR 3e-4. Batch size 4096
    # version 54: Feature_dim 30. NCE_K 4096 * 2. Adam. LR 3e-4. Batch size 4096
    # version 55: Feature_dim 60. NCE_K 4096 * 2. Adam. LR 3e-4. Batch size 4096
    # version 56: Feature_dim 50. NCE_K 4096 * 2. Adam. LR 3e-4. Batch size 4096. Two GPUs
    # version 57: Replicate of version 50

    # version 61: SNARE dataset
    # version 62: Batch size 512
    # version 63: Lower feature_dim: 20. Train for longer. Batch size 32.
    # version 64: Batch size 8192
    # version 65: Batch size 128
    # version 66: Train 2 epoch only
    # version 67: Change model architecture
    # version 68: Smaller nce_k: 1024.
    # version 69: Larger feature dim. Change model architecture
    # version 70: Train for only 3 epochs.

    # version 71: (800, 400, 200), feature_dim: 400. 3 Epochs.
    # version 72: 40 Epochs.
    # 73: (800, 400, 400), feature_dim: 800. 3 Epochs.
    # 74: (800. 400, 200), feature_dim: 200. 3 Epochs
    # 75: (800. 400, 200), feature_dim: 50. 3 Epochs
    # 76: (800, 400, 200), feature_dim: 3. 3 Epochs.
    # 77: (800, 400, 200), feature_dim: 3. 30 Epochs.

    from rich.traceback import install
    install()

    pl.seed_everything(42)
    DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/babel/snareseq_GSE126074/"
    BATCH_SIZE = 128

    DIMS = (800, 400, 200)
    FEATURE_DIM = 3
    LR = 3e-4
    NCE_K = 1024
    NCE_T = 0.1
    NCE_M = 0.5
    N_NEIGHBORS = 40
    MAX_EPOCHS = 30

    CHECKPOINT = ""

    datamodule = SNAREDataModule(DATA_DIR, BATCH_SIZE)

    model = Net(
        in_dim=datamodule.dims[0],
        dims=DIMS,
        feature_dim=FEATURE_DIM,
        lr=LR,
        nce_k=NCE_K,
        nce_t=NCE_T,
        nce_m=NCE_M,
        length_train=datamodule.length_train,
        batch_size=BATCH_SIZE,
        n_neighbors=N_NEIGHBORS,
    )

    # callbacks = [
    #     KNNOnlineEvaluator(
    #         length_train=datamodule.length_train,
    #         n_neighbors=N_NEIGHBORS,
    #         num_classes=datamodule.num_classes,
    #     ),
    #     ModelCheckpoint(monitor='top1_acc', save_top_k=1, mode='max')
    # ]

    if not CHECKPOINT:
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            gpus=[2],
            deterministic=True,
            # callbacks=callbacks,
        )
        trainer.fit(model, datamodule)
    else:
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            gpus=[8],
            deterministic=True,
            # callbacks=callbacks,
            resume_from_checkpoint=CHECKPOINT,
        )
        trainer.fit(model, datamodule)
