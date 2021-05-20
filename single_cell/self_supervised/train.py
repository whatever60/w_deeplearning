import math
import torch
from torch import nn
from torch.nn.functional import normalize
import pytorch_lightning as pl

from datamodule import scRNADataModule
from loss import NCECriterion
from utils import LinearAverage, AliasMethod
from callbacks import KNNOnlineEvaluator

from rich.traceback import install
install()


class NPIDHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return normalize(self.model(x))


class MLP(nn.Module):
    def __init__(self, in_dim, dims=(800, 400, 200)):
        super().__init__()
        layers = []
        for out_dim in dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        self.out_dim = out_dim
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NPID(pl.LightningModule):
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
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.backbone = MLP(in_dim, dims)
        self.head = NPIDHead(self.backbone.out_dim, feature_dim)
        stdv = 1 / math.sqrt(feature_dim / 3)
        self.register_buffer(
            "memory", torch.randn(length_train, feature_dim) * 2 * stdv + stdv
        )
        if nce_k > 0:
            # use NCE Loss
            self.lemniscate = self.nce_average
            self.criterion = NCECriterion(length_train)
            self.multinomial = AliasMethod(torch.ones(length_train))
        else:
            self.lemniscate = LinearAverage(feature_dim, length_train, nce_t, nce_m)
            self.criterion = nn.CrossEntropyLoss()
            self.multinomial = None

    def nce_average(self, repres, pos_indices):
        # repres: [batch, out_dim]
        indices = self.multinomial.draw(repres.shape[0] * self.hparams.nce_k).view(
            repres.shape[0], self.hparams.nce_k
        )
        indices[:, 0] = pos_indices
        # [batch, nce_k, out_dim]
        weight = self.memory[indices].view(repres.shape[0], self.hparams.nce_k, -1)
        # similarities [batch, nce_k]
        sims = (torch.einsum("bki, bi -> bk", weight, repres) / self.hparams.nce_t).exp()
        sims = (
            sims / sims.sum(dim=1, keepdim=True)
            * (self.hparams.nce_k / self.hparams.length_train)
        )  # each sample add up to K / N
        return sims

    def forward(self, imgs):
        return self.head(self.backbone(imgs))

    def training_step(self, batch, batch_idx):
        imgs, _, indices = batch
        repres = self(imgs)
        sims = self.lemniscate(repres, indices)
        loss = self.criterion(sims, indices)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        scheduler = self.lr_schedulers()
        if self.trainer.is_last_batch:
            scheduler.step()
        self.log("train_loss", loss, prog_bar=True)
        with torch.no_grad():
            self.memory[indices] = normalize(
                self.memory[indices] * self.hparams.nce_m
                + repres.detach() * (1 - self.hparams.nce_m)
            )

    def validation_step(self, batch, batch_idx):
        imgs, labels_val, indices = batch
        repres = self(imgs).detach()
        return dict(repres=repres, labels_val=labels_val, indices=indices)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.head.parameters()),
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
    # version 22: 
    pl.seed_everything(42)
    DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/MARS/Tabula_Muris/"
    BATCH_SIZE = 32

    DIMS = (800, 400, 200)
    FEATURE_DIM = 50
    LR = 3e-4
    NCE_K = 4096
    NCE_T = 0.1
    NCE_M = 0.5
    N_NEIGHBORS = 40
    MAX_EPOCHS = 20

    datamodule = scRNADataModule(DATA_DIR, BATCH_SIZE)
    net = NPID(
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
    callback = KNNOnlineEvaluator(datamodule.length_train, n_neighbors=N_NEIGHBORS)
    CHECKPOINT = ""

    if not CHECKPOINT:
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS, gpus=[6], deterministic=True, callbacks=callback
        )
        trainer.fit(net, datamodule)
    else:
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            gpus=[6],
            deterministic=True,
            callbacks=callback,
            resume_from_checkpoint=CHECKPOINT,
        )
        trainer.fit(net, datamodule)
