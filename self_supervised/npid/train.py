import math
import torch
from torch import nn
from torch.nn.functional import normalize
from torchvision import transforms as T
import pytorch_lightning as pl

from resnet import resnet18
from cifar10 import CIFAR10DataModule
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


class NPID(pl.LightningModule):
    def __init__(
        self,
        feature_dim,
        lr,
        nce_k,  # number of negative samples
        nce_t,  # temperature
        nce_m,  # momentum for memory update
        length_train,  # number of training data, also number of classes.
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.backbone = resnet18(first_conv=False, maxpool1=False)
        self.head = NPIDHead(self.backbone.out_dim, feature_dim)
        stdv = 1 / math.sqrt(feature_dim / 3)
        self.register_buffer('memory', torch.randn(length_train, feature_dim) * 2 * stdv + stdv)
        if nce_k > 0:
            # use NCE Loss
            # self.lemniscate = NCEAverage(feature_dim, length_train, nce_k, nce_t, nce_m)
            self.lemniscate = self.nce_average
            self.criterion = NCECriterion(length_train)
            self.multinomial = AliasMethod(torch.ones(length_train))
        else:
            self.lemniscate = LinearAverage(feature_dim, length_train, nce_t, nce_m)
            self.criterion = nn.CrossEntropyLoss()
            self.multinomial = None
    
    def nce_average(self, repres, pos_indices):
        # repres: [batch, feature_dim]
        indices = self.multinomial.draw(repres.shape[0] * self.hparams.nce_k).view(repres.shape[0], self.hparams.nce_k)
        indices[:, 0] = pos_indices
        weight = self.memory[indices].view(repres.shape[0], self.hparams.nce_k, -1)  # [batch, nce_k, feature_dim]
        sims = (torch.einsum("bki, bi -> bk", weight, repres) / self.hparams.nce_t).exp()  # similarities [batch, nce_k]
        sims = sims / sims.sum(dim=1, keepdim=True) * self.hparams.nce_k / self.hparams.length_train  # each sample add up to K / N
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
        optimizer = torch.optim.SGD(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=self.hparams.lr
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[80, 120, 160], gamma=0.1
        )
        return [optimizer], [scheduler]


def get_datamodule(data_dir, batch_size, datamodule):
    transform_train = T.Compose(
        [
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
            T.RandomGrayscale(p=0.2),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(datamodule.means, datamodule.stds),
        ]
    )

    transform_val = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(datamodule.means, datamodule.stds),
        ]
    )
    return datamodule(data_dir, batch_size, transform_train, transform_val)


if __name__ == "__main__":
    pl.seed_everything(42)
    DATA_DIR = "~/wusuowei/data"
    BATCH_SIZE = 128
    DATAMODULE = CIFAR10DataModule

    LR = 3e-2
    FEATURE_DIM = 128
    NCE_K = 4096
    NCE_T = 0.1
    NCE_M = 0.5

    MAX_EPOCHS = 200

    datamodule = get_datamodule(DATA_DIR, BATCH_SIZE, CIFAR10DataModule)
    net = NPID(
        feature_dim=FEATURE_DIM,
        lr=LR,
        nce_k=NCE_K,
        nce_t=NCE_T,
        nce_m=NCE_M,
        length_train=datamodule.length_train
    )
    callback = KNNOnlineEvaluator(datamodule.length_train)
    CHECKPOINT = "" # "/home/tiankang/wusuowei/deeplearning/self_supervised/npid/lightning_logs/version_22/checkpoints/epoch=199-step=62599.ckpt"

    if not CHECKPOINT:
        trainer = pl.Trainer(
            max_epochs=200, gpus=[9], deterministic=True, callbacks=callback
        )
        trainer.fit(net, datamodule)
