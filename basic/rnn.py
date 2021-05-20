import torch
from torch import nn
from torchvision import transforms as T
import pytorch_lightning as pl

from mnist import MNISTDataModule
from cifar10 import CIFAR10DataModule


class RNN(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layers,
        num_classes,
        lr,
        module=nn.RNN,
        bidirectional=False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams.num_directions = 2 if bidirectional else 1
        self.backbone = module(
            in_dim, hid_dim, num_layers, batch_first=True, bidirectional=bidirectional
        )
        self.head = nn.Linear(hid_dim * self.hparams.num_directions, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        # x: [batch, seq_len, in_dim]
        h0 = torch.zeros(
            self.hparams.num_layers * self.hparams.num_directions,
            x.shape[0],
            self.hparams.hid_dim,
            device=self.device,
        )
        # h0: [num_layers * num_directions, batch, hid_dim]. Hidden state
        if isinstance(self.backbone, nn.LSTM):
            c0 = torch.zeros_like(h0)  # cell state
            output, (hn, cn) = self.backbone(x, (h0, c0))
        else:
            output, hn = self.backbone(x, h0)
        # output: [batch, seq_len, hid_dim * num_directions], hn: [num_layers * num_directions, seq_len, hid_dim]
        preds = self.head(
            output[:, -1, :]
        )  # only use the last hidden state of the last layer
        return preds

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.lr)


def get_transforms(datamodule, in_dim, seq_len):
    def reshape(x: torch.Tensor):
        return x.reshape(seq_len, in_dim)

    transform = T.Compose(
        [T.ToTensor(), T.Normalize(datamodule.means, datamodule.stds), reshape]
    )
    return transform


if __name__ == "__main__":
    # Custom RNN
    # Elman RNN

    pl.seed_everything(42)
    DATA_DIR = "~/wusuowei/data"
    BATCH_SIZE = 100
    LR = 1e-3
    IN_DIM = 32 * 3  # 28 for MNIST
    SEQ_LEN = 32  # 28 for MNIST
    HID_DIM = 128
    NUM_LAYERS = 2
    transform = get_transforms(CIFAR10DataModule, IN_DIM, SEQ_LEN)
    datamodule = CIFAR10DataModule(DATA_DIR, BATCH_SIZE, transform, transform)
    net = RNN(
        IN_DIM,
        HID_DIM,
        NUM_LAYERS,
        datamodule.num_classes,
        LR,
        module=nn.LSTM,
        bidirectional=True,
    )
    trainer = pl.Trainer(gpus=[9], max_epochs=100, deterministic=True)
    trainer.fit(net, datamodule)
