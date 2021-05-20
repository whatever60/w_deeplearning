import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms as TF


class LitModel(pl.LightningModule):
    def __init__(
        self,
        channels,
        height,
        width,
        num_classes,
        hidden_size=64,
        learning_rate=2e-4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return dict(loss=loss)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        return dict(val_loss=loss, val_acc=acc)


class MNISTDataModule(pl.LightningDataModule):
    '''A DataModule defines 5 methods:
    `prepare_data`
    `setup`
    `train_dataloader`
    `val_dataloader`
    `test_dataloader`
    '''
    def __init__(self, data_dir='./', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.num_classes = 10

        mean, std = (0.1307,), (0.3081,)
        self.transform = TF.Compose([
            TF.ToTensor(),
            TF.Normalize(mean, std)
        ])

    def prepare_data(self):
        '''download, tokenize, etc
        This is called from one single GPU, so do not use it to assign state 
        '''
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        '''This is called from every GPU, so it is ok to set state here,
        '''
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            self.dims = self.mnist_train[0][0].shape  # this attribute is necessary

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.dims = getattr(self, 'dims', self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


if __name__ == '__main__':
    dm = MNISTDataModule(data_dir='/home/tiankang/wusuowei/dataset')
    print(dm.dims)
    model = LitModel(*dm.size(), dm.num_classes)
    trainer = pl.Trainer(max_epochs=1, gpus=[7], progress_bar_refresh_rate=20)
    trainer.fit(model, dm)
