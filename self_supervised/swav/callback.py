"""
Adapted from:
    - https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/callbacks/ssl_online.py
    - https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/evaluator.py
"""

from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn, device, Tensor
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.optim import Optimizer
from torchmetrics.functional import accuracy


class SSLEvaluator(nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim=512, p=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        if hid_dim is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(in_dim, out_dim, bias=True),
                nn.Softmax(dim=1),
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(in_dim, hid_dim, bias=False),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(hid_dim, out_dim, bias=True),
                nn.Softmax(dim=1),
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """
    Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    Example::
        # your model must have 2 attributes
        model = Model()
        model.z_dim = ... # the representation dim
        model.num_classes = ... # the num of classes in the model
        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim,
            num_classes=model.num_classes,
            dataset='imagenet'
        )
    """

    def __init__(
        self,        
        dataset: str,
        z_dim: int = None,
        hid_dim: Optional[int] = None,
        num_classes: int = None,
        drop_p: float = 0.2,
    ):
        """
        Args:
            dataset: if stl10, need to get the labeled batch
            drop_p: Dropout probability
            hid_dim: Hidden dimension for the fine-tune MLP
            z_dim: Representation dimension
            num_classes: Number of classes
        """
        super().__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.drop_p = drop_p
        

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        pl_module.non_linear_evaluator = SSLEvaluator(
            in_dim=self.z_dim,
            out_dim=self.num_classes,
            p=self.drop_p,
            hid_dim=self.hid_dim,
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(pl_module.non_linear_evaluator.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def to_device(self, batch: Sequence, device: Union[str, device]) -> Tuple[Tensor, Tensor]:
        # get the labeled batch
        if self.dataset == 'stl10':
            labeled_batch = batch[1]
            batch = labeled_batch
        inputs, y = batch
        # last input is for online eval
        x = inputs[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        mlp_loss, train_acc = self.shared_step(batch, pl_module)
        # log metrics
        pl_module.log('online_train_loss', mlp_loss, on_step=True, on_epoch=False)
        pl_module.log('online_train_acc', train_acc, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        mlp_loss, val_acc = self.shared_step(batch, pl_module)
        # log metrics
        pl_module.log('online_val_loss', mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True)

    def shared_step(self, batch, pl_module):
        x, y = self.to_device(batch, pl_module.device)
        with torch.no_grad():
            representations = pl_module(x).detach().flatten(start_dim=1)
        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
        mlp_loss = self.criterion(mlp_preds, y)
        val_acc = accuracy(mlp_preds, y)
        return mlp_loss, val_acc
