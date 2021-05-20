"""
This part consists of
- SSLEvaluator: An linear projection head to evaluate the representation learned by a model.
- SSLFineTuner: A wrapper around SSLEvaluator, for fine-tuning.
- SSLOnlineEvaluator: Another wrapper around SSLEvaluator, but for online evaluation.
"""


from typing import List, Optional, Sequence, Tuple, Union
from torch import device, Tensor

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy


class SSLEvaluator(nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim=512, p=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        if hid_dim is None:
            # use single linear layer
            self.block_forward = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(in_dim, out_dim, bias=True),
                nn.Softmax(dim=1),
            )
        else:
            # use double linear layer
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


class SSLFineTuner(pl.LightningModule):
    """
    Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP
    with 1024 units
    Example::
        from pl_bolts.utils.self_supervised import SSLFineTuner
        from pl_bolts.models.self_supervised import CPC_v2
        from pl_bolts.datamodules import CIFAR10DataModule
        from pl_bolts.models.self_supervised.cpc.transforms import CPCEvalTransformsCIFAR10,
                                                                    CPCTrainTransformsCIFAR10
        # pretrained model
        backbone = CPC_v2.load_from_checkpoint(PATH, strict=False)
        # dataset + transforms
        dm = CIFAR10DataModule(data_dir='.')
        dm.train_transforms = CPCTrainTransformsCIFAR10()
        dm.val_transforms = CPCEvalTransformsCIFAR10()
        # finetuner
        finetuner = SSLFineTuner(backbone, in_features=backbone.z_dim, num_classes=backbone.num_classes)
        # train
        trainer = pl.Trainer()
        trainer.fit(finetuner, dm)
        # test
        trainer.test(datamodule=dm)
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int = 2048,
        num_classes: int = 1000,
        epochs: int = 100,
        hid_dim: Optional[int] = None,
        dropout: float = 0.,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = 'cosine',
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.
    ):
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hid_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr

        self.backbone = backbone
        self.linear_layer = SSLEvaluator(in_dim=in_features, hid_dim=hid_dim, out_dim=num_classes, p=dropout)
        self.criterion = nn.CrossEntropyLoss()
        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc_step', acc, prog_bar=True)
        self.log('train_acc_epoch', self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.val_acc(logits, y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc)

        return loss

    # def test_step(self, batch, batch_idx):
    #     loss, logits, y = self.shared_step(batch)
    #     self.test_acc(logits, y)

    #     self.log('test_loss', loss, sync_dist=True)
    #     self.log('test_acc', self.test_acc)

    #     return loss

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.backbone(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = self.criterion(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            nesterov=self.nesterov,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        # if self.scheduler_type == "step":
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        # elif self.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.epochs,
            eta_min=self.final_lr  # total epochs to run
        )

        return [optimizer], [scheduler]


class SSLOnlineEvaluator(pl.Callback):  # pragma: no cover
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
        

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

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
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
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
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
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

