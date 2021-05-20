from pytorch_lightning import Callback, LightningModule, Trainer

import torch
from torch.nn.functional import one_hot

from rich import print


class KNNOnlineEvaluator(Callback):  # pragma: no cover
    """
    Evaluates self-supervised K nearest neighbors.
    Example::
        # your model must have 1 attribute
        model = Model()
        model.num_classes = ... # the num of classes in the model
        online_eval = KNNOnlineEvaluator(
            num_classes=model.num_classes,
            dataset='imagenet'
        )
    """
    def __init__(self, length_train) -> None:
        super().__init__()
        self.labels_train = torch.zeros(length_train, dtype=int)
        
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.labels_train = self.labels_train.to(pl_module.device)
        for _, targets, indices in trainer.datamodule.train_dataloader():
            targets, indices = targets.to(pl_module.device), indices.to(pl_module.device)
            self.labels_train[indices] = targets

    def on_validation_start(self, trainer, pl_module: LightningModule) -> None:
        self.labels_train = self.labels_train.to(pl_module.device)
        self.labels_val = torch.zeros(trainer.datamodule.length_val, dtype=int, device=pl_module.device)
        self.output_val = torch.zeros(trainer.datamodule.length_val, pl_module.hparams.feature_dim, device=pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx) -> None:
        repres, labels_val, indices = outputs['repres'], outputs['labels_val'], outputs['indices']
        self.labels_val[indices] = labels_val
        self.output_val[indices] = repres

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        pl_module.eval()
        relax = 3
        K = 200
        dist = self.output_val @ pl_module.memory.t()  # [batch, length_train]
        similarity, indices = dist.topk(K, dim=1, largest=True, sorted=True)  # both [batch, K]
        candidates = self.labels_train[indices.view(-1)].view(-1, K)  # Is there better way of doing such indexing?
        retrieval_one_hot = one_hot(candidates, num_classes=trainer.datamodule.num_classes)  # [batch, K, C]
        logits = ((similarity / pl_module.hparams.nce_t).exp().unsqueeze(-1) * retrieval_one_hot).sum(dim=1) # [batch, K, 1] * [batch, K, C] -> [batch, C]
        _, preds = logits.topk(relax, dim=1, largest=True, sorted=True)  # [batch, 3]
        result = (preds == self.labels_val.unsqueeze(-1))
        top1_acc = result[:, 0].float().mean()
        top_relax_acc = result.any(dim=1).float().mean()
        pl_module.log_dict({'top1_acc': top1_acc, f'top{relax}_acc': top_relax_acc})
