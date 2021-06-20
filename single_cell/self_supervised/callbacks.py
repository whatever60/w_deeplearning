import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torchmetrics.functional import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils import knn


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

    def __init__(self, length_train, n_neighbors, num_classes, relax=3) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.relax = relax
        self.num_classes = num_classes
        self.labels_train = torch.zeros(length_train, dtype=int)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.labels_train = self.labels_train.to(pl_module.device)
        for _, targets, indices in trainer.datamodule.train_dataloader():
            targets, indices = targets.to(pl_module.device), indices.to(
                pl_module.device
            )
            self.labels_train[indices] = targets

    def on_validation_start(self, trainer, pl_module: LightningModule) -> None:
        self.labels_train = self.labels_train.to(pl_module.device)
        self.labels_val = torch.zeros(
            trainer.datamodule.length_val, dtype=int, device=pl_module.device
        )
        self.output_val = torch.zeros(
            trainer.datamodule.length_val,
            pl_module.hparams.feature_dim,
            device=pl_module.device,
        )

    def on_validation_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        repres, labels_val, indices = (
            outputs["repres"],
            outputs["labels_val"],
            outputs["indices"],
        )
        self.labels_val[indices] = labels_val
        self.output_val[indices] = repres

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        pl_module.eval()
        relax = 3
        preds = knn(
            num_classes=self.num_classes,
            memory=pl_module.model.memory,
            input_data=self.output_val,
            memory_labels=self.labels_train,
            n_neighbors=self.n_neighbors,
            nce_t=pl_module.hparams.nce_t,
            relax=self.relax,
        )
        result = preds == self.labels_val.unsqueeze(dim=-1)
        top_1_acc = result[:, 0].float().mean()
        top_relax_acc = result.any(dim=1).float().mean()
        pl_module.log_dict({"top1_acc": top_1_acc, f"top{relax}_acc": top_relax_acc})

        matrix = confusion_matrix(preds[:, 0], self.labels_val, self.num_classes).cpu()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(matrix, center=int(matrix.mean()), ax=ax)
        pl_module.logger.experiment.add_figure(
            "Confusion matrix", fig, pl_module.current_epoch
        )
