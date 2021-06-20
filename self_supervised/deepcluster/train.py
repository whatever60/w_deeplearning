import math
import os

import torch
from torch import nn
from torch import distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from torchmetrics import Accuracy

from models import DeepClusterV2, resnet18, weight_init
from optimizers import LARS
from schedulers import LinearWarmupCosineAnnealingLR
from datamodules import MultiCropDataModule, get_aug
from utils import get_indices_sparse


class Net(pl.LightningModule):
    def __init__(
        self,
        # ---- model arch ----
        backbone,
        hid_dim: int,
        emb_dim: int,
        normalize: bool,
        # ---- DeepCluster params ----
        dataset_length: int,
        num_gpus: int,  # num_nodes * num_gpus
        num_prototypes: list,
        temperature: float,  # before softmax
        # ----- optim ----
        # - optimizer -
        base_lr: float,
        momentum: float,
        weight_decay: float,
        trust_coefficient: float,
        # - scheduler -
        warmup_start_lr: float,
        warmup_epochs: float,
        final_lr: float,
        # - other -
        freeze_prototypes_iters: int,
        max_epochs: int,
        batch_size: int,
        # ---- about crop ----
        # - for dataloader -
        num_crops: list,  # just for logging
        size_crops: list,  # just for logging
        min_scale_crops: list,  # just for logging
        max_scale_crops: list,  # just for logging
        # - for knn assignment -
        num_kmeans_iters: int,
        crops_for_assign: list,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepClusterV2(
            backbone=backbone(),
            hid_dim=hid_dim,
            out_dim=emb_dim,
            normalize=normalize,
            num_prototypes=num_prototypes,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.metric = Accuracy()
        self.model.apply(weight_init)

        if num_gpus == 1:
            self.cluster_memory = self.cluster_memory_single
            self.total_steps = max_epochs * math.ceil(dataset_length // batch_size)
            self.size_memory_per_process = dataset_length
            steps_per_epochs = math.ceil(dataset_length // batch_size)
        else:
            self.cluster_memory = self.cluster_memory_dist
            total_batch_size = batch_size * num_gpus
            steps_per_epochs = math.ceil(dataset_length // total_batch_size)
            self.total_steps = max_epochs * steps_per_epochs

            if dataset_length % total_batch_size:
                self.size_memory_per_process = dataset_length // num_gpus + batch_size
            else:
                self.size_memory_per_process = dataset_length // num_gpus
        self.hparams.warmup_steps = warmup_epochs * steps_per_epochs

    def on_train_start(self) -> None:
        self.local_memory_index, self.local_memory_embeddings = self.init_memory()

    def on_train_epoch_start(self) -> None:
        self._start_idx = 0
        if self.current_epoch > 0:
            self.assignments = self.cluster_memory()

    def configure_optimizers(self):
        # LARS optimizer
        optimizer = LARS(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            trust_coefficient=self.hparams.trust_coefficient,
        )
        lr_scheduler = dict(
            scheduler=LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_steps,
                warmup_start_lr=self.hparams.warmup_start_lr,
                eta_min=self.hparams.final_lr,
                max_epochs=self.total_steps,
            ),
            interval="step",
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        inputs, label, index = batch
        emb, outputs = self.model(inputs)
        emb = emb.detach()
        if self.current_epoch > 0:
            loss = 0
            acc = 0
            # the first epoch is used to fill the memory and do no perform backprop.
            for output, assignment in zip(outputs, self.assignments):
                scores = output / self.hparams.temperature
                targets = (
                    assignment[index]
                    .repeat(sum(self.hparams.num_crops))
                    .cuda(non_blocking=True)
                )
                loss += self.criterion(scores, targets)
                acc += self.metric(scores.argmax(dim=1), targets)
            loss /= len(outputs)
            acc /= len(outputs)
            self.log_dict(dict(loss_train=loss, acc_train=acc))
        else:
            loss = None

        crops_for_assign = self.hparams.crops_for_assign
        batch_size = inputs[0].shape[0]
        end_idx = self._start_idx + batch_size
        self.local_memory_index[self._start_idx : end_idx] = index
        # fill the memory bank
        for i, crop_idx in enumerate(crops_for_assign):
            self.local_memory_embeddings[i][self._start_idx : end_idx] = emb[
                crop_idx * batch_size : (crop_idx + 1) * batch_size
            ]
        self._start_idx += batch_size

        return loss

    def on_after_backward(self) -> None:
        if self.global_step < self.hparams.freeze_prototypes_iters:
            for prototype in self.model.prototypes:
                prototype.grad = None

    @torch.no_grad()
    def init_memory(self):
        emb_dim = self.hparams.emb_dim
        local_memory_index = torch.full(
            [self.size_memory_per_process], -1, dtype=int, device=self.device
        )
        local_memory_embeddings = torch.zeros(
            len(self.hparams.crops_for_assign),
            self.size_memory_per_process,
            emb_dim,
            device=self.device,
        )
        return local_memory_index, local_memory_embeddings

    @torch.no_grad()
    def cluster_memory_dist(self):
        num_prototypes = self.hparams.num_prototypes
        dataset_length = self.hparams.dataset_length
        emb_dim = self.hparams.emb_dim
        num_kmeans_iters = self.hparams.num_kmeans_iters
        world_size = os.environ["WORLD_SIZE"]
        crops_for_assign = self.hparams.crops_for_assign

        j = 0
        assignments = -100 * torch.ones(len(num_prototypes), dataset_length)
        for i, K in enumerate(num_prototypes):
            # distributed k-means

            # init centroids with elements from memory bank of rank 0
            centroids = torch.empty(K, emb_dim).cuda(non_blocking=True)
            if self.global_rank == 0:
                random_idx = torch.randperm(len(self.local_memory_embeddings[j]))[:K]
                assert len(random_idx) >= K
                centroids = self.local_memory_embeddings[j][random_idx]
            dist.broadcast(centroids, 0)

            for n_iter in range(num_kmeans_iters + 1):
                # E
                dot_products = self.local_memory_embeddings[j] @ centroids.t()
                _, local_assignments = dot_products.max(dim=1)
                if n_iter == num_kmeans_iters:
                    break

                # M
                where_helper = get_indices_sparse(local_assignments.cpu().numpy())
                counts = torch.zeros(K).cuda(non_blocking=True).int()
                emb_sums = torch.zeros(K, emb_dim).cuda(non_blocking=True)
                for k, idxs in enumerate(where_helper):
                    if len(idxs) > 0:
                        emb_sums[k] = self.local_memory_embeddings[j][idxs].sum(dim=0)
                        counts[k] = len(idxs)
                dist.all_reduce(emb_sums)
                dist.all_reduce(counts)
                mask = counts > 0
                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)
                centroids = nn.functional.normalize(centroids, dim=1, p=2)
            self.model.module.prototypes[i].weight.copy_(centroids)

            # gather the assignments
            assignments_all = torch.empty(
                world_size,
                local_assignments.shape[0],
                dtype=local_assignments.dtype,
                device=local_assignments.device,
            )
            assignments_all = list(assignments_all.unbind(0))
            dist_process = dist.all_gather(
                assignments_all, local_assignments, async_op=True
            )
            dist_process.wait()
            assignments_all = torch.cat(assignments_all).cpu()

            # gather the indexes
            indexes_all = torch.empty(
                world_size,
                self.local_memory_index.shape[0],
                dtype=self.local_memory_index.dtype,
                device=self.local_memory_index.device,
            )
            indexes_all = list(indexes_all.unbind(0))
            dist_process = dist.all_gather(
                indexes_all, self.local_memory_index, async_op=True
            )
            dist_process.wait()
            indexes_all = torch.cat(indexes_all).cpu()

            # log assignments
            assignments[i][indexes_all] = assignments_all

            # next memory bank to use
            j = (j + 1) % len(crops_for_assign)
        return assignments

    @torch.no_grad()
    def cluster_memory_single(self):
        num_prototypes = self.hparams.num_prototypes
        dataset_length = self.hparams.dataset_length
        emb_dim = self.hparams.emb_dim
        num_kmeans_iters = self.hparams.num_kmeans_iters
        crops_for_assign = self.hparams.crops_for_assign

        j = 0
        assignments = torch.full((len(num_prototypes), dataset_length), -100, dtype=int)
        for i, K in enumerate(num_prototypes):
            # distributed k-means
            index_mask = self.local_memory_index >= 0
            # init centroids with elements from memory bank of rank 0
            local_memory_embeddings = self.local_memory_embeddings[j][index_mask]
            centroids = torch.empty(K, emb_dim).cuda(non_blocking=True)

            random_idx = torch.randperm(len(local_memory_embeddings))[:K]
            assert len(random_idx) >= K
            centroids = local_memory_embeddings[random_idx]

            for n_iter in range(num_kmeans_iters + 1):
                # E
                dot_products = local_memory_embeddings @ centroids.t()
                _, local_assignments = dot_products.max(dim=1)
                if n_iter == num_kmeans_iters:
                    break

                # M
                where_helper = get_indices_sparse(local_assignments.cpu().numpy(), K)
                counts = torch.zeros(K).int().cuda(non_blocking=True)
                emb_sums = torch.zeros(K, emb_dim).cuda(non_blocking=True)
                for k, idxs in enumerate(where_helper):
                    if len(idxs) > 0:
                        emb_sums[k] = local_memory_embeddings[idxs].sum(dim=0)
                        counts[k] = len(idxs)
                mask = counts > 0
                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)
                centroids = nn.functional.normalize(centroids, dim=1, p=2)
            self.model.prototypes[i].weight.copy_(centroids)

            # gather the assignments
            assignments_all = local_assignments

            # gather the indexes
            indexes_all = self.local_memory_index

            # log assignments
            indexes_all = indexes_all[index_mask].cpu()
            assignments_all = assignments_all[index_mask].cpu()
            assignments[i][indexes_all] = assignments_all

            # next memory bank to use
            j = (j + 1) % len(crops_for_assign)
        return assignments


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()

    pl.seed_everything(42)

    DATAMODULE = MultiCropDataModule
    DATA_DIR = "~/wusuowei/data/imagenette/imagewoof2/train/"
    NUM_CROPS = [2, 6]  # list of number of crops (example: [2, 6])
    SIZE_CROPS = [224, 96]  # crops resolutions (example: [224, 96])
    MIN_SCALE_CROPS = [0.14, 0.05]  # argument in RandomResizedCrop, e.g. [0.14, 0.05]
    MAX_SCALE_CROPS = [1, 0.14]  # argument in RandomResizedCrop, e.g. [1., 0.14]
    BATCH_SIZE = 128

    BACKBONE = resnet18
    HID_DIM = None
    EMB_DIM = 128

    NUM_PROTOTYPES = [300, 300]  # number of prototypes - it can be multihead
    TEMPERATURE = 0.1

    BASE_LR = 2
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-6
    TRUST_COEFFICIENT = 1e-3
    WARMUP_START_LR = 0.3  # initial warmup learning rate
    WARMUP_EPOCHS = 10
    FINAL_LR = 4.8e-3
    FREEZE_PROTOTYPES_NITERS = 3e3

    NUM_KMEANS_ITERS = 10
    CROPS_FOR_ASSIGN = [0, 1]  # list of crops id used for computing assignments

    MAX_EPOCHS = 800
    GPUS = [1]

    multi_crop_transform = get_aug(
        DATAMODULE.means,
        DATAMODULE.stds,
        NUM_CROPS,
        SIZE_CROPS,
        MIN_SCALE_CROPS,
        MAX_SCALE_CROPS,
    )
    datamodule = DATAMODULE(DATA_DIR, BATCH_SIZE, multi_crop_transform, None)
    model = Net(
        #
        backbone=BACKBONE,
        hid_dim=HID_DIM,
        emb_dim=EMB_DIM,
        normalize=True,
        #
        dataset_length=DATAMODULE.length_train,
        num_gpus=len(GPUS),
        num_prototypes=NUM_PROTOTYPES,
        temperature=TEMPERATURE,
        #
        base_lr=BASE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        trust_coefficient=TRUST_COEFFICIENT,
        warmup_start_lr=WARMUP_START_LR,
        warmup_epochs=WARMUP_EPOCHS,
        final_lr=FINAL_LR,
        freeze_prototypes_iters=FREEZE_PROTOTYPES_NITERS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        #
        num_crops=NUM_CROPS,
        size_crops=SIZE_CROPS,
        min_scale_crops=MIN_SCALE_CROPS,
        max_scale_crops=MAX_SCALE_CROPS,
        #
        num_kmeans_iters=NUM_KMEANS_ITERS,
        crops_for_assign=CROPS_FOR_ASSIGN,
    )

    CHECKPOINT = ""
    if not CHECKPOINT:
        callbacks = [
            # EarlyStopping(
            #     monitor="acc_val",
            #     min_delta=0.00,
            #     patience=10,
            #     verbose=False,
            #     mode="max",
            # ),
            # ModelCheckpoint(monitor="acc_val", save_last=True, save_top_k=2),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        trainer = pl.Trainer(
            # fast_dev_run=True,
            gpus=GPUS,
            max_epochs=MAX_EPOCHS,
            deterministic=True,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule)
