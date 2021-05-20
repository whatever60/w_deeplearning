"""
Adapted from
    - https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
Bugs to fix:
- STL10 augmentation
- weight_init, bn3 and so on.
"""

import os
import math

import torch
from torch import nn
from torch import distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from backbone import resnet18, resnet50, SwAV
from optimizer import LARS
from lr_scheduler import linear_warmup_decay
from dataloader import get_swav_transform, SwAVFinetuneTransform, STL10DataModule, CIFAR10DataModule
from callback import SSLOnlineEvaluator


def exclude_from_wt_decay(named_params, weight_decay):
    skip_list = ["bias", "bn"]
    params = []
    excluded_params = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)
    return [
        {"params": params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0.0},
    ]


class Net(pl.LightningModule):
    def __init__(
        self,
        # ---- Model -----
        backbone_arch,
        first_conv,
        maxpool1,
        hidden_mlp,
        feature_dim,
        num_prototypes,
        # ----- Optimizer/schedular -----
        optimizer,
        lr,
        exclude_bn_bias,
        weight_decay,
        warmup_epochs,
        max_epochs,
        # start_lr,
        # final_lr,
        # ----- Queue -----
        num_samples,
        queue_length,
        queue_path,
        queue_start_epoch,
        # ----- Miscellaneous -----
        dataset,
        total_gpus,
        batch_size,
        crops_for_assign,
        num_crops,
        sinkhorn_iterations,
        epsilon,
        temperature,
        freeze_prototypes_epochs,
    ):
        """
        Args:
            crops_for_assign: list of crop ids for computing assignment
            num_crops: number of global and local crops, ex: [2, 6]

            queue_length: set queue when batch size is small,
                must be divisible by total batch-size (i.e. total_gpus * batch_size),
                set to 0 to remove the queue
            queue_path: folder within the logs directory
            queue_start_epoch: start uing the queue after this epoch
        """

        super().__init__()
        self.total_batch_size = batch_size * total_gpus
        # self.queue_length = queue_length
        # self.queue_path = queue_path
        # self.queue_start_epoch = queue_start_epoch

        self.save_hyperparameters()

        if backbone_arch == "resnet18":
            backbone = resnet18
        elif backbone_arch == "resnet50":
            backbone = resnet50
        else:
            raise ValueError("Backbone unknown.")
        self.model = SwAV(
            backbone=backbone(first_conv=first_conv, maxpool1=maxpool1),
            hid_dim=hidden_mlp,
            out_dim=feature_dim,
            normalize=True,
            num_prototypes=num_prototypes,
        )

        self.get_assignments = (
            self.distributed_sinkhorn if total_gpus > 1 else self.sinkhorn
        )

    def setup(self, stage):
        assert self.hparams.queue_length % self.total_batch_size == 0
        self.queue = None
        if self.hparams.queue_length > 0:
            # if not os.path.exists(queue_folder):
            #     os.makedirs(queue_folder)
            self.hparams.queue_path = os.path.join(
                self.logger.log_dir,
                self.hparams.queue_path,
                "queue" + str(self.trainer.global_rank) + ".pth",
            )
            if os.path.isfile(self.hparams.queue_path):
                self.queue = torch.load(self.hparams.queue_path)["queue"]
            self.queue_use = False

    def forward(self, x):
        return self.model.backbone(x)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        steps_per_epoch = math.ceil(
            self.hparams.num_samples // self.total_batch_size
        )
        if self.hparams.exclude_bn_bias:
            params = exclude_from_wt_decay(
                self.named_parameters(), weight_decay=self.hparams.weight_decay
            )
        else:
            params = self.parameters()

        if self.hparams.optimizer == "lars":
            optimizer = LARS(
                params,
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )

        warmup_steps = steps_per_epoch * self.hparams.warmup_epochs
        total_steps = steps_per_epoch * self.hparams.max_epochs

        return dict(
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            interval="step",
            frequency=1,
        )

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu,
    #     using_native_amp,
    #     using_lbfgs,
    # ):
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = self.lr_schedule[self.trainer.global_step]
    #     self.log(
    #         "lr",
    #         self.lr_schedule[self.trainer.global_step],
    #         on_step=True,
    #         on_epoch=False,
    #     )

    #     super().optimizer_step(
    #         epoch,
    #         batch_idx,
    #         optimizer,
    #         optimizer_idx,
    #         optimizer_closure,
    #         on_tpu,
    #         using_native_amp,
    #         using_lbfgs,
    #     )

    def on_after_backward(self) -> None:
        if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
                    break
            else:
                raise RuntimeError("No prototype tensor found in your model.")

    def on_train_epoch_start(self):
        if self.hparams.queue_length > 0:
            if self.current_epoch == self.hparams.queue_start_epoch and self.queue is None:
                # Init the queue
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.hparams.queue_length // (self.hparams.total_gpus),
                    self.hparams.feature_dim,
                )
                if self.hparams.total_gpus > 0:
                    self.queue = self.queue.cuda()

    def on_train_epoch_end(self, outputs) -> None:
        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)

    def shared_step(self, batch):
        if self.hparams.dataset == "stl10":
            batch = batch[0]  # unlabeled data
        images, _ = batch
        images = images[:-1]
        
        # Step 1: normalize the prototypes
        with torch.no_grad():
            w = self.model.head.prototype.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.head.prototype.weight.copy_(w)

        # Step 2: multi forward pass
        embedding, outputs = self.model(images)
        embedding = embedding.detach()

        # Step 3: SwAV loss
        batch_size = self.hparams.batch_size
        loss = 0
        for i, crop_id in enumerate(self.hparams.crops_for_assign):
            with torch.no_grad():
                out = outputs[batch_size * crop_id : batch_size * (crop_id + 1)]

                # Step 4: use the queue
                if self.queue is not None:
                    if not self.queue_use:
                        if not torch.all(self.queue[i, -1, :] == 0):
                            # Only concatenate the queue when it is filled.
                            self.queue_use = True
                    if self.queue_use:
                        out = torch.cat(
                            (self.queue[i] @ self.model.head.prototype.weight.t(), out)
                        )
                    # fill the queue
                    self.queue[i, batch_size:] = self.queue[i, :-batch_size].clone()
                    self.queue[i, :batch_size] = embedding[
                        crop_id * batch_size : (crop_id + 1) * batch_size
                    ]

                # Step 5: get assignments
                q = torch.exp(out / self.hparams.epsilon).t()
                q = self.get_assignments(q, self.hparams.sinkhorn_iterations)[
                    -batch_size:
                ]
            sub_loss = 0
            for v in torch.arange(sum(self.hparams.num_crops)):
                if v == crop_id:
                    continue
                p = (
                    outputs[batch_size * v: batch_size * (v + 1)]
                    / self.hparams.temperature
                ).softmax(dim=1)
                sub_loss -= (q * torch.log(p)).sum(dim=1).mean()
            loss += sub_loss / (sum(self.hparams.num_crops) - 1)
        return loss / len(self.hparams.crops_for_assign)

    # def get_linear_warmup_cosine_annealing_scheduler(self):
    #     global_batch_size = self.hparams.total_gpus * self.hparams.batch_size
    #     step_per_epoch = torch.ceil(self.hparams.num_samples // global_batch_size)

    #     warmup_lrs = torch.linspace(
    #         self.hparams.start_lr, self.hparams.lr, self.warmup_epochs * step_per_epoch
    #     )

    #     cosine_steps = step_per_epoch * (
    #         self.hparams.max_epochs - self.hparams.warmup_epochs
    #     )
    #     # Angular frequency of cosine function. Thus the period equals 2 * cosine_steps.
    #     w = math.pi / cosine_steps
    #     # Amplitude of cosine function
    #     a = 0.5 * (self.hparams.lr - self.hparams.final_lr)
    #     cosine_lrs = [
    #         0.5 * (self.hparams.lr + self.hparams.final_lr) + a * torch.cos(w * step)
    #         for step in range(cosine_steps)
    #     ]
    #     return torch.cat([warmup_lrs, cosine_lrs])

    def sinkhorn(self, Q, num_iters):
        with torch.no_grad():
            Q /= Q.sum()
            K, B = Q.shape
            u = torch.zeros(K, device=self.device)
            r = torch.ones(K, device=self.device) / K
            c = torch.ones(B, device=self.device) / B
            for _ in range(num_iters):
                u = Q.sum(dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / Q.sum(dim=0)).unsqueeze(0)
            return (Q / Q.sum(dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_ = Q.sum()
            dist.all_reduce(sum_)

            K, B = Q.shape
            r = torch.ones(K).cuda(non_blocking=True) / K
            c = torch.ones(B).cuda(non_blocking=True) / (self.hparams.total_gpus * B)

            for _ in range(nmb_iters):
                sum_1 = Q.sum(dim=1)
                dist.all_reduce(sum_1)
                Q *= (r / sum_1).unsqueeze(1)
                Q *= (c / Q.sum(dim=0)).unsqueeze(0)
            return (Q / Q.sum(dim=0, keepdim=True)).t().float()


if __name__ == "__main__":
    pl.seed_everything(42)
    # [32, 16] for CIFAR10, [96, 36] for STL10 [224, 96] for ImageNet
    SIZE_CROPS = [32, 16]
    CROPS_FOR_ASSIGN = [0, 1]
    NUM_CROPS = [2, 1]  # [2, 1] for CIFAR10, [2, 4] for STL10, [2, 6] for ImageNet.
    # [0.33, 0.10] for CIFAR10 & STL10, [0.14, 0.05] for ImageNet
    MIN_SCALE_CROPS = [0.33, 0.10]
    MAX_SCALE_CROPS = [1, 0.33]
    USE_GAUSSIAN_BLUR = True  # False for CIFAR10, True for ImageNet & STL10
    JITTER_STRENGTH = 1.0
    DATA_DIR = "/home/tiankang/wusuowei/data/"
    BATCH_SIZE = 128
    DATAMODULE = CIFAR10DataModule
    transforms = get_swav_transform(
        means=DATAMODULE.means,
        stds=DATAMODULE.stds,
        size_crops=SIZE_CROPS,
        num_crops=NUM_CROPS,
        min_scale_crops=MIN_SCALE_CROPS,
        max_scale_crops=MAX_SCALE_CROPS,
        use_gaussian_blur=USE_GAUSSIAN_BLUR,
        jitter_strength=JITTER_STRENGTH,
    )
    datamodule = DATAMODULE(DATA_DIR, BATCH_SIZE, *transforms)

    BACKBONE_ARCH = "resnet50"
    FIRST_CONV = False  # False for CIFAR10 & STL10, True for ImageNet
    MAXPOOL1 = False  # False for CIFAR10 & STL10, True for ImageNet
    HIDDEN_MLP = 2048
    FEATURE_DIM = 128
    NUM_PROTOTYPES = [512]

    OPTIMIZER = "adam"  # 'adam' for CIFAR10 & STL10, 'lars' for ImageNet
    LR = 1e-3
    WEIGHT_DECAY = 1e-6
    EXCLUDE_BN_BIAS = True
    WARMUP_EPOCHS = 10
    MAX_EPOCHS = 100
    MAX_STEPS = -1

    NUM_SAMPLES = 50000  # 95_000 for STL10 mixed dataloader
    QUEUE_PATH = "queue"
    QUEUE_LENGTH = 0
    QUEUE_START_EPOCH = 15

    GPUS = [9]
    TOTAL_GPUS = len(GPUS) * 1  # I only have one node, sadly
    SINKHORN_ITERATIONS = 3
    EPSILON = 0.05
    TEMPERTURE = 0.1
    FREEZE_PROTOTYPES_EPOCHS = 1

    model = Net(
        backbone_arch=BACKBONE_ARCH,
        first_conv=FIRST_CONV,
        maxpool1=MAXPOOL1,
        hidden_mlp=HIDDEN_MLP,
        feature_dim=FEATURE_DIM,
        num_prototypes=NUM_PROTOTYPES,
        optimizer=OPTIMIZER,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        exclude_bn_bias=EXCLUDE_BN_BIAS,
        warmup_epochs=WARMUP_EPOCHS,
        max_epochs=MAX_EPOCHS,
        num_samples=NUM_SAMPLES,
        queue_length=QUEUE_LENGTH,
        queue_path=QUEUE_PATH,
        queue_start_epoch=QUEUE_START_EPOCH,
        dataset=DATAMODULE.name,
        total_gpus=TOTAL_GPUS,
        batch_size=BATCH_SIZE,
        crops_for_assign=CROPS_FOR_ASSIGN,
        num_crops=NUM_CROPS,
        sinkhorn_iterations=SINKHORN_ITERATIONS,
        epsilon=EPSILON,
        temperature=TEMPERTURE,
        freeze_prototypes_epochs=FREEZE_PROTOTYPES_EPOCHS,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss"),
        SSLOnlineEvaluator(
            dataset=DATAMODULE.name,
            z_dim=model.model.backbone.out_dim,
            hid_dim=None,
            num_classes=DATAMODULE.num_classes,
            drop_p=0.0,
        ),
    ]
    CHECKPOINT = ""
    FINETUNE = False
    if not CHECKPOINT:
        trainer = pl.Trainer(
            deterministic=True,
            max_epochs=MAX_EPOCHS,
            max_steps=None if MAX_STEPS == -1 else MAX_STEPS,
            gpus=GPUS,
            num_nodes=1,
            distributed_backend="ddp" if TOTAL_GPUS > 1 else None,
            sync_batchnorm=True if TOTAL_GPUS > 1 else False,
            precision=16,
            callbacks=callbacks,
            # fast_dev_run=True,
        )

        trainer.fit(model, datamodule=datamodule)
    else:
        if not FINETUNE:
            trainer = pl.Trainer(
                deterministic=True,
                resume_from_checkpoint=CHECKPOINT,
                max_epochs=MAX_EPOCHS,
                max_steps=None if MAX_STEPS == -1 else MAX_STEPS,
                gpus=GPUS,
                num_nodes=1,
                distributed_backend="ddp" if TOTAL_GPUS > 1 else None,
                sync_batchnorm=True if TOTAL_GPUS > 1 else False,
                precision=32,
                callbacks=callbacks,
                # fast_dev_run=True,
            )
            trainer.fit(model, datamodule=datamodule)
        else:
            raise NotImplementedError
