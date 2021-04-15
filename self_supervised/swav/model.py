import os
import math

import torch
from torch import nn
from torch import distributed as dist

import pytorch_lightning as pl
from torch._C import device

from backbone import resnet18, resnet50


class SwAV(pl.LightningModule):
    def __init__(
        self,
        total_gpus,
        backbone,
        num_samples,
        batch_size,
        hidden_mlp,
        feature_dim,
        num_prototypes,
        first_conv,
        maxpool1,
        warmup_epochs,
        max_epochs,
        freeze_prototype_epochs,
        temperature,
        sinkhorn_iteration,
        crops_for_assign,
        crop_num,
        start_lr,
        lr,
        final_lr,
        weight_decay,
        epsilon,
        queue_length,
        queue_path,
        queue_start_epoch,
    ):
        """
        Args:
            crops_for_assign: list of crop ids for computing assignment
            crop_num: number of global and local crops, ex: [2, 6]

            queue_length: set queue when batch size is small,
                must be divisible by total batch-size (i.e. total_gpus * batch_size),
                set to 0 to remove the queue
            queue_path: folder within the logs directory
            queue_start_epoch: start uing the queue after this epoch
        """

        super().__init__()

        assert not queue_length % (batch_size * total_gpus)
        self.queue = None
        self.queue_length = queue_length
        self.queue_path = queue_path
        self.queue_start_epoch = queue_start_epoch

        self.save_hyperparameters()
        self.lr_schedule = self.get_linear_warmup_cosine_annealing_scheduler()
        self.get_assignment = (
            self.distributed_sinkhorn if total_gpus > 1 else self.sinkhorn
        )
        if backbone == "resnet18":
            backbone = resnet18
        elif backbone == "resnet50":
            backbone = resnet50
        else:
            raise ValueError("Backbone unknown.")
        self.model = backbone(
            hidden_mlp, feature_dim, num_prototypes, first_conv, maxpool1
        )

    def setup(self, stage):
        if self.queue_length > 0:
            queue_folder = os.path.join(self.logger.log_dir, self.queue_path)
            if not os.path.exists(queue_folder):
                os.makedirs(queue_folder)
            self.queue_path = os.path.join(
                queue_folder, "queue" + str(self.trainer.global_rank) + ".pth"
            )
            if os.path.isfile(self.queue_path):
                self.queue = torch.load(self.queue_path)["queue"]

    def forward(self, x):
        return self.model.backbone_forward(x)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val loss", loss)

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(
                self.named_parameters(), weight_decay=self.weight_decay
            )
        else:
            params = self.parameters()
        if self.optim == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )
        return optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]
        self.log(
            "lr",
            self.lr_schedule[self.trainer.global_step],
            on_step=True,
            on_epoch=False,
        )

        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu,
            using_native_amp,
            using_lbfgs,
        )

    def on_after_backward(self) -> None:
        if self.current_epoch < self.hparams.freeze_prototype_epochs:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
                    break
            else:
                raise RuntimeError("No prototype tensor found in your model.")

    def on_train_epoch_start(self):
        if self.queue_length > 0:
            if self.current_epoch == self.queue_start_epoch and self.queue is None:
                # Init the queue
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length // (self.hparams.total_gpus),
                    self.hparams.feature_dim,
                    device=self.device,
                )

    def on_train_epoch_end(self, outputs) -> None:
        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)

    def shared_step(self, batch):
        # Only use concatenate the queue when it is filled.
        queue_use = any(self.queue[:, -1, :] != 0)
        if self.dataset == "stl10":
            batch = batch[0]  # unlabeled data
        images, _ = batch[:-1]
        batch_size = images[0].shape[0]

        # Step 1: normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # Step 2: multi forward pass
        embedding, outputs = self.model(images)
        embedding = embedding.detach()

        # Step 3: SwAV loss
        loss = 0
        for i, crop_id in enumerate(self.hparams.crops_for_assign):
            with torch.no_grad():
                out = outputs[batch_size * crop_id : batch_size * (crop_id + 1)]

                # Step 4: use the queue
                if self.queue is not None:
                    if queue_use:
                        out = torch.cat(
                            (self.queue[i] @ self.model.prototypes.weight.t(), out)
                        )
                    # fill the queue
                    self.queue[i, batch_size:] = self.queue[i, :-batch_size].clone()
                    self.queue[i, :batch_size] = embedding[
                        crop_id * batch_size : (crop_id + 1) * batch_size
                    ]

                # Step 5: get assignments
                q = torch.exp(out / self.epsilon).t()
                q = self.get_assignments(q, self.hparams.sinkhorn_iterations)[
                    -batch_size:
                ]
            sub_loss = 0
            for v in torch.arange(sum(self.hparams.crop_num)):
                if v == crop_id:
                    continue
                p = (
                    outputs[batch_size * v, batch_size * (v + 1)]
                    / self.hparams.temperature
                ).softmax(dim=1)
                sub_loss -= (q * torch.log(p)).sum(dim=1).mean()
            loss += sub_loss / (sum(self.hparams.crop_num) - 1)
        return loss / len(self.hparams.crops_for_assign)

    def exclude_from_wt_decay(self, named_params, weight_decay):
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

    def get_linear_warmup_cosine_annealing_scheduler(self):
        global_batch_size = self.hparams.total_gpus * self.hparams.batch_size
        step_per_epoch = torch.ceil(self.hparams.num_samples // global_batch_size)

        warmup_lrs = torch.linspace(
            self.hparams.start_lr, self.hparams.lr, self.warmup_epochs * step_per_epoch
        )

        cosine_steps = step_per_epoch * (
            self.hparams.max_epochs - self.hparams.warmup_epochs
        )
        # Angular frequency of cosine function. Thus the period equals 2 * cosine_steps.
        w = math.pi / cosine_steps
        # Amplitude of cosine function
        a = 0.5 * (self.hparams.lr - self.hparams.final_lr)
        cosine_lrs = [
            0.5 * (self.hparams.lr + self.hparams.final_lr) + a * torch.cos(w * step)
            for step in range(cosine_steps)
        ]
        return torch.cat([warmup_lrs, cosine_lrs])

    def sinkhorn(self, Q, num_iters):
        with torch.no_grad():
            Q /= Q.sum()
            K, B = Q.shape
            u = torch.zeros(K, device=self.device)
            r = torch.full(K, 1 / K, device=self.device)
            c = torch.full(B, 1 / B, device=self.device)
            for _ in range(num_iters):
                u = Q.sum(dim=1)
                Q *= (r / u).unsqueeze(dim=1)
                Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
            return Q / Q.sum(dim=0, keepdim=True).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_ = Q.sum()
            sum_0 = Q.sum(dim=0)
            sum_1 = Q.sum(dim=1)
            dist.all_reduce(sum_)
            dist.all_reduce(sum_0)
            dist.all_reduce(sum_1)
            Q /= sum_
            K, B = Q.shape
            u = torch.zeros(K).cuda(non_blocking=True)
            r = torch.ones(K).cuda(non_blocking=True) / K
            c = torch.ones(B).cuda(non_blocking=True) / (self.hparams.total_gpus * B)
            for _ in range(nmb_iters):
                Q *= (r / sum_1).unsqueeze(1)
                Q *= (c / sum_0).unsqueeze(0)
            return (Q / sum_0.unsqueeze(0)).t().float()
