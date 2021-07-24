# from transformers import AutoTokenizer, AutoModelForMaskedLM
import math
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics import Accuracy
import transformers

from models import EntityModel
from schedulers import LinearWarmupCosineAnnealingLR
import datamodules


class Net(pl.LightningModule):
    def __init__(
        self, model_dir, num_poses, num_tags, lr, batch_size, max_epochs, total_steps
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = EntityModel(model_dir, num_poses, num_tags)
        self.crierion = nn.CrossEntropyLoss(ignore_index=-100)
        self.acc = Accuracy()

    def forward(self, ids, mask, token_type_ids):
        logits_pos, logits_tag = self.model(ids, mask, token_type_ids)
        return logits_pos, logits_tag

    def shared_step(self, batch):
        ids, mask, token_type_ids, target_pos, target_tag = (
            batch["ids"],
            batch["mask"],
            batch["token_type_ids"],
            batch["target_pos"],
            batch["target_tag"],
        )
        logits_pos, logits_tag = self(ids, mask, token_type_ids)

        logits_pos = logits_pos.view(-1, self.hparams.num_poses)
        logits_tag = logits_tag.view(-1, self.hparams.num_tags)
        target_pos = target_pos.view(-1)
        target_tag = target_tag.view(-1)
        loss = self.crierion(logits_pos, target_pos) + self.crierion(
            logits_tag, target_tag
        )
        return logits_pos, target_pos, logits_tag, target_tag, loss

    def training_step(self, batch, batch_idx):
        logits_pos, target_pos, logits_tag, target_tag, loss = self.shared_step(batch)
        self.log("loss_train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits_pos, target_pos, logits_tag, target_tag, loss = self.shared_step(batch)
        mask = target_pos >= 0
        acc_pos = self.acc(logits_pos[mask], target_pos[mask])
        acc_tag = self.acc(logits_tag[mask], target_tag[mask])
        self.log_dict(dict(loss_val=loss, acc_pos_val=acc_pos, acc_tag_val=acc_tag))

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            dict(
                params=[
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                weight_decay=1e-3,
            ),
            dict(
                params=[
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                weight_decay=0.0,
            ),
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.hparams.lr)
        scheduler = dict(
            scheduler=LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=0, max_epochs=self.hparams.total_steps
            ),
            interval="step",
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    from rich.traceback import install

    install()
    pl.seed_everything(42)

    MAX_LENGTH = 128
    MODEL_DIR = "/home/tiankang/wusuowei/models/bert-base-uncased"
    # MODEL_PATH = 'model.bin'
    DATA_PATH = "/home/tiankang/wusuowei/data/kaggle/annotated_corpus_for_named_entity_recognition/ner_dataset.csv"
    CACHE_PATH = "./data/meta.bin"
    BATCH_SIZE = 32
    tokenizer = transformers.BertTokenizer.from_pretrained(
        MODEL_DIR, do_lower_case=True
    )
    datamodules.tokenizer = tokenizer
    sentences, poses, tags = datamodules.process_data(DATA_PATH, cache_path=CACHE_PATH)
    datamodule = datamodules.EntityDataModule(
        sentences, poses, tags, MAX_LENGTH, BATCH_SIZE
    )

    LR = 3e-5
    MAX_EPOCHS = 10
    TOTAL_STEPS = math.ceil(datamodule.length_train / BATCH_SIZE) * MAX_EPOCHS
    print(TOTAL_STEPS)

    model = Net(
        model_dir=MODEL_DIR,
        num_poses=datamodule.num_classes_pos,
        num_tags=datamodule.num_classes_tag,
        lr=LR,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        total_steps=TOTAL_STEPS,
    )

    CHECKPOINT = ""

    if not CHECKPOINT:
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="acc_pos_val", mode="max", save_last=True, save_top_k=1
            ),
        ]
        trainer = pl.Trainer(
            gpus=[1],
            max_epochs=MAX_EPOCHS,
            callbacks=callbacks,
            deterministic=True,
            # fast_dev_run=True,
        )
        trainer.fit(model, datamodule)


# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
