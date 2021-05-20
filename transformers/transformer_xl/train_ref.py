import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from model_ref import TransformerXL
from scheduler import LinearWarmupCosineAnnealingLR
from datamodule import WikiText2DataModule

from rich import print as rprint
from rich.traceback import install

install()


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.normal_(m.weight, 1.0, 0.02)
    elif hasattr(m, "u"):
        nn.init.normal_(m.u, std=0.02)
    elif hasattr(m, "v"):
        nn.init.normal_(m.v, std=0.02)


class Net(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        num_layers,
        emb_dim,
        head_dim,
        num_heads,
        p_ff,
        p_att,
        memory_length_train,
        memory_length_val,
        bptt_train,
        bptt_val,
        lr,
        warmup_steps,
        max_steps,
        eta_min,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerXL(
            num_embeddings=vocab_size,
            n_layers=num_layers,
            d_model=emb_dim,
            d_head_inner=head_dim,
            n_heads=num_heads,
            dropout=p_ff,
            dropouta=p_att,
            mem_len=memory_length_train,
            seq_len=0,
            d_ff_inner=4 * emb_dim,
        )
        self.model.apply(init_weight)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_idxs):
        return self.model(input_idxs)

    def shared_step(self, batch):
        input_idxs, target_idxs = batch
        input_idxs, target_idxs = (
            input_idxs.t(),
            target_idxs.t(),
        )  # both: [seq_length, batch_size]
        logits = self(input_idxs)  # [seq_length, batch_size, vocab_size]
        seq_length = input_idxs.shape[0]
        loss = self.criterion(logits.flatten(end_dim=1), target_idxs.flatten())
        return loss, seq_length

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)
        self.log_dict(dict(loss_train=loss, pp_train=torch.exp(loss)))
        return loss

    def training_epoch_end(self, outputs):
        self.model.memory = None  # empty memory
        self.model.memory_length = (
            self.hparams.memory_length_val
            + self.hparams.bptt_train
            - self.hparams.bptt_val
        )

    def validation_step(self, batch, batch_idx):
        loss, seq_length = self.shared_step(batch)
        self.log("loss_val", loss)
        return loss, seq_length

    def validation_epoch_end(self, outputs):
        self.model.memory = None  # empty memory
        self.model.memory_length = self.hparams.memory_length_train
        losses, seq_lengths = zip(*outputs)
        total_loss = sum([loss * seq_length for loss, seq_length in outputs])
        total_len = sum(seq_lengths)
        self.log("pp_val", torch.exp(total_loss / total_len))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = dict(
            scheduler=LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=self.hparams.warmup_steps,
                max_epochs=self.hparams.max_steps,
                eta_min=self.hparams.eta_min,
            ),
            interval="step",
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    pl.seed_everything(42)

    DATA_DIR = "./data/tokenized.pkl"
    BATCH_SIZE = 128

    NUM_LAYERS = 4
    EMB_DIM = 32
    HEAD_DIM = 16
    NUM_HEADS = 3
    P_FF = 0.1
    P_ATT = 0
    MEMORY_LENGTH_TRAIN = 128
    MEMORY_LENGTH_VAL = 128
    BPTT_TRAIN = 64
    BPTT_VAL = 64

    LR = 2.5e-4
    WARMUP_STEPS = 1
    MAX_STEPS = 400_000
    ETA_MIN = 0
    CLIP = 0.25

    datamodule = WikiText2DataModule(DATA_DIR, BATCH_SIZE, BPTT_TRAIN, BPTT_VAL)

    model = Net(
        vocab_size=datamodule.vocab_size,
        num_layers=NUM_LAYERS,
        emb_dim=EMB_DIM,
        head_dim=HEAD_DIM,
        num_heads=NUM_HEADS,
        p_ff=P_FF,
        p_att=P_ATT,
        memory_length_train=MEMORY_LENGTH_TRAIN,
        memory_length_val=MEMORY_LENGTH_VAL,
        bptt_train=BPTT_TRAIN,
        bptt_val=BPTT_VAL,
        lr=LR,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        eta_min=ETA_MIN,
    )
    callback = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        gpus=[9],
        max_steps=MAX_STEPS,
        deterministic=True,
        gradient_clip_val=CLIP,
        reload_dataloaders_every_epoch=True,
        callbacks=callback,
    )

    CHECKPOINT = ""
    if not CHECKPOINT:
        trainer.fit(model, datamodule)
