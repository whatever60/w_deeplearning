"""
Adapted from
- https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/transformer_xl_from_scratch.ipynb
"""

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from models import TransformerXL, GPT2, LSTMLM
from scheduler import LinearWarmupCosineAnnealingLR
from datamodule import WikiText2DataModule


def init_weight_trm(m):
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


def init_weight_lstm(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight)


class Net(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        vocab_size: int,
        num_layers: int,
        emb_dim: int,
        head_dim: int,
        hid_dim: int,  # LSTM only
        mog_iters: int,  # LSTM only
        num_heads: int,
        p_ff: float,
        p_att: float,
        memory_length_train: int,
        memory_length_val: int,
        bptt_train: int,
        bptt_val: int,
        lr: float,
        warmup_steps: int,
        max_steps: int,
        eta_min: float,
        batch_size: int,  # just for hyperparameter logging.
        clip: float,  # just for hyperparameter logging as well.
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if backbone_name == "transformer_xl":
            self.model = TransformerXL(
                vocab_size,
                num_layers,
                emb_dim,
                head_dim,
                num_heads,
                p_ff,
                p_att,
                memory_length_train,
            )
            self.model.apply(init_weight_trm)
        elif backbone_name == 'gpt2':
            self.model = GPT2(
                vocab_size,
                num_layers,
                emb_dim,
                num_heads,
                num_positions=max(bptt_train, bptt_val),
            )  # no init for gpt2
        else:
            self.model = LSTMLM(
                backbone_name,
                emb_dim,
                hid_dim,
                vocab_size,
                mog_iters,
                num_layers,
            )
            self.model.apply(init_weight_lstm)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_idxs, return_att=False):
        return self.model(input_idxs, return_att)

    def shared_step(self, batch):
        input_idxs, target_idxs = batch
        # both: [seq_length, batch_size]
        input_idxs, target_idxs = input_idxs.t(), target_idxs.t()
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
        )  # why set to this??

    def validation_step(self, batch, batch_idx):
        loss, seq_length = self.shared_step(batch)
        self.log("loss_val", loss)
        return loss, seq_length

    def validation_epoch_end(self, outputs):
        # we make sure validation loss is not affected by batch size and bptt value.
        self.model.memory = None  # empty memory
        self.model.memory_length = self.hparams.memory_length_train
        losses, seq_lengths = zip(*outputs)
        total_loss = sum([loss * seq_length for loss, seq_length in outputs])
        total_len = sum(seq_lengths)
        self.log("pp_val", torch.exp(total_loss / total_len))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
        # # scheduler = dict(
        # #     scheduler=LinearWarmupCosineAnnealingLR(
        # #         optimizer=optimizer,
        # #         warmup_epochs=self.hparams.warmup_steps,
        # #         max_epochs=self.hparams.max_steps,
        # #         eta_min=self.hparams.eta_min,
        # #     ),
        # #     interval="step",
        # )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    # version 4: start point
    # version 6: bptt 100. Higher batch size 180
    # version 9: No scheduler. 6 layers.
    # version 10: Larger learning rate 5e-4. Higher batch size 200 √
    # version 11: Larger clip 0.5
    # version 25: Larger memory length 150. Batch size 128. Clip 0.5
    # version 26: Batch size 200. Clip 0.25. Memory length 150. Bptt 100.
    # ---- LSTM from now on ----
    # version 13: Naive LSTM (The implementation is WRONG!!! Output gate is calculated using Tanh instead of Sigmoid)
    # version 14: Built-in LSTM, 4 layers.
    # version 15: 1 layers.
    # version 16: Naive LSTM
    # version 17: Mogrifier LSTM
    # ---- every custom LSTM before this is wrongly initialized.
    # version 18: Naive LSTM
    # version 19: Built-in LSTM
    # version 20: Mogrifier LSTM
    # version 21: Larger LR 1e-3. Built-in LSTM. Smaller dim emb_dim 32, hid_dim 64. 
    # version 24: GPT2

    # version 27: replicate of version 24 (GPT2)
    # version 28: replicate of version 19 (builtin LSTM)
    # version 29: replicate of version 18 (naive LSTM)
    # version 30: replicate of version 20 (mogrifier LSTM)
    # version 31: replicate of version 21 (builtin LSTM different parameters)

    from rich.traceback import install
    install()
    pl.seed_everything(42)

    DATA_DIR = "./data/tokenized.pkl"
    BACKBONE_NAME = "builtin_lstm"
    BATCH_SIZE = 256
    NUM_LAYERS = 1
    EMB_DIM = 32
    HEAD_DIM = -1
    HID_DIM = 64
    MOG_ITERS = -1
    NUM_HEADS = -1
    P_FF = -1
    P_ATT = -1
    MEMORY_LENGTH_TRAIN = -1
    MEMORY_LENGTH_VAL = -1
    BPTT_TRAIN = 30
    BPTT_VAL = 30

    LR = 1e-3
    WARMUP_STEPS = 1
    MAX_STEPS = 400_000
    ETA_MIN = 0
    CLIP = 0.5

    datamodule = WikiText2DataModule(DATA_DIR, BATCH_SIZE, BPTT_TRAIN, BPTT_VAL)

    model = Net(
        backbone_name=BACKBONE_NAME,
        vocab_size=datamodule.vocab_size,
        num_layers=NUM_LAYERS,
        emb_dim=EMB_DIM,
        head_dim=HEAD_DIM,
        hid_dim=HID_DIM,
        mog_iters=MOG_ITERS,
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
        batch_size=BATCH_SIZE,
        clip=CLIP,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor='pp_val', save_top_k=1, mode='min')
    ]
    trainer = pl.Trainer(
        gpus=[0],
        max_steps=MAX_STEPS,
        deterministic=True,
        gradient_clip_val=CLIP,
        reload_dataloaders_every_epoch=True,
        callbacks=callbacks,
    )
    
    CHECKPOINT = ""
    if not CHECKPOINT:
        trainer.fit(model, datamodule)
