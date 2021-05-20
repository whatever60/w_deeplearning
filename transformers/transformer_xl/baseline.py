import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM

from datamodule import WikiText2DataModule

from tqdm.auto import tqdm


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()

    pl.seed_everything(42)
    DATA_DIR = "./data/tokenized_gpt.pkl"
    BATCH_SIZE = 128
    BPTT_TRAIN = 64
    BPTT_VAL = 64
    datamodule = WikiText2DataModule(DATA_DIR, BATCH_SIZE, BPTT_TRAIN, BPTT_VAL)
    datamodule.setup(stage="fit")
    loader = datamodule.val_dataloader()

    device = "cuda:9"
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss().to(device)
    losses = []
    seq_lengths = []
    with torch.no_grad():
        for batch in tqdm(loader):
            input_idxs, target_idxs = batch
            # rprint(target_idxs.shape)
            input_idxs, target_idxs = input_idxs.to(device), target_idxs.to(
                device
            )  # both: [seq_length, batch_size]
            logits = model(input_idxs).logits  # [seq_length, batch_size, vocab_size]
            seq_length = input_idxs.shape[0]
            loss = criterion(logits.flatten(end_dim=1), target_idxs.flatten())
            losses.append(loss)
            seq_lengths.append(seq_length)
        total_loss = sum(
            [loss * seq_length for loss, seq_length in zip(losses, seq_lengths)]
        )
        total_len = sum(seq_lengths)
        rprint(torch.exp(total_loss / total_len))
