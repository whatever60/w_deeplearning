import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm

from train import Net
from datamodule import WikiText2DataModule


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()

    pl.seed_everything(42)

    device = "cuda:9"
    model_name = 'transformer_xl'
    rprint(f'[green]{model_name}')
    if model_name == 'baseline':
        data_dir = "./data/tokenized_gpt.pkl"
        batch_size = 128
        bptt_train = 64
        bptt_val = 64
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    else:
        hparams_file = None
        if model_name == 'builtin_lstm':
            checkpoint = './lightning_logs/version_31/checkpoints/epoch=124-step=36874.ckpt'
        elif model_name == 'naive_lstm':
            checkpoint = './lightning_logs/version_29/checkpoints/epoch=112-step=33334.ckpt'
        elif model_name == 'mog_lstm':
            checkpoint = './lightning_logs/version_30/checkpoints/epoch=98-step=29204.ckpt'
        elif model_name == 'gpt2':
            checkpoint = './lightning_logs/version_27/checkpoints/epoch=19-step=2759.ckpt'
        elif model_name == 'transformer_xl':
            checkpoint = './lightning_logs/version_10/checkpoints/epoch=3539-step=399999.ckpt'
            hparams_file = './lightning_logs/version_10/hparams.yaml'
        else:
            raise NotImplementedError
        model = Net.load_from_checkpoint(checkpoint, hparams_file=hparams_file)
        data_dir = "./data/tokenized.pkl"
        batch_size = model.hparams.batch_size
        bptt_train = model.hparams.bptt_train
        bptt_val = model.hparams.bptt_val
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss().to(device)
    
    datamodule = WikiText2DataModule(data_dir, batch_size, bptt_train, bptt_val)
    datamodule.setup(stage="test")
    loader = datamodule.test_dataloader()

    
    losses = []
    seq_lengths = []
    with torch.no_grad():
        for batch in tqdm(loader):
            input_idxs, target_idxs = batch
            # both: [batch_size, seq_length]
            if model_name == 'baseline':
                # Huggingface 🤗 models assume batch first
                input_idxs = input_idxs.to(device)
                target_idxs = target_idxs.to(device)
                logits = model(input_idxs).logits  # [seq_length, batch_size, vocab_size]
            else:
                input_idxs = input_idxs.t().to(device)
                target_idxs = target_idxs.t().to(device)
                logits = model(input_idxs)  # [seq_length, batch_size, vocab_size]
            seq_length = input_idxs.shape[0]
            loss = criterion(logits.flatten(end_dim=1), target_idxs.flatten())
            losses.append(loss)
            seq_lengths.append(seq_length)
        total_loss = sum(
            [loss * seq_length for loss, seq_length in zip(losses, seq_lengths)]
        )
        total_len = sum(seq_lengths)
        rprint(torch.exp(total_loss / total_len))
