import torch
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from train import Net

from rich import print as rprint
from rich.traceback import install
install()


if __name__ == '__main__':
    # ---- tokenize the test text ----
    test_text = """
<unk> played basketball and baseball during high school as well . He <unk> in basketball with the Knights . At the 2001 Big League World Series , <unk> played for Team Canada as a shortstop and center <unk> , helping the team to third place in the international competition . Despite being skilled as a baseball player , <unk> chose football over professional baseball due to the <unk> of the former sport . 
"""
    tokenizer = Tokenizer.from_file('./tokenizer/tokenizer.json')
    token_ids = tokenizer.encode(test_text).ids
    tokens = tokenizer.encode(test_text).tokens
    rprint(len(token_ids))
    memory_length = 40
    bptt = 42
    input_1 = torch.tensor(token_ids[:memory_length]).unsqueeze(dim=1)  # seq_length first
    input_2 = torch.tensor(token_ids[memory_length: memory_length+bptt]).unsqueeze(dim=1)

    checkpoint = './lightning_logs/version_10/checkpoints/epoch=3539-step=399999.ckpt'
    hparams_file = './lightning_logs/version_10/hparams.yaml'
    model = Net.load_from_checkpoint(checkpoint, hparams_file=hparams_file)
    # ---- fill the memory ----
    model.eval()
    with torch.no_grad():
        model(input_1)  # fill the memory
        rprint(len(model.model.memory))
        rprint(model.model.memory[1].shape)
        _, atts = model(input_2, return_att=True)

    # ---- normalization to account for masking ----
    for att in atts:
        for i in range(bptt):
            att[i] *= ((i + memory_length + 1) / (memory_length + bptt))

    # ---- make attention value sparse ----
    for att in atts:
        for i in range(model.hparams.num_heads):
            att[..., i] = torch.where(
                att[..., i] < att[..., i].quantile(0.9),
                torch.tensor(0).float(),
                att[..., i]
            )

    # ---- plot heatmap ----
    num_layers = model.hparams.num_layers
    num_heads = model.hparams.num_heads
    f, axs = plt.subplots(num_layers, num_heads, figsize=(22 * num_heads, 10 * num_layers))
    for i in range(model.hparams.num_layers):
        for j in range(model.hparams.num_heads):
            fig = sns.heatmap(atts[i][..., 0, j], ax=axs[i, j], cmap="OrRd", linewidths=.5)
            fig.set_xticklabels(tokens)
            fig.set_yticklabels(tokens[memory_length: memory_length + bptt])
            plt.xticks(
                rotation=90,
                horizontalalignment='right',
            )
            plt.yticks(
                rotation=0,
                horizontalalignment='right',
            )
    plt.savefig('./att_all.jpg')    
    