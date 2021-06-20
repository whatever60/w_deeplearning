from typing import Type

import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics.functional import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from train import Net
from datamodule import scRNADataModule, SNAREDataModule
from utils import knn
import pipeline


@torch.no_grad()
def test_npid(
    net: Type[pl.LightningModule],
    checkpoint: str,
    datamodule: Type[pl.LightningDataModule],
    data_dir: str,
    heatmap_path: str,
    umap_path: str,
    hparams_file: str = None
):
    model = net.load_from_checkpoint(checkpoint, hparams_file=hparams_file)
    module = scRNADataModule(data_dir, model.hparams.batch_size)
    model.eval()
    module.setup(stage="fit")
    train_loader = module.train_dataloader()
    val_loader = module.val_dataloader()
    labels_train = torch.zeros(module.length_train, dtype=int)
    labels_val = torch.zeros(module.length_val, dtype=int)
    output_val = torch.zeros(module.length_val, model.hparams.feature_dim)
    for _, targets, indices in tqdm(train_loader):
        labels_train[indices] = targets
    for imgs, targets, indices in tqdm(val_loader):
        labels_val[indices] = targets
        output_val[indices] = model(imgs)

    rprint('[green]KNNing...')
    preds = knn(
        module.num_classes,
        model.model.memory,
        output_val,
        labels_train,
        model.hparams.n_neighbors,
        model.hparams.nce_t,
        relax=3,
    )
    matrix = confusion_matrix(preds[:, 0], labels_val, module.num_classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix.cpu().numpy(),
        center=(matrix.max() + matrix.min()) / 2,
        ax=ax
    )
    plt.savefig(heatmap_path)
    rprint('[green]UMAPing...')
    _, X_umap = pipeline.umap(output_val.cpu().numpy(), n_components=2)
    fig = pipeline.plot(X_umap, labels_val.cpu().numpy())
    fig.write_image(umap_path, width=600, height=600)


@torch.no_grad()
def test_npid_snare(
    net,
    checkpoint,
    datamodule,
    data_dir,
    labels_path,
    umap_path,
    hparams_file=None,
):
    model = net.load_from_checkpoint(checkpoint, hparams_file=hparams_file)
    labels = np.load(labels_path)
    module = datamodule(data_dir, model.hparams.batch_size)
    model.eval()
    module.setup(stage="fit")
    train_loader = module.train_dataloader()
    output = torch.zeros(module.length_train, model.hparams.feature_dim)
    for imgs, _, indices in tqdm(train_loader):
        output[indices] = model(imgs)
    _, X_umap = pipeline.umap(output.cpu().numpy(), n_components=2)
    fig = pipeline.plot(X_umap, labels)
    fig.write_image(umap_path, width=700, height=500)


if __name__ == '__main__':
    from rich import print as rprint
    from rich.traceback import install
    install()

    checkpoint = '/home/tiankang/wusuowei/deeplearning/single_cell/self_supervised/lightning_logs/version_75/checkpoints/epoch=2-step=242.ckpt'
    data_dir = "/home/tiankang/wusuowei/data/single_cell/babel/snareseq_GSE126074/"
    heatmap_path = './imgs/snare_seq/heatmap_1.jpg'
    umap_path = './imgs/snare_seq/umap_version75.jpg'

    labels_path = "/home/tiankang/wusuowei/data/single_cell/babel/snareseq_GSE126074/train/labels.npy"
    test_npid_snare(Net, checkpoint, SNAREDataModule, data_dir, labels_path, umap_path)
