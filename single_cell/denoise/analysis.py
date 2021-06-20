import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from tqdm.auto import tqdm

import pipeline

@torch.no_grad()
def test_unet(
    net: pl.LightningModule,
    checkpoint: str,
    dataset: Dataset,
    data_dir: str,
    label_dir: str,
    plot_dir: str,
):
    model = net.load_from_checkpoint(checkpoint)
    model.eval()
    loader = DataLoader(
        dataset(data_dir),
        batch_size=model.hparams.batch_size,
        shuffle=False,
        num_workers=4
    )
    recons = []
    for x in tqdm(loader):
        recons.append(model(x))
    X_recon = torch.cat(recons, dim=0).cpu().numpy()
    _, X_umap = pipeline.umap(X_recon, n_components=2)
    fig = pipeline.plot(X_umap, np.load(label_dir))
    fig.write_image(plot_dir, width=700, height=500)


if __name__ == '__main__':
    import os
    from train import Net
    from datamodule import scRNADataset
    version = 68
    checkpoint_dir = f'./lightning_logs/version_{version}/checkpoints/'
    checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
    data_dir = '/home/tiankang/wusuowei/data/single_cell/babel/snareseq_GSE126074/train/data.h5ad'
    label_dir = '/home/tiankang/wusuowei/data/single_cell/babel/snareseq_GSE126074/train/labels.npy'
    plot_dir = f'./imgs/imgs/umap_{version}.jpg'
    fig = test_unet(Net, checkpoint, scRNADataset, data_dir, label_dir, plot_dir)
