'''
Args:
    naive: if True, use a naive model instead of lego model
    pretrain: params.pt file to use to warm initialize the model (instead of starting from scratch)
'''
import os
import logging

import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

import model
import loss
import dataloader

NAIVE = True
BATCH_SIZE = 512
OPTIMIZER = optim.Adam
HID_DIM = 16
LOSSWEIGHT = 4 / 3
LR = 1e-2
LINEAR = False
CLUSTER_METHOD = 'leiden'
VALID_CLUSTER = 0
TEST_CLUSTER = 1

CHECKPOINT = None



DATA_DIR = '/home/tiankang/wusuowei/data/single_cell/babel'
SNARESEQ_DATA_DIR = os.path.join(DATA_DIR, 'snareseq_GSE126074')


SNARESEQ_ATAC_CELL_INFO = pd.read_csv(
    os.path.join(
        SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_chromatin.barcodes.tsv.gz"
    ),
    sep="\t",
    header=None,
    index_col=0,
)

SNARESEQ_ATAC_PEAK_INFO = pd.read_csv(
    os.path.join(
        SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_chromatin.peaks.tsv.gz"
    ),
    sep="\t",
    header=None,
    index_col=0,
)

SNAERSEQ_ATAC_DATA_KWARGS = dict(
    fname=os.path.join(SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_chromatin.counts.mtx.gz"),
    gene_info=SNARESEQ_ATAC_PEAK_INFO,
    cell_info=SNARESEQ_ATAC_CELL_INFO,
    transpose=True,  # True means the matrix is peak/gene x cell.
    selfsupervised=True,  # Useless
    binarize=True,
    autosomes_only=True,
    split_by_chrom=True,
    concat_outputs=True,
    filt_gene_min_counts=5,  # peaks with fewer than five counts overall
    filt_gene_min_cells=5,  # keep peaks seek in >= 5 cells
    filt_gene_max_cells=0.1,  # filter peaks expressing in more than 10% of cells
    pool_genomic_interval=0,
    normalize=False,
    log_trans=False,
    y_mode='x',
    calc_size_factors=False,
    return_sf=False
)


SNARESEQ_RNA_GENE_INFO = pd.read_csv(
    os.path.join(SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_cDNA.genes.tsv.gz"),
    sep='\t',
    header=None,
    index_col=0
)

SNARESEQ_RNA_CELL_INFO = pd.read_csv(
    SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_cDNA.barcodes.tsv.gz",
    sep='\t',
    header=None,
    index_col=0
)

MM10_GTF = os.path.join(DATA_DIR, "gencode.vM7.annotation.gtf.gz")

SNARESEQ_RNA_DATA_KWARGE = dict(
    fname=os.path.join(SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_cDNA.counts.mtx.gz"),
    gene_info=SNARESEQ_RNA_GENE_INFO,
    cell_info=SNARESEQ_RNA_CELL_INFO,
    transpose=True,  # True means the matrix is peak/gene x cell.
    selfsupervised=True,  # Useless
    binarize=False,
    autosomes_only=True,
    split_by_chrom=True,
    concat_outputs=True,
    sort_by_pos=True,
    gtf_file=MM10_GTF,
    filt_cell_min_genes=200,
    filt_cell_max_genes=2500,
    normalize=True,
    log_trans=True,
    clip=0.5,  # Clip the bottom and top 0.5%
    y_mode='size_norm',
    calc_size_factors=True,
    return_sf=False,
    cluster_res=1.5
)


sc_rna_dataset = dataloader.SingleCellDataset(
    valid_cluster_id=VALID_CLUSTER,
    test_cluster_id=TEST_CLUSTER,
    data_split_by_cluster_log=LINEAR,
    data_split_by_cluster=CLUSTER_METHOD
    **SNARESEQ_RNA_DATA_KWARGE
)

sc_rna_train_dataset = dataloader.SingleCellDatasetSplit(
    sc_rna_dataset,
    split="train",
)
sc_rna_valid_dataset = dataloader.SingleCellDatasetSplit(
    sc_rna_dataset,
    split="valid",
)
sc_rna_test_dataset = dataloader.SingleCellDatasetSplit(
    sc_rna_dataset,
    split="test",
)

sc_atac_dataset = dataloader.SingleCellDataset(
    predefined_split=sc_rna_dataset,
    cluster_res=0,
    **SNAERSEQ_ATAC_DATA_KWARGS
)

sc_atac_train_dataset = dataloader.SingleCellDatasetSplit(
    sc_atac_dataset,
    split="train",
)
sc_atac_valid_dataset = dataloader.SingleCellDatasetSplit(
    sc_atac_dataset,
    split="valid",
)
sc_atac_test_dataset = dataloader.SingleCellDatasetSplit(
    sc_atac_dataset,
    split="test",
)

sc_dual_train_dataset = dataloader.PairedDataset(
    sc_rna_train_dataset,
    sc_atac_train_dataset,
    flat_mode=True,
)
sc_dual_valid_dataset = dataloader.PairedDataset(
    sc_rna_valid_dataset,
    sc_atac_valid_dataset,
    flat_mode=True,
)
sc_dual_test_dataset = dataloader.PairedDataset(
    sc_rna_test_dataset,
    sc_atac_test_dataset,
    flat_mode=True,
)
sc_dual_full_dataset = dataloader.PairedDataset(
    sc_rna_dataset,
    sc_atac_dataset,
    flat_mode=True,
)



# Instantiate and train model
model_class = (
    model.NaiveSplicedAutoEncoder
    if NAIVE
    else model.AssymSplicedAutoEncoder
)
spliced_net = model.SplicedmodelkorchNet(
    module=model_class,
    module__hidden_dim=HID_DIM,  # Based on hyperparam tuning
    module__input_dim1=sc_rna_dataset.data_raw.shape[1],
    module__input_dim2=sc_atac_dataset.get_per_chrom_feature_count(),
    module__final_activations1=[
        model.ClippedExp(),
        model.ClippedSoftplus(),
    ],
    module__final_activations2=nn.Sigmoid(),
    module__flat_mode=True,
    # module__seed=rand_seed,
    lr=LR,  # Based on hyperparam tuning
    criterion=loss.QuadLoss,
    criterion__loss2=loss.BCELoss,  # handle output of encoded layer
    criterion__loss2_weight=LOSSWEIGHT,  # numerically balance the two losses with different magnitudes
    criterion__record_history=True,
    optimizer=OPTIMIZER,
    iterator_train__shuffle=True,
    # device=utils.get_device(args.device),
    batch_size=BATCH_SIZE,  # Based on  hyperparam tuning
    max_epochs=500,
    # callbacks=[
    #     skorch.callbacks.EarlyStopping(patience=args.earlystop),
    #     skorch.callbacks.LRScheduler(
    #         policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
    #         **model_utils.REDUCE_LR_ON_PLATEAU_PARAMS,
    #     ),
    #     skorch.callbacks.GradientNormClipping(gradient_clip_value=5),
    #     skorch.callbacks.Checkpoint(
    #         dirname=outdir_name,
    #         fn_prefix="net_",
    #         monitor="valid_loss_best",
    #     ),
    # ],
    # train_split=skorch.helper.predefined_split(sc_dual_valid_dataset),
    iterator_train__num_workers=8,
    iterator_valid__num_workers=8,
)
if CHECKPOINT:
    # Load in the warm start parameters
    raise ValueError
    # spliced_net.load_params(f_params=args.pretrain)
    # spliced_net.partial_fit(sc_dual_train_dataset, y=None)
else:
    spliced_net.fit(sc_dual_train_dataset, y=None)


logging.info("Evaluating on test set")
logging.info("Evaluating RNA > RNA")
sc_rna_test_preds = spliced_net.translate_1_to_1(sc_dual_test_dataset)
sc_rna_test_preds_anndata = sc.AnnData(
    sc_rna_test_preds,
    var=sc_rna_test_dataset.data_raw.var,
    obs=sc_rna_test_dataset.data_raw.obs,
)
# sc_rna_test_preds_anndata.write_h5ad(
#     os.path.join(outdir_name, "rna_rna_test_preds.h5ad")
# )

logging.info("Evaluating ATAC > ATAC")
sc_atac_test_preds = spliced_net.translate_2_to_2(sc_dual_test_dataset)
sc_atac_test_preds_anndata = sc.AnnData(
    sc_atac_test_preds,
    var=sc_atac_test_dataset.data_raw.var,
    obs=sc_atac_test_dataset.data_raw.obs,
)
# sc_atac_test_preds_anndata.write_h5ad(
#     os.path.join(outdir_name, "atac_atac_test_preds.h5ad")
# )

logging.info("Evaluating ATAC > RNA")
sc_atac_rna_test_preds = spliced_net.translate_2_to_1(sc_dual_test_dataset)
sc_atac_rna_test_preds_anndata = sc.AnnData(
    sc_atac_rna_test_preds,
    var=sc_rna_test_dataset.data_raw.var,
    obs=sc_rna_test_dataset.data_raw.obs,
)
# sc_atac_rna_test_preds_anndata.write_h5ad(
#     os.path.join(outdir_name, "atac_rna_test_preds.h5ad")
# )

logging.info("Evaluating RNA > ATAC")
sc_rna_atac_test_preds = spliced_net.translate_1_to_2(sc_dual_test_dataset)
sc_rna_atac_test_preds_anndata = sc.AnnData(
    sc_rna_atac_test_preds,
    var=sc_atac_test_dataset.data_raw.var,
    obs=sc_atac_test_dataset.data_raw.obs,
)
# sc_rna_atac_test_preds_anndata.write_h5ad(
#     os.path.join(outdir_name, "rna_atac_test_preds.h5ad")
# )

