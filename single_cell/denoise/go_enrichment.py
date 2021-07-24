import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from rich import print as rprint
from rich.traceback import install

install()

from pipeline import read_mtx, qc
from train import Net


DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/babel/snareseq_GSE126074/"
DATA_DIR_NCBI = "/home/tiankang/wusuowei/data/ncbi/"

rprint("---- QC ----")
X = read_mtx(DATA_DIR + "GSE126074_AdBrainCortex_SNAREseq_cDNA.counts.mtx.gz")
rprint(X.shape)
X, good_genes, good_cells = qc(
    X, gene_min_cells=50, gene_min_counts=100, logic="mine", plot=True
)
rprint(X.shape)

rprint("---- experiment gene name ----")
snare_gene = pd.read_csv(
    DATA_DIR + "GSE126074_AdBrainCortex_SNAREseq_cDNA.genes.tsv.gz",
    sep="\t",
    header=None,
    squeeze=True,
)
rprint(snare_gene.is_unique)
snare_gene = snare_gene[good_genes].str.lower()
rprint(snare_gene.is_unique)

rprint("---- NCBI gene name ----")
gene_df = pd.read_csv(
    DATA_DIR_NCBI + "human.tsv", sep="\t", dtype={"chromosome": "str"}
).drop_duplicates(subset="Symbol")[["Symbol", "GeneID"]]
rprint(gene_df.GeneID.is_unique)
rprint(gene_df.Symbol.str.lower().is_unique)
name2id = dict(zip(gene_df.Symbol.str.lower(), gene_df.GeneID))

rprint("---- look for intersection ----")
weight_index1 = snare_gene.isin(gene_df.Symbol.str.lower()).tolist()
good_gene_name1 = snare_gene[weight_index1]
good_gene_id1 = pd.Series(list(map(lambda name: name2id[name], good_gene_name1)))
rprint(good_gene_name1.shape)

rprint("---- NCBI gene2go ----")
gene2go_df = pd.read_csv(DATA_DIR_NCBI + "gene2go", sep="\t")
gene2go_df = gene2go_df[gene2go_df["#tax_id"] == 9606][["GeneID", "GO_ID"]]
gene_ids = gene2go_df.GeneID.unique()
go_terms = gene2go_df.GO_ID.unique()
rprint(len(gene2go_df))
rprint("Number of go terms", len(go_terms))
rprint("Number of genes:", len(gene_ids))

weight_index2 = [id_ in gene_ids for id_ in good_gene_id1]
good_gene_id2 = good_gene_id1[weight_index2]
rprint(len(good_gene_id2))

geneid2index = dict(zip(good_gene_id2, np.arange(len(good_gene_id2))))
go2index = dict(zip(go_terms, np.arange(len(go_terms))))
go_labels = np.zeros((len(go_terms), len(good_gene_id2)), dtype=int)

gene2go_df_good = gene2go_df.query("GeneID in @geneid2index")
# this automatically binarizes.
go_labels[
    (gene2go_df_good.GO_ID.map(go2index), gene2go_df_good.GeneID.map(geneid2index))
] += 1

rprint("---- Loading model weight ----")
CHECKPOINT = "/home/tiankang/wusuowei/deeplearning/single_cell/denoise/lightning_logs/version_65/checkpoints/epoch=49-step=12899.ckpt"
model = Net.load_from_checkpoint(CHECKPOINT)
weight1 = model.model.encoder[0].model[0].weight.detach()
weight2 = model.model.final_linear[0].weight.T.detach()
weight1 = weight1[:, weight_index1][:, weight_index2]
weight2 = weight2[:, weight_index1][:, weight_index2]


def rank_test(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    res = zip(
        *[stats.spearmanr(pred, target) for pred in tqdm(preds) for target in targets]
    )
    sim = np.array(next(res)).reshape(preds.shape[0], targets.shape[0])
    p = np.array(next(res)).reshape(preds.shape[0], targets.shape[0])
    return sim, p


def multilabel_classification(preds: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    loss = nn.BCEWithLogitsLoss()
    sim = np.array(
        [
            loss(pred.unsqueeze(0), target.unsqueeze(0))
            for pred in tqdm(preds)
            for target in targets
        ]
    ).reshape(preds.shape[0], targets.shape[0])
    return sim
