import numpy as np
from scipy import io
import pandas as pd
import anndata
from sklearn.model_selection import train_test_split

from utils import read_mtx, normalize, qc, plot_qc


def get_splitseq_p2_brain(mat_path, processed_dir, image_dir) -> None:
    """
    Process raw data of split_seq (in .mat format) and get only p2_brain profile.
    """
    dic = io.loadmat(mat_path)
    mask = dic["sample_type"] == "p2_brain "  # believe it or not, there is a space.
    X = dic["DGE"][mask]
    gene = dic["genes"]
    cell = dic["barcodes"][0][mask]  # the shape of `barcodes` is (1, 156049)
    plot_qc(X, f"{image_dir}/processed/splitseq_preqc.jpg")

    X, good_genes, good_cells = normalize(
        X,
        apply_qc=True,
        gene_min_cells=50,
        gene_min_counts=100,
        plot=True,
        postqc_path=f"{image_dir}/processed/splitseq_postqc.jpg",
    )
    gene = gene[good_genes]
    cell = cell[good_cells]

    anndata.AnnData(X=X).write(f"{processed_dir}/data.h5ad")
    pd.Series(gene).to_csv(f"{processed_dir}/gene.csv", index=False, header=None)
    pd.Series(cell).to_csv(f"{processed_dir}/barcode.csv", index=False, header=None)


def get_snareseq_p0(data_dir, image_dir) -> None:
    """
    Process raw data of snare_seq (in .mtx format)
    """
    X = read_mtx(f"{data_dir}/raw/GSE126074_P0_BrainCortex_SNAREseq_cDNA.counts.mtx.gz")
    plot_qc(X, f"{image_dir}/snareseq_preqc.jpg")
    gene = pd.read_csv(
        f"{data_dir}/GSE126074_P0_BrainCortex_SNAREseq_cDNA.genes.tsv.gz", header=None
    )
    cell = pd.read_csv(
        f"{data_dir}/GSE126074_P0_BrainCortex_SNAREseq_cDNA.barcodes.tsv.gz",
        header=None,
    )

    X, good_genes, good_cells = normalize(
        X,
        apply_qc=True,
        gene_min_cells=25,
        gene_min_counts=50,
        plot=True,
        postqc_path=f"{image_dir}/snareseq_postqc.jpg",
    )
    gene = gene[good_genes]
    cell = cell[good_cells]

    anndata.AnnData(X=X).write(f"{data_dir}/processed/data.h5ad")
    gene.to_csv(f"{data_dir}/processed/gene.csv", index=False, header=None)
    cell.to_csv(f"{data_dir}/processed/barcode.csv", index=False, header=None)


def profile_train_test_split(data_dir, val_size) -> None:
    X = anndata.read_h5ad(f"{data_dir}/data.h5ad").X
    barcode = pd.read_csv(f"{data_dir}/barcode.csv", header=None)
    X_train, X_val, cell_train, cell_val = train_test_split(
        X, barcode, test_size=val_size
    )

    anndata.AnnData(X=X_train).write(f"{data_dir}/train/data.h5ad")
    anndata.AnnData(X=X_val).write(f"{data_dir}/val/data.h5ad")
    cell_train.to_csv(f"{data_dir}/train/barcode.csv", index=False, header=None)
    cell_val.to_csv(f"{data_dir}/val/barcode.csv", index=False, header=None)


def gen_snare_replicate(data_dir, val_size, binomial_p=0.85):
    X = read_mtx(f"{data_dir}/raw/GSE126074_P0_BrainCortex_SNAREseq_cDNA.counts.mtx.gz")
    gene = pd.read_csv(
        f"{data_dir}/raw/GSE126074_P0_BrainCortex_SNAREseq_cDNA.genes.tsv.gz",
        header=None,
    )
    cell = pd.read_csv(
        f"{data_dir}/raw/GSE126074_P0_BrainCortex_SNAREseq_cDNA.barcodes.tsv.gz",
        header=None,
    )

    X, good_genes, good_cells = qc(
        X, gene_min_cells=25, gene_min_counts=50, logic="mine"
    )

    # save preprocessed data
    X_normalized = normalize(X, apply_qc=False, norm_factor=1000)
    anndata.AnnData(X=X_normalized).write(f"{data_dir}/processed/data.h5ad")
    gene = gene[good_genes]
    cell = cell[good_cells]
    gene.to_csv(f"{data_dir}/processed/gene.csv", index=False, header=None)
    cell.to_csv(f"{data_dir}/processed/barcode.csv", index=False, header=None)

    # generate pseudo replicate
    X1, X2 = pseudo_replicate(X, binomial_p)
    X1, X2 = normalize(X1, apply_qc=False, norm_factor=1000), normalize(X2, apply_qc=False, norm_factor=1000)
    X1_train, X1_val, X2_train, X2_val, cell_train, cell_val = train_test_split(
        X1, X2, cell, test_size=val_size, random_state=42
    )

    # save pseudo replicate
    anndata.AnnData(X1_train).write_h5ad(f"{data_dir}/processed/train/data1.h5ad")
    anndata.AnnData(X2_train).write_h5ad(f"{data_dir}/processed/train/data2.h5ad")
    anndata.AnnData(X1_val).write_h5ad(f"{data_dir}/processed/val/data1.h5ad")
    anndata.AnnData(X2_val).write_h5ad(f"{data_dir}/processed/val/data2.h5ad")
    cell_train.to_csv(
        f"{data_dir}/processed/train/barcode.csv", index=False, header=None
    )
    cell_val.to_csv(f"{data_dir}/processed/val/barcode.csv", index=False, header=None)


def pseudo_replicate(X_sparse, p):
    replicate_data = np.array([np.random.binomial(i, p, 2) for i in X_sparse.data])
    replicate1 = X_sparse.copy()
    replicate1.data = replicate_data[:, 0]
    replicate2 = X_sparse.copy()
    replicate2.data = replicate_data[:, 1]
    return replicate1, replicate2


if __name__ == "__main__":
    from rich.traceback import install

    install()

    splitseq_mat_path = "/home/tiankang/wusuowei/data/single_cell/processed/split_seq/raw/GSM3017261_150000_CNS_nuclei.mat"
    splitseq_processed_dir = (
        "/home/tiankang/wusuowei/data/single_cell/processed/split_seq/processed"
    )
    image_path = "./images/qc"
    # get_splitseq_p2_brain(splitseq_mat_path, splitseq_processed_dir, image_path)

    snareseq_data_dir = "/home/tiankang/wusuowei/data/single_cell/snare_seq"
    # get_snareseq_p0(snareseq_data_dir, image_path)
    # profile_train_test_split(f'{snareseq_data_dir}/processed', 800)
    gen_snare_replicate(snareseq_data_dir, 800)
