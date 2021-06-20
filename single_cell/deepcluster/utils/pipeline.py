import gzip

import numpy as np
from scipy import sparse as ss
import pandas as pd
import datatable as dt
from sklearn import preprocessing
import matplotlib.pyplot as plt


def get_skip_to_line(filename: str):
    if filename.endswith(".gz"):
        open_func = gzip.open
        open_mode = "rb"
    else:
        open_func = open
        open_mode = "r"
    skip_to_line = 0
    with open_func(filename, open_mode) as f:
        for line in f:
            skip_to_line += 1
            statistics = line.split(",")
            if statistics[0].isdigit():
                gene_num, cell_num, total_count = list(map(int, line.split(",")))
                break
    return skip_to_line, gene_num, cell_num, total_count


def read_mtx(mtx_path):

    skip_to_line, gene_num, cell_num, total_count = get_skip_to_line(mtx_path)
    df = dt.fread(mtx_path, skip_to_line=skip_to_line, header=False).to_pandas()
    df.columns = ["gene", "cell", "counts"]
    X_sparse = ss.csr_matrix(
        (df.counts, (df.cell, df.gene)), shape=(cell_num, gene_num), dtype="int16"
    )
    return X_sparse


def plot_qc(X, path: str) -> None:
    """
    Args:
        X: np.ndarray | scipy.sparse.spmatrix. Expression matrix.
        labels: np.ndarray | pd.Series | None. Optional labels in case qc filters cells.
    """
    bins = 80
    X_binarized = X > 0
    cell_counts = _sum(X, axis=1)
    cell_genes = _sum(X_binarized, axis=1)
    gene_counts = _sum(X, axis=0)
    gene_cells = _sum(X_binarized, axis=0)
    _, axs = plt.subplots(2, 2, figsize=(20, 15))
    axs[0, 0].set_title("Total counts per cell")
    axs[0, 1].set_title("Detected genes per cell")
    axs[1, 0].set_title("Total counts per gene")
    axs[1, 1].set_title("Appeared cells per gene")
    axs[0, 0].set_xlabel("Counts")
    axs[0, 1].set_xlabel("Genes")
    axs[1, 0].set_xlabel("Counts")
    axs[1, 1].set_xlabel("Cells")
    axs[0, 0].hist(cell_counts, bins=bins)
    axs[0, 1].hist(cell_genes, bins=bins)
    axs[1, 0].hist(gene_counts, bins=bins)
    axs[1, 1].hist(gene_cells, bins=bins)
    plt.savefig(path)


def normalize(
    X,
    apply_qc: bool = True,
    log: bool = True,
    log_first: bool = False,
    norm_factor: int = 0,
    plot: bool = False,
    preqc_path: str = "./imgs/preqc.jpg",
    postqc_path: str = "./imgs/postqc.jpg",
    **qc_kwargs,
):
    """
    Args:
        X: np.ndarray | scipy.sparse.spmatrix. Expression matrix.

    """
    if plot:  # qc plot may be needed even if we do not apply qc
        plot_qc(X, preqc_path)
    if apply_qc:
        print(X.shape)
        X, good_genes, good_cells = qc(X, **qc_kwargs)
        print(X.shape)
        if plot:
            plot_qc(X, postqc_path)
    if log:
        if log_first:
            X = preprocessing.normalize(_log1p(X), norm="l1")
        else:
            if not norm_factor:
                # use the median of total counts as factor.
                norm_factor = np.median(_sum(X, axis=1))
                print("Median:", norm_factor)
            X = _log1p(preprocessing.normalize(X, norm="l1") * norm_factor)
    else:
        X = preprocessing.normalize(X, norm="l1")
    X /= X.max()
    # X = log1p(X)
    # max_ = X.max(axis=1).A.flatten()
    # X = spmatrix_divide_vector(X, np.where(max_ == 0, np.inf, max_))
    return X.astype("float32"), good_genes, good_cells


def _sum(X, axis):
    if ss.issparse(X):
        return X.sum(axis=axis).A.flatten()
    else:
        return X.sum(axis=axis)


def _log1p(X):
    if ss.issparse(X):
        return X.log1p()
    else:
        return np.log1p(X)


def _clip(X, q) -> None:
    if ss.issparse(X):
        thres = np.percentile(X.data, q)
        X.data = X.data.clip(max=thres)
    else:
        thres = np.percentile(X, q)
        X.clip(max=thres)


def qc(
    X,
    *,
    clip_q=100,
    cell_min_counts=0,
    cell_max_counts=np.inf,
    cell_min_genes=0,
    cell_max_genes=np.inf,
    gene_min_counts=0,
    gene_max_counts=np.inf,
    gene_min_cells=0,
    gene_max_cells=np.inf,
    logic="mine",
):

    assert logic in ("standard", "mine")
    cell_num, gene_num = X.shape
    for i in cell_min_genes, cell_max_genes:
        if (not np.isinf(i)) and isinstance(i, float):
            i = int(i * gene_num)
    for i in gene_min_cells, gene_max_cells:
        if (not np.isinf(i)) and isinstance(i, float):
            i = int(i * cell_num)

    if clip_q < 100:
        _clip(X, clip_q)  # clip before calculating the following statistics
    X_binarized = X > 0
    cell_counts = pd.Series(_sum(X, axis=1))  # total count per cell
    cell_genes = pd.Series(_sum(X_binarized, axis=1))  # detected gene per cell
    gene_counts = pd.Series(_sum(X, axis=0))
    gene_cells = pd.Series(_sum(X_binarized, axis=0))

    def qc_standard():
        good_cells = (
            (cell_min_counts <= cell_counts)
            & (cell_counts <= cell_max_counts)
            & (cell_min_genes <= cell_genes)
            & (cell_genes <= cell_max_genes)
        ).values
        good_genes = (
            (gene_min_counts <= gene_counts)
            & (gene_min_counts <= gene_max_counts)
            & (gene_min_cells <= gene_cells)
            & (gene_cells <= gene_max_cells)
        ).values
        return good_cells, good_genes

    def qc_mine():
        good_cells = (
            (cell_min_counts <= cell_counts)
            & (cell_counts <= cell_max_counts)
            & (cell_min_genes <= cell_genes)
            & (cell_genes <= cell_max_genes)
        ).values
        good_genes = (
            (gene_counts <= gene_max_counts)
            & ((gene_min_counts <= gene_counts) | (gene_min_cells <= gene_cells))
            & (gene_cells <= gene_max_cells)
        ).values
        return good_cells, good_genes

    if logic == "standard":
        good_cells, good_genes = qc_standard()
    elif logic == "mine":
        good_cells, good_genes = qc_mine()
    else:
        raise NotImplementedError

    return X[good_cells][:, good_genes].copy().astype("int32"), good_genes, good_cells


def spmatrix_divide_vector(X_sparse, vec):
    """Divide a scipy sparse matrix by a vector.
    This function exists because division is not implemented for scipy sparse matrix.
    """
    if len(vec) == X_sparse.shape[1]:
        return X_sparse @ ss.diags(1 / vec)
    else:
        return (X_sparse.T @ ss.diags(1 / vec)).T


def labels_to_ids(series: pd.Series) -> list:
    assert len(series.shape) == 1
    labels_to_ids_dict = {label: idx for idx, label in enumerate(series.unique())}
    return (
        np.array(list(map(lambda label: labels_to_ids_dict[label], series))).astype(
            int
        ),
        labels_to_ids_dict,
    )
