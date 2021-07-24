import pickle
import gc
import gzip

import numpy as np
from scipy import sparse as ss
from scipy.sparse.linalg import LinearOperator, svds
import pandas as pd
import datatable as dt
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import svd_flip
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, pairwise_distances  # , r2_score
from sklearn.model_selection import train_test_split
import torch

from umap import UMAP
from hdbscan import HDBSCAN

import optuna
import xgboost as xgb

import matplotlib.pyplot as plt
from plotly import graph_objects as go
from babyplots import Babyplot

import clustering


def get_skip_to_line(filename: str):
    if filename.endswith(".gz"):
        open_func = gzip.open
        split_func = lambda x: x.decode().strip().split(" ")
    else:
        open_func = open
        split_func = lambda x: x.strip().split(" ")
    skip_to_line = 1
    with open_func(filename) as f:
        for line in f:
            skip_to_line += 1
            statistics = split_func(line)
            if statistics[0].isdigit():
                gene_num, cell_num, total_count = list(map(int, split_func(line)))
                break
    return skip_to_line, gene_num, cell_num, total_count


def read_mtx(mtx_path):
    skip_to_line, gene_num, cell_num, total_count = get_skip_to_line(mtx_path)
    df = dt.fread(mtx_path, skip_to_line=skip_to_line, header=False).to_pandas()
    df.columns = ["gene", "cell", "counts"]
    df.gene -= 1
    df.cell -= 1
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
    postqc_path: str = "./images/postqc.jpg",
    **qc_kwargs,
):
    """
    Args:
        X: np.ndarray | scipy.sparse.spmatrix. Expression matrix.

    """
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
    if apply_qc:
        return X.astype("float32"), good_genes, good_cells
    else:
        return X.astype("float32")


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
    clip_q=99,
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


def magic(X: np.ndarray, ka: int, X_pca: np.ndarray = None) -> np.ndarray:
    """Magic Algorithm
    Args:
        X: Full expression matrix after normalization, before or after PCA.
        ka: int. Number of nearest neighbor when estimating kernel.
        X_pca: Precomputed PCA. If PCA matrix has been precomputed, leave this to None and X will be used for distance calculation.
    """
    if X_pca is None:
        X_pca = X

    dist = pairwise_distances(X_pca)
    sigma = np.partition(dist, ka)[:, ka]  # the ka-th smallest
    A = np.exp(-((dist / sigma[:, np.newaxis]) ** 2))
    sigma3 = np.partition(A, -3 * ka)[:, -3 * ka][:, np.newaxis]  # the 3ka-th largest
    A = np.where(A < sigma3, 0, A)
    np.fill_diagonal(A, 1)
    A += A.T
    A /= A.sum(axis=1, keepdims=True)
    return A


def diffusion(A, X):
    i = R2 = 1
    M = A
    X_old = X
    while i < 20 and R2 > 1e-3:  # < 0.95:
        X_new = M @ X
        # R2 = r2_score(X_old, X_new)
        R2 = (np.sqrt(np.square(X_new - X_old).sum(axis=0)) / X_old.sum(axis=0)).mean()
        # R2 = 1 - np.square(X_new - X_old).sum() / np.square(X_old - X_old.mean(axis=0)).sum()
        print(R2)
        M = preprocessing.normalize(A @ M, "l1")
        X_old = X_new
    return X_old


def test_magic():
    X_sparse = ss.csr_matrix(np.random.binomial(20, 0.02, size=(100, 1000))).astype(
        "float32"
    )
    ka = 5
    return magic(X_sparse, ka)


def svd_with_sparse(X, k, solver="arpack", fit_transform=True, random_state=None):
    """
    Perform svd on sparse matrix
    Args:
        X: scipy.spmatrix
    """
    random_init = np.random.rand(np.min(X.shape))

    mu = X.mean(axis=0).A.flatten()  # d
    vars_ = preprocessing.StandardScaler(with_mean=False).fit(X).var_
    XH = X.T.conj()  # d x n

    def matvec(x):
        # print(x.shape)
        return X @ x - mu @ x
        # return Xdot(x) - mdot(x)

    def matmat(x):
        # print(x.shape)
        return X @ x - (mu @ x)[:, np.newaxis]
        # return Xmat(x) - mmat(x)

    def rmatvec(x):
        # x: n
        return XH @ x - mu * x.sum()
        # return XHdot(x) - mhdot(ones(x))

    def rmatmat(x):
        # x: n x k
        return XH @ x - (mu * x.sum())[:, np.newaxis]
        # return XHmat(x) - mhmat(ones(x))

    XL = LinearOperator(
        matvec=matvec,
        matmat=matmat,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
        shape=X.shape,
        dtype=X.dtype,
    )

    u, s, v = svds(XL, solver=solver, k=k, v0=random_init)
    u, v = svd_flip(u, v)
    idx = np.argsort(-s)
    u = u[:, idx]
    v = v[idx]
    s = s[idx]
    X_pca = u * s
    ev_ratio = (s ** 2).sum() / X.shape[0] / vars_.sum()

    return (X_pca, ev_ratio) if fit_transform else (v, ev_ratio)


def tsne(
    X,
    n_components=3,
    perplexity=30,
    early_exaggeration=12,
    learning_rate=1000,
):
    return manifold.TSNE(
        n_components, perplexity, early_exaggeration, learning_rate
    ).fit_transform(X)


def umap(X, n_neighbors=50, n_components=3, min_dist=0.0):
    embedding = UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist
    ).fit(X)
    return embedding, embedding.transform(X)


def labels_to_ids(series: pd.Series) -> list:
    labels_to_ids_dict = {label: idx for idx, label in enumerate(series.unique())}
    return (
        np.array(list(map(lambda label: labels_to_ids_dict[label], series))),
        labels_to_ids_dict,
    )


def plot(
    u,
    labels=None,
    colorscale="Viridis",
    backend="plotly",
    xrange=None,
    yrange=None,
    zrange=None,
    folded=False,
    folded_embedding=None,
):
    assert len(u.shape) == 2
    assert u.shape[1] in (2, 3)
    assert backend in ("plotly", "babyplots")

    if backend == "plotly":
        if u.shape[1] == 2:
            fig = go.Figure(
                data=go.Scattergl(
                    x=u[:, 0],
                    y=u[:, 1],
                    # z=u[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2, color=labels, colorscale=colorscale, opacity=0.9
                    ),
                )
            )
        elif u.shape[1] == 3:
            fig = go.Figure(
                data=go.Scatter3d(
                    x=u[:, 0],
                    y=u[:, 1],
                    z=u[:, 2],
                    mode="markers",
                    marker=dict(size=2, color=labels, colorscale=colorscale, opacity=0),
                )
            )
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=xrange),
                yaxis=dict(range=yrange),
                zaxis=dict(range=zrange),
            )
        )
    elif backend == "babyplots":
        if labels is None:
            labels = np.zeros(len(u))
        fig = Babyplot()
        options = dict(
            # shape='sphere',
            color_scale=colorscale,
            show_axes=[True, True, True],
            show_legend=True,
            folded=folded,
        )
        options["show_axes"] = [True] * u.shape[1]
        if folded is True:
            options["folded_embedding"] = folded_embedding
        fig.add_plot(u, "point_cloud", "values", labels, options)
    return fig


def hdbscan(X, min_cluster_size=50, cluster_selection_epsilon=0.5):
    return (
        HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
        .fit(X)
        .labels_
    )


def objective(dtrain, dval, yval, num_class):
    def func(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 600),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            learning_rate=trial.suggest_uniform("learning_rate", 0.01, 0.1),
            subsample=trial.suggest_uniform("subsample", 0.50, 1),
            colsample_bytree=trial.suggest_uniform("colsample_bytree", 0.5, 1),
            gamma=trial.suggest_int("gamma", 0, 10),
            tree_method="gpu_hist",
            gpu_id=0,
            objective="multi:softmax",
            num_class=num_class,
        )
        bst = xgb.train(params, dtrain)
        preds = bst.predict(dval)
        pred_labels = np.rint(preds)
        acc = accuracy_score(yval, pred_labels)
        return acc

    return func


def xgboost_hypertune(sparse_X, labels, num_class):
    X_train, X_val, y_train, y_val = train_test_split(
        sparse_X, labels, test_size=0.2, random_state=42
    )
    # generate sample weight
    weights = len(y_train) / y_train.value_counts()
    weights /= weights.sum()
    w_train = y_train.apply(lambda x: weights[x])
    w_val = y_val.apply(lambda x: weights[x])

    d_train = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    d_val = xgb.DMatrix(X_val, label=y_val, weight=w_val)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective(d_train, d_val, y_val, num_class), n_trials=10)
    return study


def fit_xgboost(sparse_X, labels, params, save_path):
    w = len(labels) / labels.value_counts()
    w /= w.sum()
    w = labels.apply(lambda x: w[x])
    bst = xgb.XGBClassifier(**params)
    print("====== Start training XGBoost ======")
    bst.fit(sparse_X, labels, sample_weight=w)  # this takes a looooooooooong time
    with open(save_path, "wb") as f:
        pickle.dump(bst, f)
    print("====== Finish training XGBoost ======")


def reorder(X):
    link, _ = clustering.hierarchical_clustering(X, thres=0.7, plot=False)
    return clustering.get_quasi_diag(link)


def standard_clustering(data_dir, skip_to_line, pca_k=1000, u=None):
    print("Reading...")
    X_normalized = normalize(data_dir, skip_to_line)  # sparse

    print("SVDing...")
    if u is None:
        X_pca, ev_ratio = svd_with_sparse(X_normalized, pca_k)
        print("Explained variance:", round(ev_ratio, 4))
    else:
        X_pca = X_normalized @ u
    print("UMAPing...")
    embedding, X_umap = umap(X_pca, n_components=3, n_neighbors=30, min_dist=0.0)
    labels = hdbscan(X_umap, min_cluster_size=100, cluster_selection_epsilon=0.5)
    print(pd.Series(labels).value_counts())
    plot(X_umap, labels)

    return X_normalized, X_pca, embedding, X_umap, labels


def magic_clustering(X, X_pca, pca_k, magic_k):
    print("MAGICing...")
    A = magic(X, magic_k, X_pca)
    X_magic = diffusion(A, X)
    print("PCAing...")
    pca = PCA(n_components=pca_k).fit(X_magic)
    print("Explained variance:", np.round(pca.explained_variance_ratio_.sum(), 4))
    X_magic_pca = pca.transform(X_magic)

    print("UMAPing")
    magic_embedding, X_magic_umap = umap(
        X_magic_pca, n_components=3, n_neighbors=30, min_dist=0.0
    )
    magic_labels = hdbscan(
        X_magic_umap, min_cluster_size=100, cluster_selection_epsilon=0.5
    )
    print(pd.Series(magic_labels).value_counts())
    plot(X_magic_umap, magic_labels)

    return X_magic, X_magic_pca, magic_embedding, X_magic_umap, magic_labels


def test_dl(model, X: np.ndarray, device: str, apply_pca: bool, pca_k: int):
    """
    Args:
        model: Your model.
        checkpoint: Path to the checkpoint of the model.
        data: Input data to your model. Should be a sparse matrix.
        device: Your model and data will be moved to this device.
    Return:
        X_pred: Prediction of input profile.
        X_umap:
    """
    print("Inferencing...")
    gc.collect()
    torch.cuda.empty_cache()
    # X_scaled = (X.toarray() - 0.5) / 0.5
    X = X.toarray()
    X = torch.from_numpy(X).float().to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(X)
        mask = torch.zeros_like(X).bool()
        loss_reconstruction = model.criterion_recon(X, preds)
        loss_ssl = model.criterion_ssl(
            preds.masked_fill(~mask, 0), X.masked_fill(~mask, 0)
        )
        loss = loss_reconstruction + model.hparams.ssl_weight * loss_ssl
        print(loss.item())
    X_pred = preds.cpu().numpy()
    gc.collect()
    torch.cuda.empty_cache()

    if apply_pca:
        print("PCAing...")
        pca = PCA(n_components=pca_k).fit(X_pred)
        print("Explained variance:", np.round(pca.explained_variance_ratio_.sum(), 4))
        X_pred_pca = pca.transform(X_pred)
        print("UMAPing...")
        magic_embedding, X_umap = umap(
            X_pred_pca, n_components=3, n_neighbors=10, min_dist=0.1
        )
    else:
        print("UMAPing...")
        magic_embedding, X_umap = umap(
            X_pred, n_components=3, n_neighbors=10, min_dist=0.1
        )
    plot(X_umap, labels=None)
    return X_pred, X_umap
