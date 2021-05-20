import numpy as np
from scipy import sparse as ss
from scipy import stats as sstats
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns
from rich import print
import tqdm


def draw_MSP(df) -> None:
    """
    Args:
        df: pandas.DataFrame. Pairwise distance.
    """
    G = nx.from_numpy_matrix(df.values)
    labels = df.columns.values
    labels = [s.replace("ft_", "") for s in df.columns]
    G = nx.relabel_nodes(G, dict(enumerate(labels)))
    T = nx.minimum_spanning_tree(G)

    fig = plt.figure(figsize=(20, 20))
    nx.draw_networkx(
        T,
        with_labels=True,
        font_size=9,
        cmap=plt.cm.coolwarm,
        pos=nx.kamada_kawai_layout(T),
        vmin=0,
        vmax=1,
    )
    plt.show()


def hierarchical_clustering(X, thres, plot=True):
    """
    Args:
        X: np.ndarray or pandas.DataFrame. Pairwise distance.
    """
    if len(X.shape) == 2:
        X = squareform(X)
    
    link = sch.linkage(X, "average")
    clusters = sch.fcluster(link, t=thres, criterion="distance")
    index = X.index if isinstance(X, pd.DataFrame) else np.arange(len(clusters))
    df_cluster = pd.DataFrame(
        dict(Cluster=clusters, Features=index)
    )
    if plot:
        fig = plt.figure(figsize=(20, 8))
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Feature")
        plt.ylabel("Distance")
        plt.hlines(thres, 0, 1320)
        dn = sch.dendrogram(link, leaf_rotation=90, leaf_font_size=11)
        plt.show()
    return link, df_cluster.groupby("Cluster").Features.apply(list).to_dict()


def get_quasi_diag(link):
    # Sort clustered items by distance
    indices = np.array([link[-1, 0], link[-1, 1]], dtype=int)
    num_items = int(link[-1, 3])  # number of original items
    while indices.max() >= num_items:
        new_indices = np.full((len(indices), 2), -1)
        new_indices[:, 0] = indices
        cluster_mask = indices >= num_items
        clusters = indices[cluster_mask] - num_items
        parent1, parent2 = link[clusters, 0], link[clusters, 1]
        new_indices[cluster_mask, 0] = parent1
        new_indices[cluster_mask, 1] = parent2
        new_indices = new_indices.flatten()
        indices = new_indices[new_indices > -1]
    return indices


def test(df1, df2, thres=0.5):
    link, _ = hierarchical_clustering(df2, thres)
    sort_idx = get_quasi_diag(link)
    df1_diag = df1.iloc[sort_idx, sort_idx]
    df2_diag = df2.iloc[sort_idx, sort_idx]

    fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(nrows=2, ncols=2, figsize=(23, 18))
    sns.heatmap(df1, ax=ax00, cmap="coolwarm")
    sns.heatmap(df2, ax=ax01, cmap="coolwarm")
    ax00.title.set_text("Correlation matrix")
    ax01.title.set_text("Distance matrix")
    sns.heatmap(df1_diag, ax=ax10, cmap="coolwarm")
    sns.heatmap(df2_diag, ax=ax11, cmap="coolwarm")
    ax10.title.set_text("Quasi-diagonal Correlation matrix")
    ax11.title.set_text("Quasi-diagonal Distance matrix")
    plt.show()


def num_bins(n_obs, corr=None):
    if corr is None:
        z = (8 + 324 * n_obs + 12 * (36 * n_obs + 729 * n_obs ** 2) ** 0.5) ** (1 / 3)
        b = round(z / 6 + 2 / (3 * z) + 1 / 3)
    else:
        b = round(2 ** -0.5 * (1 + (1 + 24 * n_obs / (1 - corr ** 2)) ** 0.5) ** 0.5)
    return b


def var_info(x, y, corr=None, norm=True):
    if (x == y).all():
        return 0
    b_xy = num_bins(x.shape[0], corr=corr)
    c_xy = np.histogram2d(x, y, b_xy)[0]
    i_xy = mutual_info_score(None, None, contingency=c_xy)
    h_x = sstats.entropy(np.histogram(x, b_xy)[0])
    h_y = sstats.entropy(np.histogram(y, b_xy)[0])
    v_xy = h_x + h_y - 2 * i_xy
    if norm:
        h_xy = h_x + h_y - i_xy  # joint
        v_xy /= h_xy  #
    return v_xy


def variation_of_info(df, corr):
    V = [
        var_info(df.iloc[:, i], df.iloc[:, j], corr.iloc[i, j])
        for i in tqdm(range(len(corr)))
        for j in range(i + 1, len(corr))
    ]
    return pd.DataFrame(squareform(V), index=df.index, columns=df.index)
