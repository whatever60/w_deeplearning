import numpy as np
from scipy import sparse as ss
import pandas as pd
import datatable as dt
from sklearn import preprocessing

from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_distances
from umap import UMAP
from umap import umap_

import igraph as ig
import leidenalg
import louvain as lv
from hdbscan import HDBSCAN


def read_raw(count_file, cell_file, gene_file):
    df = dt.fread(count_file, skip_to_line=2, header=False).to_pandas()
    print(df.columns)
    cell_df = pd.read_csv(cell_file, header=None, names=["cell"], squeeze=True)
    gene_df = pd.read_csv(gene_file, header=None, names=["gene"], squeeze=True)
    df.columns = ["gene", "cell", "counts"]
    gene_num, cell_num, total_count = df.iloc[0]
    df.drop(index=0, inplace=True)
    df.cell -= 1
    df.gene -= 1
    X = ss.csr_matrix((df.counts, (df.cell, df.gene)), shape=(cell_num, gene_num))

    df = pd.DataFrame.sparse.from_spmatrix(
        X, index=cell_df, columns=gene_df, dtype="Sparse[int16]"
    )
    df.sort_index(inplace=True)
    return df


def qc(
    df,
    cell_min_count=0,
    cell_max_count=np.inf,
    cell_min_gene=0,
    cell_max_gene=np.inf,
    gene_min_count=0,
    gene_max_count=np.inf,
    gene_min_cell=0,
    gene_max_cell=np.inf,
):
    cell_num, gene_num = df.shape
    for i in cell_min_gene, cell_max_gene:
        if isinstance(i, float):
            i = int(i * gene_num)
    for i in gene_min_cell, gene_max_cell:
        if isinstance(i, float):
            i = int(i * cell_num)

    binarized_df = df > 0
    cell_count = df.sum(axis=1)
    cell_gene = binarized_df.sum(axis=1)

    gene_count = df.sum(axis=0)
    gene_cell = binarized_df.sum(axis=0)

    good_cell = (
        (cell_min_count <= cell_count)
        & (cell_count <= cell_max_count)
        & (cell_min_gene <= cell_gene)
        & (cell_gene <= cell_max_gene)
    )
    good_gene = (
        (gene_min_count <= gene_count)
        & (gene_min_count <= gene_max_count)
        & (gene_min_cell <= gene_cell)
        & (gene_cell <= gene_max_cell)
    )
    return df.iloc[good_cell, good_gene].copy().astype(pd.SparseDtype("int16"))


def normalize(X):
    X_l1 = preprocessing.normalize(X, norm='l1')
    X_l1_log1p = X.log1p()
    X_l1_log1p_scale = preprocessing.scale(X)
    return X  # , scaler.mean_, scaler.scale_


def pca(sparse_matrix):
    return ss.linalg.svds(sparse_matrix.T, k=sparse_matrix.shape[1] * 0.2)


# def embedding(method="tsne", **embedding_kwargs):
#     if method == "tsne":
#         embedding_kwargs = dict(
#             n_components=2, perplexity=30, early_exaggeration=12, learning_rate=200
#         ).update(embedding_kwargs)
#         method = manifold.TSNE

#     elif method == "umap":
#         method = UMAP

#     embedder = method(**embedding_kwargs)


def tsne(
    X,
    n_components=3,
    perplexity=30,
    early_exaggeration=12,
    learning_rate=1000,
):
    return manifold.TSNE(n_components, perplexity, early_exaggeration, learning_rate).fit_transform(X)


def umap(
    X,
    n_neighbors=15,
    n_components=2,
    min_dist=0.5,
    spread=1.0,
    n_epochs:int=500,
    lr:float=1.0,
    gamma:float=1.0,
    negative_sample_rate:int=5,
    init_pos:str="spectral",
    a: float = None,
    b: float = None,
    # neighbors_key: str = "neighbors",
):
    neighbors = compute_neighbors_umap(X, n_neighbors)
    if a is None or b is None:
        a, b = umap_.find_ab_params(spread, min_dist)
    if init_pos == 'paga':
        return NotImplementedError
    X_umap, _ = umap_.simplicial_set_embedding(
        X,
        neighbors,
        n_components,
        lr,
        a,
        b,
        gamma,
        negative_sample_rate,
        n_epochs,
        densmap=False,
        densmap_kwds={},
        output_dens=False,
    )
    return X_umap


def leiden(
    X,
    restrict_to=None,
    use_weights=True,
    adjacency=None,
    directed=True,
    resolution_parameter=1,
    n_iterations=-1,
    partition_type=leidenalg.RBConfigurationVertexPartition,
):
    if adjacency is None:
        raise NotImplementedError
    if restrict_to is not None:
        raise NotImplementedError
    g = get_igraph_from_adjacency(adjacency, directed=directed)
    if use_weights:
        weights = np.array(g.es['weight']).astype(np.float64)
    else:
        weights = None
    part = leidenalg.find_partition(g, partition_type, weights, n_iterations, resolution_parameter)
    return np.array(part.membership)


def louvain(
    X,
    resolution_parameter=None,
    restrict_to=None,
    adjacency=None,
    flavor='vtraag',
    directed=True,
    use_weights=False,
    partition_type=lv.RBConfigurationVertexPartition,
    # partition_kwargs={},
    # neighbors_key=None
):
    assert flavor in ('vtragg', 'igraph', 'rapids')
    if adjacency is None:
        raise NotImplementedError
    if restrict_to is not None:
        raise NotImplementedError
    if flavor in ('vtragg', 'igraph'):
        if flavor == 'igraph':
            assert resolution_parameter is None
            directed = False
        g = get_igraph_from_adjacency(adjacency, directed=directed)
        if use_weights:
            weights = np.array(g.es["weight"]).astype(np.float64)
        else:
            weights = None
        if flavor == 'vtragg':
            part = lv.find_partition(
                g,
                partition_type,
                resolution_parameter,
                weights,
            )
        else:
            part = g.community_multilevel(weights=weights)
        return np.array(part.membership)
    elif flavor == 'rapids':
        raise NotImplementedError
    else:
        raise NotImplementedError

def hdbscan(
    X,
    min_cluster_size=25,
    cluster_selection_epsilon=0.5
):
    cluster_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    cluster_model.fit(X)
    return cluster_model.labels_


def compute_neighbors_umap(
    X,
    n_neighbors,
    knn=True,
    method='umap',
    metric='euclidean',
    metric_kwds={},
    angular=False,
):
    assert method in ('umap', 'rapids', 'gauss')
    use_dense_distance = (metric == 'euclidean') or (not knn)
    use_dense_distance = False
    if use_dense_distance:
        distance = pairwise_distances(X, metric=metric, **metric_kwds)
        knn_indices, knn_distances = get_idices_distances_from_dense_matrix(distance, n_neighbors)
        if knn:
            distance = get_sparse_matrix_from_indices_distance_numpy(
                knn_indices, knn_distances, X.shape[0], n_neighbors
            )
    elif method == 'rapids':
        raise NotImplementedError
    else:
        if X.shape[0] < 4096:
            X = pairwise_distances(X, metric=metric, **metric_kwds)
            metric = 'precomputed'
        knn_indices, knn_distances, forest = umap_.nearest_neighbors(
            X,
            n_neighbors,
            metric,
            metric_kwds,
            angular,
        )
    if (not use_dense_distance) or method in ('umap', 'rapids'):
        distance, connectivity = compute_connectivities_umap(
            knn_indices,
            knn_distances,
            X.shape[0],
            n_neighbors
        )
    return distance, connectivity


def get_idices_distances_from_dense_matrix():
    raise NotImplementedError


def get_sparse_matrix_from_indices_distance_numpy():
    raise NotImplementedError


def compute_connectivities_umap():
    raise NotImplementedError


def compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_obs,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """

    X = ss.coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = umap_.fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    distances = get_sparse_matrix_from_indices_distances_umap(
        knn_indices, knn_dists, n_obs, n_neighbors
    )

    return distances, connectivities.tocsr()


def get_sparse_matrix_from_indices_distances_umap(
    knn_indices, knn_dists, n_obs, n_neighbors
):
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = ss.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()



def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except KeyError:
        pass 
    return g
