import numpy as np
import anndata

import pipeline


def gen_dataset(data_path: str, cache_dir: str = "./data") -> None:
    """
    Script for generating Tabula Muris dataset in hdf5 format, including a pre-processed
    RNA profilem and label ids. Also store label names and gene names in text format.
    """
    tubula_muris_h5ad = anndata.read_h5ad(data_path)
    X = tubula_muris_h5ad.X.copy()
    labels = tubula_muris_h5ad.obs.cell_ontology_class_reannotated
    X_normalized, good_genes, good_cells = pipeline.normalize(
        X,
        clip_q=99,
        gene_min_cells=50,
        gene_min_counts=100,
        cell_min_genes=100,
        cell_min_counts=1000,
        cell_max_counts=4_000_000,
        logic="mine",
        plot=False,
    )
    labels = labels[good_cells]
    ids, _ = pipeline.labels_to_ids(labels)
    anndata.AnnData(X_normalized).write_h5ad(f"{cache_dir}/data.h5ad")
    np.savetxt(f"{cache_dir}/label.txt", ids, delimiter="\n", fmt="%d")
    labels.to_csv(f"{cache_dir}/label_name.txt", index=False)
    tubula_muris_h5ad.var.to_csv(f"{cache_dir}/gene_name.txt", header=None)


if __name__ == "__main__":
    from rich.traceback import install

    install()

    data_path = "/home/tiankang/wusuowei/data/single_cell/MARS/tabula-muris-senis-facs_mars.h5ad"
    gen_dataset(data_path)
