import re
import collections
import logging
import copy
from typing import Callable, List, Tuple, Dict, Union

import numpy as np
import scipy
import pandas as pd
import scanpy as sc
from anndata import AnnData
import intervaltree

import torch
from torch.utils.data import Dataset



class SingleCellDataset(Dataset):
    def __init__(
        self,
        # ------ For raw data reading ------
        fname,
        reader,  # sc.read_mtx
        data_raw,
        transpose,
        cell_info,
        gene_info,
        # ------ For filtering, sorting, annotating, autosome filtering and pooling ------
        filter_features: dict = {},
        filter_samples: dict = {},
        sort_by_pos: bool = False,  # False
        gtf_file: str = None,  # GTF file mapping genes to chromosomes, unused for ATAC
        autosomes_only: bool = False,  # False
        pool_genomic_interval = 0,  # ATAC only
        # ------ For QC ------
        binarize = False,
        filt_cell_min_counts = None,
        filt_cell_max_counts = None,
        filt_cell_min_genes=None,
        filt_cell_max_genes=None,
        filt_gene_min_counts=None,
        filt_gene_max_counts=None,
        filt_gene_min_cells=None,
        filt_gene_max_cells=None,
        calc_size_factors: bool = True,
        normalize: bool = True,
        log_trans: bool = True,
        clip: float = 0,
        # ------ And other transformations you want ------
        transforms: List[Callable] = [],
        # ------ For train-val-test split -------
        predefined_split=None,  # of type SingleCellDataset
        mode: str = 'all',
        data_split_by_cluster: str = "leiden",  # Specify as leiden
        valid_cluster_id: int = 0,  # Only used if data_split_by_cluster is on
        test_cluster_id: int = 1,
        data_split_by_cluster_log: bool = True,
        # ------ Postprocessing ------
        sample_y: bool = False,
        return_sf: bool = True,
        # ------ Caching ------
        cache_prefix: str = "",
        # ------ Other attrs ------
        split_by_chrom: bool = False,
        concat_outputs: bool = False,  # Instead of outputting a list of tensors, concat
        # high_confidence_clustering_genes: List[str] = [],  # used to build clustering
        x_dropout: bool = False,
        y_mode: str = "size_norm",
        selfsupervise: bool = True,
        return_pbulk: bool = False,
        cluster_res: float = 2.0,
    ):
        assert mode in ('all', 'skip')
        assert y_mode in ('size_norm', 'log_size_norm', 'raw_count', 'log_raw_count', 'x')
        # Step 1
        self.y_mode = y_mode
        self.split_by_chrom = split_by_chrom
        self.concat_output = concat_outputs
        self.x_dropout = x_dropout
        self.selfsupervise = selfsupervise
        self.return_pbulk = return_pbulk
        self.filter_features = filter_features
        self.cluster_res = cluster_res
        
        # Step 2: read in the expression matrix
        if data_raw is not None:
            self.data_raw = data_raw
        else:
            self.data_raw = reader(fname)
        if transpose:
            self.data_raw = self.data_raw.T
        
        # Step 2.5: filter
        self.data_raw = filter_adata(
            self.data_raw, filt_cells=filter_samples, filt_var=filter_features
        )

        # Step 3: attach obs/var annotations
        if cell_info is not None:
            # self.data_raw.obs = self.data_raw.obs.join(cell_info, how='left', sort=False)
            self.data_raw.obs = cell_info
        
        if gene_info is not None:
            self.data_raw.var = gene_info

        # Step 4: sort
        if sort_by_pos:
            genes_reordered, chroms_reordered = reorder_genes_by_pos(
                self.data_raw.var_names, gtf_file=gtf_file, return_chrom=True
            )
            self.data_raw = self.data_raw[:, genes_reordered]

        # Step 5: annotate
        self.annotate_chroms(gtf_file)

        # Step 6: autosome filtering
        if autosomes_only:
            '''Only leave autosome gene (filter allosome gene)'''

        # Step 7: sort
        # Sort by the observation names so we can combine datasets
        sort_order_idx = np.argsort(self.data_raw.obs_names)
        self.data_raw = self.data_raw[sort_order_idx, :]

        # Step 8: pooling
        # NOTE pooling occurs AFTER feature/observation filtering
        if pool_genomic_interval:
            self.pool_features(pool_genomic_interval=pool_genomic_interval)
            # Re-annotate because we have lost this information
            self.annotate_chroms(gtf_file)

        # Step 9: binarize
        # Preprocess the data now that we're done filtering
        if binarize:
            # If we are binarizing data we probably don't care about raw counts
            # self.data_raw.raw = self.data_raw.copy()  # Store original counts
            self.data_raw.X[self.data_raw.X.nonzero()] = 1  # .X here is a csr matrix
        
        # Step 10: QC
        annotate_basic_adata_metrics(self.data_raw)
        filter_adata_cells_and_genes(
            self.data_raw,
            filter_cell_min_counts=filt_cell_min_counts,
            filter_cell_max_counts=filt_cell_max_counts,
            filter_cell_min_genes=filt_cell_min_genes,
            filter_cell_max_genes=filt_cell_max_genes,
            filter_gene_min_counts=filt_gene_min_counts,
            filter_gene_max_counts=filt_gene_max_counts,
            filter_gene_min_cells=filt_gene_min_cells,
            filter_gene_max_cells=filt_gene_max_cells,
        )
        self.data_raw = normalize_count_table(  # Normalizes in place
            self.data_raw,
            size_factors=calc_size_factors,
            normalize=normalize,
            log_trans=log_trans,
        )

        # Step 11: clipping
        if clip > 0:
            assert isinstance(clip, float) and 0.0 < clip < 50.0
            clip_low, clip_high = np.percentile(
                self.data_raw.X.flatten(), [clip, 100.0 - clip]
            )
            assert (
                clip_low < clip_high
            ), f"Got discordant values for clipping ends: {clip_low} {clip_high}"
            self.data_raw.X = np.clip(self.data_raw.X, clip_low, clip_high)

        # Step 12: and more
        # Apply any final transformations
        if transforms:
            for trans in transforms:
                self.data_raw.X = trans(self.data_raw.X)

        # Step 13: verify again after all the preprocessing
        # Make sure the data is a sparse matrix
        if not isinstance(self.data_raw.X, scipy.sparse.csr_matrix):
            self.data_raw.X = scipy.sparse.csr_matrix(self.data_raw.X)

        # Step 14: train-val split
        self.data_split_to_idx = {}
        if predefined_split is not None:
            logging.info("Got predefined split, ignoring mode")
            # Subset items
            self.data_raw = self.data_raw[
                [
                    i
                    for i in predefined_split.data_raw.obs.index
                    if i in self.data_raw.obs.index
                ],
            ]
            assert (
                self.data_raw.n_obs > 0
            ), "No intersected obs names from predefined split"
            # Carry over cluster indexing
            self.data_split_to_idx = copy.copy(predefined_split.data_split_to_idx)
        elif mode != "skip":
            # Create dicts mapping string to list of indices
            if data_split_by_cluster:
                self.data_split_to_idx = self.split_train_valid_test_cluster(
                    clustering_key=data_split_by_cluster
                    if isinstance(data_split_by_cluster, str)
                    else "leiden",
                    valid_cluster={str(valid_cluster_id)},
                    test_cluster={str(test_cluster_id)},
                    data_split_by_cluster_log=data_split_by_cluster_log
                )
            else:
                self.data_split_to_idx = self.split_train_valid_test()
        else:
            logging.info("Got data split skip, skipping data split")
        self.data_split_to_idx["all"] = np.arange(len(self.data_raw))
        
        # Step 15: save size factors
        self.size_factors = (
            torch.from_numpy(self.data_raw.obs.size_factors.values, dtype=float)
            if return_sf
            else None
        )
        
        # Step 16: similarity
        self.cell_sim_mat = (
            euclidean_sim_matrix(self.size_norm_counts) if sample_y else None
        )  # Skip calculation if we don't need
    
        # Step 17: backing
        # Perform file backing if necessary
        data_raw_cache_fname = ""
        if cache_prefix:
            data_raw_cache_fname = cache_prefix + ".data_raw.h5ad"
            logging.info(f"Setting cache at {data_raw_cache_fname}")
            self.data_raw.filename = data_raw_cache_fname
            if hasattr(self, "_size_norm_counts"):
                size_norm_cache_name = cache_prefix + ".size_norm_counts.h5ad"
                logging.info(
                    f"Setting size norm counts cache at {size_norm_cache_name}"
                )
                self._size_norm_counts.filename = size_norm_cache_name
            if hasattr(self, "_size_norm_log_counts"):
                size_norm_log_cache_name = (
                    cache_prefix + ".size_norm_log_counts.h5ad"
                )
                logging.info(
                    f"Setting size log norm counts cache at {size_norm_log_cache_name}"
                )
                self._size_norm_log_counts.filename = size_norm_log_cache_name


    def annotate_chroms(self, gtf_file):
        """Annotates chromosome information on the var field, without the chr prefix"""
        feature_chroms = (
            get_chrom_from_intervals(self.data_raw.var_names)
            if list(self.data_raw.var_names)[0].startswith("chr")
            else get_chrom_from_genes(self.data_raw.var_names, gtf_file)
        )
        self.data_raw.var["chrom"] = feature_chroms

    def pool_features(self, pool_genomic_interval):
        n_obs = self.data_raw.n_obs
        if isinstance(pool_genomic_interval, int):
            if pool_genomic_interval > 0:
                # WARNING This will wipe out any existing var information
                idx, names = get_indices_to_combine(
                    list(self.data_raw.var.index), interval=pool_genomic_interval
                )
                data_raw_aggregated = combine_array_cols_by_idx(  # Returns np ndarray
                    self.data_raw.X,
                    idx,
                )
                data_raw_aggregated = scipy.sparse.csr_matrix(data_raw_aggregated)
                self.data_raw = AnnData(
                    data_raw_aggregated,
                    obs=self.data_raw.obs,
                    var=pd.DataFrame(index=names),
                )
            elif pool_genomic_interval < 0:
                assert (
                    pool_genomic_interval == -1
                ), f"Invalid value: {pool_genomic_interval}"
                # Pool based on proximity to genes
                data_raw_aggregated, names = combine_by_proximity(self.data_raw)
                self.data_raw = AnnData(
                    data_raw_aggregated,
                    obs=self.data_raw.obs,
                    var=pd.DataFrame(index=names),
                )
            else:
                raise ValueError(f"Invalid integer value: {pool_genomic_interval}")
        elif isinstance(pool_genomic_interval, (list, set, tuple)):
            idx = get_indices_to_form_target_intervals(
                self.data_raw.var.index, target_intervals=pool_genomic_interval
            )
            data_raw_aggregated = scipy.sparse.csr_matrix(
                combine_array_cols_by_idx(
                    self.data_raw.X,
                    idx,
                )
            )
            self.data_raw = AnnData(
                data_raw_aggregated,
                obs=self.data_raw.obs,
                var=pd.DataFrame(index=pool_genomic_interval),
            )
        else:
            raise TypeError(
                f"Unrecognized type for pooling features: {type(pool_genomic_interval)}"
            )
        assert self.data_raw.n_obs == n_obs

    def split_train_valid_test_cluster(
        self,
        clustering_key: str = "leiden",
        valid_cluster={"0"},
        test_cluster={"1"},
        data_split_by_cluster_log: bool = True,
    ) -> Dict[str, List[int]]:
        """
        Splits the dataset into appropriate split based on clustering
        Retains similarly sized splits as train/valid/test random from above
        """
        assert not valid_cluster.intersection(
            test_cluster
        ), f"Overlap between valid and test clusters: {valid_cluster} {test_cluster}"
        if clustering_key not in ["leiden", "louvain"]:
            raise ValueError(
                f"Invalid clustering key for data splits: {clustering_key}"
            )
        logging.info(
            f"Constructing {clustering_key} {'log' if data_split_by_cluster_log else 'linear'} clustered data split with valid test cluster {valid_cluster} {test_cluster}"
        )
        cluster_labels = (
            self.size_norm_log_counts.obs[clustering_key]
            if data_split_by_cluster_log
            else self.size_norm_counts.obs[clustering_key]
        )
        cluster_labels_counter = collections.Counter(cluster_labels.to_list())
        assert not valid_cluster.intersection(
            test_cluster
        ), "Valid and test clusters overlap"

        train_idx, valid_idx, test_idx = [], [], []
        for i, label in enumerate(cluster_labels):
            if label in valid_cluster:
                valid_idx.append(i)
            elif label in test_cluster:
                test_idx.append(i)
            else:
                train_idx.append(i)

        assert train_idx, "Got empty training split"
        assert valid_idx, "Got empty validation split"
        assert test_idx, "Got empty test split"
        data_split_idx = {}
        data_split_idx["train"] = train_idx
        data_split_idx["valid"] = valid_idx
        data_split_idx["test"] = test_idx
        return data_split_idx

    @property
    def size_norm_counts(self):
        """Computes and stores table of normalized counts w/ size factor adjustment and no other normalization"""
        if not hasattr(self, "_size_norm_counts"):
            self._size_norm_counts = self._set_size_norm_counts()
        assert self._size_norm_counts.shape == self.data_raw.shape
        return self._size_norm_counts
    
    def _set_size_norm_counts(self) -> AnnData:
        logging.info(f"Setting size normalized counts")
        raw_counts_anndata = AnnData(
            scipy.sparse.csr_matrix(self.data_raw.raw.X),
            obs=pd.DataFrame(index=self.data_raw.obs_names),
            var=pd.DataFrame(index=self.data_raw.var_names),
        )
        sc.pp.normalize_total(raw_counts_anndata, inplace=True)
        # After normalizing, do clustering
        preprocess_anndata(
            raw_counts_anndata,
            louvain_resolution=self.cluster_res,
            leiden_resolution=self.cluster_res,
        )
        return raw_counts_anndata

    @property
    def size_norm_log_counts(self):
        """Compute and store adata of counts with size factor adjustment and log normalization"""
        if not hasattr(self, "_size_norm_log_counts"):
            self._size_norm_log_counts = self._set_size_norm_log_counts()
        assert self._size_norm_log_counts.shape == self.data_raw.shape
        return self._size_norm_log_counts

    def _set_size_norm_log_counts(self) -> AnnData:
        retval = self.size_norm_counts.copy()  # Generates a new copy
        logging.info(f"Setting log-normalized size-normalized counts")
        # Apply log to it
        sc.pp.log1p(retval, chunked=True, copy=False, chunk_size=10000)
        preprocess_anndata(
            retval,
            louvain_resolution=self.cluster_res,
            leiden_resolution=self.cluster_res,
        )
        return retval

    def __len__(self):
        """Number of examples"""
        return self.data_raw.n_obs

    def __getitem__(self, i):
        # TODO compatibility with slices
        expression_data = (
            torch.from_numpy(ensure_arr(self.data_raw.X[i]).flatten()).type(
                torch.FloatTensor
            )
            if not self.split_by_chrom
            else self.get_chrom_split_features(i)
        )
        if self.x_dropout and not self.split_by_chrom:
            # Apply dropout to the X input
            raise NotImplementedError

        # Handle case where we are shuffling y a la noise2noise
        # Only use shuffled indices if it is specifically enabled and if we are doing TRAINING
        # I.e. validation/test should never be shuffled
        y_idx = (
            self.sample_similar_cell(i)
            if (self.sample_y and self.mode == "train")
            else i  # If not sampling y and training, return the same idx
        )
        if self.y_mode.endswith("raw_count"):
            target = torch.from_numpy(
                ensure_arr(self.data_raw.raw.var_vector(y_idx))
            ).type(torch.FloatTensor)
        elif self.y_mode.endswith("size_norm"):
            target = torch.from_numpy(self.size_norm_counts.var_vector(y_idx)).type(
                torch.FloatTensor
            )
        elif self.y_mode == "x":
            target = torch.from_numpy(
                ensure_arr(self.data_raw.X[y_idx]).flatten()
            ).type(torch.FloatTensor)
        else:
            raise NotImplementedError(f"Unrecognized y_mode: {self.y_mode}")
        if self.y_mode.startswith("log"):
            target = torch.log1p(target)  # scapy is also natural logaeritm of 1+x

        # Structure here is a series of inputs, followed by a fixed tuple of expected output
        retval = [expression_data]
        if self.return_sf:
            sf = self.size_factors[i]
            retval.append(sf)
        # Build expected truth
        if self.selfsupervise:
            if not self.return_pbulk:
                retval.append(target)
            else:  # Return both target and psuedobulk
                ith_cluster = self.data_raw.obs.iloc[i]["leiden"]
                pbulk = torch.from_numpy(
                    self.get_cluster_psuedobulk().var_vector(ith_cluster)
                ).type(torch.FloatTensor)
                retval.append((target, pbulk))
        elif self.return_pbulk:
            ith_cluster = self.data_raw.obs.iloc[i]["leiden"]
            pbulk = torch.from_numpy(
                self.get_cluster_psuedobulk().var_vector(ith_cluster)
            ).type(torch.FloatTensor)
            retval.append(pbulk)
        else:
            raise ValueError("Neither selfsupervise or retur_pbulk is specified")

        return tuple(retval)

    def get_per_chrom_feature_count(self) -> List[int]:
        """
        Return the number of features from each chromosome
        If we were to split a catted feature vector, we would split
        into these sizes
        """
        chrom_to_idx = self.get_chrom_idx()
        return [len(indices[0]) for _chrom, indices in chrom_to_idx.items()]

    def get_cluster_psuedobulk(self, mode="leiden", normalize=True):
        """
        Return a dictionary mapping each cluster label to the normalized psuedobulk
        estimate for that cluster
        If normalize is set to true, then we normalize such that each cluster's row
        sums to the median count from each cell
        """
        assert mode in self.data_raw.obs.columns
        cluster_labels = sorted(list(set(self.data_raw.obs[mode])))
        norm_counts = self.get_normalized_counts()
        aggs = []
        for cluster in cluster_labels:
            cluster_cells = np.where(self.data_raw.obs[mode] == cluster)
            pbulk = norm_counts.X[cluster_cells]
            pbulk_aggregate = np.sum(pbulk, axis=0, keepdims=True)
            if normalize:
                pbulk_aggregate = (
                    pbulk_aggregate
                    / np.sum(pbulk_aggregate)
                    * self.data_raw.uns["median_counts"]
                )
                assert np.isclose(
                    np.sum(pbulk_aggregate), self.data_raw.uns["median_counts"]
                )
            aggs.append(pbulk_aggregate)
        retval = AnnData(
            np.vstack(aggs),
            obs={mode: cluster_labels},
            var=self.data_raw.var,
        )
        return retval

    def get_chrom_idx(self) -> Dict[str, np.ndarray]:
        """Helper func for figuring out which feature indexes are on each chromosome"""
        chromosomes = sorted(
            list(set(self.data_raw.var["chrom"]))
        )  # Sort to guarantee consistent ordering
        chrom_to_idx = collections.OrderedDict()
        for chrom in chromosomes:
            chrom_to_idx[chrom] = np.where(self.data_raw.var["chrom"] == chrom)
        return chrom_to_idx

    def get_chrom_split_features(self, i):
        """Given an index, get the features split by chromsome, returning in chromosome-sorted order"""
        if self.x_dropout:
            raise NotImplementedError
        features = torch.from_numpy(
            ensure_arr(self.data_raw.X[i]).flatten()
        ).type(torch.FloatTensor)
        assert len(features.shape) == 1  # Assumes one dimensional vector of features

        chrom_to_idx = self.__get_chrom_idx()
        retval = tuple([features[indices] for _chrom, indices in chrom_to_idx.items()])
        if self.concat_outputs:
            retval = torch.cat(retval)
        return retval



class SingleCellDatasetSplit(Dataset):
    """
    Wraps SingleCellDataset to provide train/valid/test splits
    """

    def __init__(self, sc_dataset: SingleCellDataset, split: str) -> None:
        assert isinstance(sc_dataset, SingleCellDataset)
        self.dset = sc_dataset  # Full dataset
        self.split = split
        assert self.split in self.dset.data_split_to_idx
        logging.info(
            f"Created {split} data split with {len(self.dset.data_split_to_idx[self.split])} examples"
        )

    def __len__(self) -> int:
        return len(self.dset.data_split_to_idx[self.split])

    def __getitem__(self, index: int):
        return self.dset.get_item_data_split(index, self.split)

    # These properties facilitate compatibility with old code by forwarding some properties
    # Note that these are NOT meant to be modified
    def size_norm_counts(self) -> AnnData:
        indices = self.dset.data_split_to_idx[self.split]
        return self.dset.size_norm_counts[indices].copy()

    def data_raw(self) -> AnnData:
        indices = self.dset.data_split_to_idx[self.split]
        return self.dset.data_raw[indices].copy()

    def obs_names(self):
        indices = self.dset.data_split_to_idx[self.split]
        return self.dset.data_raw.obs_names[indices]


def read_gtf_gene_to_pos(
    fname,
    acceptable_types=None,
    addtl_attr_filters: dict = None,
    extend_upstream: int = 0,
    extend_downstream: int = 0,
):
    """
    Given a gtf file, read it in and return as a ordered dictionary mapping genes to genomic ranges
    Ordering is done by chromosome then by position
    """
    import collections
    import gzip
    # https://uswest.ensembl.org/info/website/upload/gff.html
    gene_to_positions = collections.defaultdict(list)
    gene_to_chroms = collections.defaultdict(set)

    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            if line.startswith(b"#"):
                continue
            line = line.decode()
            (
                chrom,
                entry_type,
                entry_class,
                start,
                end,
                score,
                strand,
                frame,
                attrs,
            ) = line.strip().split("\t")
            assert strand in ("+", "-")
            if acceptable_types and entry_type not in acceptable_types:
                continue
            attr_dict = dict(
                [t.strip().split(" ", 1) for t in attrs.strip().split(";") if t]
            )
            if addtl_attr_filters:
                tripped_attr_filter = False
                for k, v in addtl_attr_filters.items():
                    if k in attr_dict:
                        if isinstance(v, str):
                            if v != attr_dict[k].strip('"'):
                                tripped_attr_filter = True
                                break
                        else:
                            raise NotImplementedError
                if tripped_attr_filter:
                    continue
            gene = attr_dict["gene_name"].strip('"')
            start = int(start)
            end = int(end)
            assert (
                start <= end
            ), f"Start {start} is not less than end {end} for {gene} with strand {strand}"
            if extend_upstream:
                if strand == "+":
                    start -= extend_upstream
                else:
                    end += extend_upstream
            if extend_downstream:
                if strand == "+":
                    end += extend_downstream
                else:
                    start -= extend_downstream

            gene_to_positions[gene].append(start)
            gene_to_positions[gene].append(end)
            gene_to_chroms[gene].add(chrom)

    slist = []
    for gene, chroms in gene_to_chroms.items():
        if len(chroms) != 1:
            print(f"Got multiple chromosomes for gene {gene}: {chroms}, skipping")
            continue
        positions = gene_to_positions[gene]
        t = (chroms.pop(), min(positions), max(positions), gene)
        slist.append(t)

    retval = collections.OrderedDict()
    for chrom, start, stop, gene in slist:
        retval[gene] = (chrom, start, stop)
    return retval



def get_chrom_from_intervals(intervals, strip_chr: bool = True):
    """
    Given a list of intervals, return a list of chromosomes that those are on
    >>> get_chrom_from_intervals(['chr2:100-200', 'chr3:100-222'])
    ['2', '3']
    """
    retval = [interval.split(":")[0].strip() for interval in intervals]
    if strip_chr:
        retval = [chrom.strip("chr") for chrom in retval]
    return retval


def get_chrom_from_genes(genes, gtf_file):
    """
    Given a list of genes, return a list of chromosomes that those genes are on
    For missing: NA
    """
    gene_to_pos = read_gtf_gene_to_pos(gtf_file)
    retval = [gene_to_pos[gene][0] if gene in gene_to_pos else "NA" for gene in genes]
    return retval


def reorder_genes_by_pos(genes, gtf_file, return_genes, return_chroms):
    """
    Reorders list of genes by their genomic coordinate in the given gtf
    Args:
        return_genes:
            If True, Return the genes themselves in order. If False, return the indices needed to rearrange the genes in order.
        return_chroms:
            If True, also return corresponding chromosomes.
    """
    pass


def get_indices_to_combine(genomic_intervals, interval: int = 1000):
    """
    Given a list of *sorted* genomic intervals in string format e.g. ["chr1:100-200", "chr1:300-400"]
    Return a list of indices to combine to create new intervals of given size, as well as new interval
    strings
    """
    # First convert to a list of tuples (chr, start, stop)
    interval_tuples = [interval_string_to_tuple(x) for x in genomic_intervals]

    curr_chrom, curr_start, _ = interval_tuples[0]  # Initial valiues
    curr_indices, ret_indices, ret_names = [], [], []
    curr_end = curr_start + interval
    for i, (chrom, start, stop) in enumerate(interval_tuples):
        if (
            chrom != curr_chrom or stop > curr_end
        ):  # Reset on new chromosome or extending past interval
            ret_indices.append(curr_indices)
            ret_names.append(
                tuple_to_interval_string((curr_chrom, curr_start, curr_end))
            )
            curr_chrom, curr_start = chrom, start
            curr_end = curr_start + interval
            curr_indices = []
        assert start >= curr_start, f"Got funky coord: {chrom} {start} {stop}"
        assert stop > start
        curr_indices.append(i)

    ret_indices.append(curr_indices)
    ret_names.append(tuple_to_interval_string((curr_chrom, curr_start, curr_end)))

    return ret_indices, ret_names


def interval_string_to_tuple(x: str):
    """
    Convert the string to tuple
    >>> interval_string_to_tuple("chr1:100-200")
    ('chr1', 100, 200)
    >>> interval_string_to_tuple("chr1:1e+06-1000199")
    ('chr1', 1000000, 1000199)
    """
    tokens = x.split(":")
    assert len(tokens) == 2, f"Malformed interval string: {x}"
    chrom, interval = tokens
    if not chrom.startswith("chr"):
        logging.warn(f"Got noncanonical chromsome in {x}")
    start, stop = map(float, interval.split("-"))
    assert start < stop, f"Got invalid interval span: {x}"
    return (chrom, int(start), int(stop))


def tuple_to_interval_string(t: Tuple[str, int, int]) -> str:
    return f"{t[0]}:{t[1]}-{t[2]}"


def interval_strings_to_itree(
    interval_strings: List[str],
) -> Dict[str, intervaltree.IntervalTree]:
    """
    Given a list of interval strings, return an itree per chromosome
    The data field is the index of the interval in the original list
    """
    interval_tuples = [interval_string_to_tuple(x) for x in interval_strings]
    retval = collections.defaultdict(intervaltree.IntervalTree)
    for i, (chrom, start, stop) in enumerate(interval_tuples):
        retval[chrom][start:stop] = i
    return retval


def combine_by_proximity(
    arr, gtf_file, start_extension: int = 10000, stop_extension: int = 10000
):
    def find_nearest(query: tuple, arr):
        """Find the index of the item in array closest to query"""
        # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        start_distances = np.abs(query[0] - arr)
        stop_distances = np.abs(query[1] - arr)
        min_distances = np.minimum(start_distances, stop_distances)
        idx = np.argmin(min_distances)
        return idx

    if isinstance(arr, AnnData):
        d = arr.X if isinstance(arr.X, np.ndarray) else arr.X.toarray()
        arr = pd.DataFrame(
            d,
            index=arr.obs_names,
            columns=arr.var_names,
        )
    assert isinstance(arr, pd.DataFrame)

    gene_to_pos = read_gtf_gene_to_pos(
        gtf_file,
        acceptable_types=["protein_coding"],
        addtl_attr_filters={"gene_biotype": "protein_coding"},
    )
    genomic_ranges_to_gene = gene_pos_dict_to_range(gene_to_pos)
    genes_to_intervals = collections.defaultdict(list)  # Maps to the ith intervals
    unassigned_count = 0
    for i, g_interval in enumerate(arr.columns):
        chrom, g_range = g_interval.split(":")
        chrom_stripped = chrom.strip("chr")
        if chrom_stripped not in genomic_ranges_to_gene:
            print("Chromosome not found: {chrom}")

        start, stop = map(int, g_range.split("-"))
        assert start < stop, f"Got illegal genomic range: {g_interval}"
        start_extended, stop_extended = start - start_extension, stop + stop_extension

        overlapping_genes = list(
            genomic_ranges_to_gene[chrom_stripped][start_extended:stop_extended]
        )
        if overlapping_genes:
            if len(overlapping_genes) == 1:
                hit = overlapping_genes.pop()  # There is only one hit so we extract it
                hit_gene = hit.data
            else:  # Pick the closer hit
                hit_starts = np.array([h.begin for h in overlapping_genes])
                hit_ends = np.array([h.end for h in overlapping_genes])
                hit_pos_combined = np.concatenate((hit_starts, hit_ends))
                hit_genes = [h.data for h in overlapping_genes] * 2
                nearest_idx = find_nearest(
                    (start_extended, stop_extended), hit_pos_combined
                )
                hit_gene = hit_genes[nearest_idx]
            genes_to_intervals[hit_gene].append(i)
        else:
            unassigned_count += 1
    print(f"{unassigned_count}/{len(arr.columns)} peaks not assigned to a gene")
    genes = list(genes_to_intervals.keys())
    aggregated = combine_array_cols_by_idx(arr, [genes_to_intervals[g] for g in genes])
    return aggregated, genes


def combine_array_cols_by_idx(
    arr, idx, combine_func=np.sum
) -> scipy.sparse.csr_matrix:
    """Given an array and indices, combine the specified columns, returning as a csr matrix"""
    if isinstance(arr, np.ndarray):
        arr = scipy.sparse.csc_matrix(arr)
    elif isinstance(arr, pd.DataFrame):
        arr = scipy.sparse.csc_matrix(arr.to_numpy(copy=True))
    elif isinstance(arr, scipy.sparse.csr_matrix):
        arr = arr.tocsc()
    elif isinstance(arr, scipy.sparse.csc_matrix):
        pass
    else:
        raise TypeError(f"Cannot combine array cols for type {type(arr)}")

    new_cols = []
    for indices in idx:
        if not indices:
            next_col = scipy.sparse.csc_matrix(np.zeros((arr.shape[0], 1)))
        elif len(indices) == 1:
            next_col = scipy.sparse.csc_matrix(arr.getcol(indices[0]))
        else:  # Multiple indices to combine
            col_set = np.hstack([arr.getcol(i).toarray() for i in indices])
            next_col = scipy.sparse.csc_matrix(
                combine_func(col_set, axis=1, keepdims=True)
            )
        new_cols.append(next_col)
    new_mat_sparse = scipy.sparse.hstack(new_cols).tocsr()
    assert (
        len(new_mat_sparse.shape) == 2
    ), f"Returned matrix is expected to be 2 dimensional, but got shape {new_mat_sparse.shape}"
    # print(arr.shape, new_mat_sparse.shape)
    return new_mat_sparse


def gene_pos_dict_to_range(gene_pos_dict: dict, remove_overlaps: bool = False):
    """
    Converts the dictionary of genes to positions to a intervaltree
    of chromsomes to positions, each corresponding to a gene
    """
    retval = collections.defaultdict(
        intervaltree.IntervalTree
    )  # Chromosomes to genomic ranges
    genes = list(gene_pos_dict.keys())  # Ordered
    for gene in genes:
        chrom, start, stop = gene_pos_dict[gene]
        retval[chrom][start:stop] = gene
    if remove_overlaps:
        raise NotImplementedError
    return retval


def get_indices_to_form_target_intervals(
    genomic_intervals: List[str], target_intervals: List[str]
) -> List[List[int]]:
    """
    Given a list of genomic intervals in string format, and a target set of similar intervals
    Return a list of indices to combine to map into the target
    """
    source_itree = interval_strings_to_itree(genomic_intervals)
    target_intervals = [interval_string_to_tuple(x) for x in target_intervals]

    retval = []
    for chrom, start, stop in target_intervals:
        overlaps = source_itree[chrom].overlap(start, stop)
        retval.append([o.data for o in overlaps])
    return retval


def annotate_basic_adata_metrics(adata: AnnData) -> None:
    """Annotate with some basic metrics"""
    assert isinstance(adata, AnnData)
    adata.obs["n_counts"] = np.squeeze(np.asarray((adata.X.sum(1))))
    adata.obs["log1p_counts"] = np.log1p(adata.obs["n_counts"])
    adata.obs["n_genes"] = np.squeeze(np.asarray(((adata.X > 0).sum(1))))
    adata.var["n_counts"] = np.squeeze(np.asarray(adata.X.sum(0)))
    adata.var["log1p_counts"] = np.log1p(adata.var["n_counts"])
    adata.var["n_cells"] = np.squeeze(np.asarray((adata.X > 0).sum(0)))


def filter_adata_cells_and_genes(
    x: AnnData,
    filter_cell_min_counts=None,
    filter_cell_max_counts=None,
    filter_cell_min_genes=None,
    filter_cell_max_genes=None,
    filter_gene_min_counts=None,
    filter_gene_max_counts=None,
    filter_gene_min_cells=None,
    filter_gene_max_cells=None,
) -> None:
    """Filter the count table in place given the parameters based on actual data"""

    def ensure_count(value, max_value) -> int:
        """Ensure that the value is a count, optionally scaling to be so
        Useful when you use both absolute count value (e.g. cells should have at least 5 genes detected.) and ratio (e.g. genes should express in at least 1/10 of cells.) interchangeably
        """
        if value is None:
            return value  # Pass through None
        retval = value
        if isinstance(value, float):
            assert 0.0 < value < 1.0
            retval = int(round(value * max_value))
        assert isinstance(retval, int)
        return retval

    assert isinstance(x, AnnData)
    # Perform filtering on cells
    logging.info(f"Filtering {x.n_obs} cells")
    if filter_cell_min_counts is not None:
        sc.pp.filter_cells(
            x,
            min_counts=ensure_count(
                filter_cell_min_counts, max_value=np.max(x.obs["n_counts"])
            ),
        )
        logging.info(f"Remaining cells after min count: {x.n_obs}")
    if filter_cell_max_counts is not None:
        sc.pp.filter_cells(
            x,
            max_counts=ensure_count(
                filter_cell_max_counts, max_value=np.max(x.obs["n_counts"])
            ),
        )
        logging.info(f"Remaining cells after max count: {x.n_obs}")
    if filter_cell_min_genes is not None:
        sc.pp.filter_cells(
            x,
            min_genes=ensure_count(
                filter_cell_min_genes, max_value=np.max(x.obs["n_genes"])
            ),
        )
        logging.info(f"Remaining cells after min genes: {x.n_obs}")
    if filter_cell_max_genes is not None:
        sc.pp.filter_cells(
            x,
            max_genes=ensure_count(
                filter_cell_max_genes, max_value=np.max(x.obs["n_genes"])
            ),
        )
        logging.info(f"Remaining cells after max genes: {x.n_obs}")

    # Perform filtering on genes
    logging.info(f"Filtering {x.n_vars} vars")
    if filter_gene_min_counts is not None:
        sc.pp.filter_genes(
            x,
            min_counts=ensure_count(
                filter_gene_min_counts, max_value=np.max(x.var["n_counts"])
            ),
        )
        logging.info(f"Remaining vars after min count: {x.n_vars}")
    if filter_gene_max_counts is not None:
        sc.pp.filter_genes(
            x,
            max_counts=ensure_count(
                filter_gene_max_counts, max_value=np.max(x.var["n_counts"])
            ),
        )
        logging.info(f"Remaining vars after max count: {x.n_vars}")
    if filter_gene_min_cells is not None:
        sc.pp.filter_genes(
            x,
            min_cells=ensure_count(
                filter_gene_min_cells, max_value=np.max(x.var["n_cells"])
            ),
        )
        logging.info(f"Remaining vars after min cells: {x.n_vars}")
    if filter_gene_max_cells is not None:
        sc.pp.filter_genes(
            x,
            max_cells=ensure_count(
                filter_gene_max_cells, max_value=np.max(x.var["n_cells"])
            ),
        )
        logging.info(f"Remaining vars after max cells: {x.n_vars}")


def normalize_count_table(
    x: AnnData,
    size_factors: bool = True,
    log_trans: bool = True,
    normalize: bool = True,
) -> AnnData:
    """
    Normalize the count table using method described in DCA paper, performing operations IN PLACE
    rows correspond to cells, columns correspond to genes (n_obs x n_vars)
    s_i is the size factor per cell, total number of counts per cell divided by median of total counts per cell
    x_norm = zscore(log(diag(s_i)^-1 X + 1))

    Reference:
    https://github.com/theislab/dca/blob/master/dca/io.py

    size_factors - calculate and normalize by size factors
    top_n - retain only the top n features with largest variance after size factor normalization
    normalize - zero mean and unit variance
    log_trans - log1p scale data
    """
    assert isinstance(x, AnnData)
    if log_trans or size_factors or normalize:
        x.raw = x.copy()  # Store the original counts as .raw
    # else:
    #     x.raw = x

    if size_factors:
        logging.info("Computing size factors")
        n_counts = np.squeeze(
            np.array(x.X.sum(axis=1))
        )  # Number of total counts per cell
        # Normalizes each cell to total count equal to the median of total counts pre-normalization
        sc.pp.normalize_total(x, inplace=True,)
        # The normalized values multiplied by the size factors give the original counts
        x.obs["size_factors"] = n_counts / np.median(n_counts)
        x.uns["median_counts"] = np.median(n_counts)
    else:
        x.obs["size_factors"] = 1.0
        x.uns["median_counts"] = 1.0

    if log_trans:  # Natural logrithm
        logging.info("Log transforming data")
        sc.pp.log1p(
            x,
            chunked=True,
            copy=False,
            chunk_size=100000,
        )

    if normalize:
        logging.info("Normalizing data to zero mean unit variance")
        sc.pp.scale(x, zero_center=True, copy=False)

    return x


def euclidean_sim_matrix(mat: np.ndarray):
    """
    Given a matrix where rows denote observations, calculate a square matrix of similarities
    Larger values indicate greater similarity
    """
    assert (
        len(mat.shape) == 2
    ), f"Input must be 2 dimensiona, but got {len(mat.shape)} dimensions"
    if isinstance(mat, AnnData):
        mat = mat.X  # We only read data here so this is ok
    assert isinstance(
        mat, np.ndarray
    ), f"Could not convert input of type {type(mat)} into np array"
    n_obs = mat.shape[0]
    retval = np.zeros((n_obs, n_obs), dtype=float)

    for i in range(n_obs):
        for j in range(i):
            s = np.linalg.norm(mat[i] - mat[j], ord=None)
            retval[i, j] = s
            retval[j, i] = s
    retval = retval / (np.max(retval))
    # Up to this point the values here are distances, where smaller = more similar
    # for i in range(n_obs):
    #     retval[i, i] = 1.0
    # Set 0 to be some minimum distance (max similarity)
    retval = np.divide(1, retval, where=retval > 0)
    retval[retval == 0] = np.max(retval)
    retval[np.isnan(retval)] = np.max(retval)
    return retval



def preprocess_anndata(
    a: AnnData,
    neighbors_n_neighbors: int = 15,
    neighbors_n_pcs: Union[int, None] = None,
    louvain_resolution: float = 1.0,
    leiden_resolution: float = 1.0,
    seed: int = 0,
    use_rep: str = None,
    inplace: bool = True,
):
    """
    Preprocess the given anndata object to prepare for plotting. This occurs in place
    Performs dimensionality reduction, projection, and clustering
    """
    assert isinstance(a, AnnData)
    if not inplace:
        raise NotImplementedError
    assert a.shape[0] >= 50, f"Got input with too few dimensions: {a.shape}"
    sc.pp.pca(a)
    sc.tl.tsne(a, n_jobs=12, use_rep=use_rep)  # Representation defaults to X_pca
    sc.pp.neighbors(
        a, use_rep=use_rep, n_neighbors=neighbors_n_neighbors, n_pcs=neighbors_n_pcs
    )  # Representation defaults to X_pca
    # https://rdrr.io/cran/Seurat/man/RunUMAP.html
    sc.tl.umap(  # Does not have a rep, looks at neighbors
        a,
        maxiter=500,
        min_dist=0.3,  # Seurat default is 0.3, scanpy is 0.5
        spread=1.0,  # Seurat default is 1.0
        alpha=1.0,  # Seurat default starting learning rate is 1.0
        gamma=1.0,  # Seurate default repulsion strength is 1.0
        negative_sample_rate=5,  # Seurat default negative sample rate is 5
    )  # Seurat default is 200 for large datasets, 500 for small datasets
    if louvain_resolution > 0:
        sc.tl.louvain(  # Depends on having neighbors or bbknn run first
            a, resolution=louvain_resolution, random_state=seed
        )  # Seurat also uses Louvain
    else:
        logging.info("Skipping louvain clustering")
    if leiden_resolution > 0:
        sc.tl.leiden(  # Depends on having neighbors or bbknn first
            a, resolution=leiden_resolution, random_state=seed, n_iterations=2
        )  # R runs 2 iterations
    else:
        logging.info("Skipping leiden clustering")


def ensure_arr(x) -> np.ndarray:
    """Return x as a np.array"""
    if isinstance(x, np.matrix):
        return np.squeeze(np.asarray(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        return x.toarray()
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")


class GenomicInterval:
    pass


def filter_adata(
    adata: AnnData,
    filt_cells: Dict[str, List[str]] = {},
    filt_var: Dict[str, List[str]] = {},
) -> AnnData:
    """
    Filter the AnnData by the given requirements, filtering by cells first then var
    This is based on metadata and returns a copy
    """
    if filt_cells:
        keep_idx = np.ones(adata.n_obs)
        for k, accepted_values in filt_cells.items():
            assert k in adata.obs or k == "index"
            keys = adata.obs[k] if k != "index" else adata.obs.index

            if isinstance(accepted_values, str):
                is_acceptable = np.array([keys == accepted_values])
            elif isinstance(accepted_values, re.Pattern):
                is_acceptable = np.array(
                    [re.search(accepted_values, x) is not None for x in keys]
                )
            elif isinstance(accepted_values, (list, tuple, set, pd.Index)):
                is_acceptable = np.array([x in accepted_values for x in keys])
            else:
                raise TypeError(f"Cannot subset cells using {type(accepted_values)}")
            keep_idx = np.logical_and(is_acceptable.flatten(), keep_idx)
            logging.info(
                f"Filtering cells by {k} retains {np.sum(keep_idx)}/{adata.n_obs}"
            )
        adata = adata[keep_idx].copy()

    if filt_var:
        keep_idx = np.ones(len(adata.var_names))
        for k, accepted_values in filt_var.items():
            assert k in adata.var or k == "index"
            keys = adata.var[k] if k != "index" else adata.var.index

            if isinstance(accepted_values, str):
                is_acceptable = np.array([keys == accepted_values])
            elif isinstance(accepted_values, re.Pattern):
                is_acceptable = np.array(
                    [re.search(accepted_values, x) is not None for x in keys]
                )
            elif isinstance(accepted_values, (list, tuple, set, pd.Index)):
                is_acceptable = np.array([x in accepted_values for x in keys])
            elif isinstance(accepted_values, GenomicInterval):
                is_acceptable = np.array([accepted_values.overlaps(x) for x in keys])
            else:
                raise TypeError(f"Cannot subset features using {type(accepted_values)}")
            keep_idx = np.logical_and(keep_idx, is_acceptable.flatten())
            logging.info(
                f"Filtering vars by {k} retains {np.sum(keep_idx)}/{len(adata.var_names)}"
            )
        adata = adata[:, keep_idx].copy()

    return adata

