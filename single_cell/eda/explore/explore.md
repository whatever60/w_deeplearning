## scRNAseq_Benchmark_datatset

Source: [zenodo](https://doi.org/10.5281/zenodo.3357167)

### Inter-dataset - Brain

There are three sub-datasets: Human MTG, Mouse ALM, Mouse V1.

NOTE: These brain data are already merged for comparison, so they have the same numebr of columns (genes). Thus these are not the original whole expression matrix. For original full data, refer to [Allen Brain Map](https://portal.brain-map.org/atlases-and-data/rnaseq) or [GSE115746](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746).

- Human MTG: 14055

  QC:

  ![brain_humanmtg_qc](./imgs/Brain_HumanMTG_qc.jpg)

  Label distribution:

  ![brain_humanmtg_label](./imgs/Brain_HumanMTG_label_dist.jpg)

  UMAP embeddings:

  ![brain_humanmtg_embedding](./imgs/Brain_HumanMTG_noqc_umap.jpg)

  ![brain_humanmtg_embedding](./imgs/Brain_HumanMTG_noqc_umap_coarse.jpg)

  ![brain_humanmtg_embedding](./imgs/Brain_HumanMTG_noqc_umap_refined.jpg)


- Mouse ALM: 8128

  QC:

  ![brain_mousealm_qc](./imgs/Brain_MouseALM_qc.jpg)

  Label distribution:

  ![brain_mousealm_label](./imgs/Brain_MouseALM_label_dist.jpg)

  UMAP embeddings:

  ![brain_mousealm_embedding](./imgs/Brain_MouseALM_noqc_umap.jpg)

  ![brain_mousealm_embedding](./imgs/Brain_MouseALM_noqc_umap_coarse.jpg)

  ![brain_mousealm_embedding](./imgs/Brain_MouseALM_noqc_umap_refined.jpg)

- Mouse V1: 12552

  QC:

  ![brain_mousev1_qc](./imgs/Brain_MouseV1_qc.jpg)

  Label distribution:

  ![brain_mousev1_label](./imgs/Brain_MouseV1_label_dist.jpg)

  UMAP embeddings:

  ![brain_mousev1_embedding](./imgs/Brain_MouseV1_noqc_umap.jpg)

  ![brain_mousev1_embedding](./imgs/Brain_MouseV1_noqc_umap_coarse.jpg)

  ![brain_mousev1_embedding](./imgs/Brain_MouseV1_noqc_umap_refined.jpg)

### Inter-dataset - CellBench

This dataset is a concatenation of the CellBench 10X and CEL-Seq2 dataset.

- 10X: 3803 

  QC:

  ![cellbench_10x_qc](./imgs/CellBench_10x_qc.jpg)

  Label distribution:

  ![cellbench_10x_label](./imgs/CellBench_10x_label_dist.jpg)

  UMAP embeddings:

  ![cellbench_10x_label](./imgs/CellBench_10x_noqc_umap.jpg)

  ![cellbench_10x_label](./imgs/CellBench_10x_noqc_umap_coarse.jpg)

- CelSeq2: 570
  
  QC:

  ![cellbench_celseq2_qc](./imgs/CellBench_celseq2_qc.jpg)

  Label distribution:

  ![cellbench_celseq2_label](./imgs/CellBench_celseq2_label_dist.jpg)

  UMAP embeddings:

  ![cellbench_celseq2_label](./imgs/CellBench_celseq2_noqc_umap.jpg)

  ![cellbench_celseq2_label](./imgs/CellBench_celseq2_noqc_umap_coarse.jpg)


