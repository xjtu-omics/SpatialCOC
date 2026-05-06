# SpatialCOC: Results Reproduction and Visualization

<div style="background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 12px 16px; margin: 16px 0; border-radius: 0 4px 4px 0;">
  <strong style="color: #1976D2;">Note</strong><br>
  If you want to reproduce the results described in this document using the relevant code provided, please <strong style="color: #0d47a1; background-color: #bbdefb; padding: 2px 4px; border-radius: 3px;">set the corresponding paths</strong> according to your environment.
</div>

## ✔️Data Access

All result files are available on Zenodo. Please download and extract to the `./Results` subdirectory:

> 🔗 

## 📁Analysis Folder

This folder contains three sections:

- `Single_Modality_Analyses` subfolder: Contains mono-modal analysis pipelines applied to each real-world dataset;
- `Ablation_Experiments.ipynb`: Implements four ablation variants, each evaluated on four simulated spatial patterns, with performance in spatial domain identification compared against the full model;
- `Sensitive_Analysises.ipynb`: Assesses the impact of input dimensionality (original features vs. varying numbers of principal components) on model outputs.

## 🚀Benchmarking Folder

Implementation code for seven methods for comparison with SpatialCOC. Apply the same preprocessing procedure, and then run each method’s program following its official guidelines.

|   Method    |        Category         |                          Reference                           |
| :---------: | :---------------------: | :----------------------------------------------------------: |
| SpatialGlue |   Spatial multi-omics   | [Source](https://www.nature.com/articles/s41592-024-02316-4) |
|   COSMOS    |   Spatial multi-omics   | [Source](https://www.nature.com/articles/s41467-024-55204-y) |
| Seurat WNN  | single-cell multi-omics | [Source](https://www.cell.com/cell/fulltext/S0092-8674(21)00583-3) |
|   MultiVI   | single-cell multi-omics | [Source](https://www.nature.com/articles/s41592-023-01909-9) |
|  MultiMAP   | single-cell multi-omics | [Source](https://link.springer.com/article/10.1186/s13059-021-02565-y) |
|   STAGATE   |  spatial transcriptome  | [Source](https://www.nature.com/articles/s41467-022-29439-6) |
|   SpaGCN    |  spatial transcriptome  | [Source](https://www.nature.com/articles/s41592-021-01255-8) |

## 📈.h5ad Files

All results from benchmark methods and SpatialCOC are provided. The `.h5ad` file, a storage format for AnnData objects, is used to store the following core results:

|         Field          |    Type     |                Description                 |
| :--------------------: | :---------: | :----------------------------------------: |
| anndata.obs['method']  | categorical | Clustering assignments per method/modality |
| anndata.obsm['method'] |   ndarray   |   Low-dimensional integrative embeddings   |

## 🖼️Visualization Folder

The `Visualization/` directory contains executable scripts for reproducing all figures from the SpatialCOC publication.

---

## 📩Contact

Mingxuan Li: 3123154029@stu.xjtu.edu.cn

## 📕Reference

Li, M., Sun, P., Luo, Y. *et al.* SpatialCOC: an integrative framework for spatial continuous mapping and cross-omics correction in spatial multi-omics data. *Nat Commun* (2026). https://doi.org/10.1038/s41467-026-71882-2
