"""
Loading packages
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scanpy as sc
import anndata
import os
import MultiMAP
import matplotlib.pyplot as plt
os.environ['R_HOME'] = 'E:/R-4.3.1'
os.environ['R_USER'] = 'E:/anaconda/lib/site-packages/rpy2'
import sys
sys.path.append(r'D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main')
from Model.utils import mclust_R
from Model.preprocess import preprocessing


"""
Loading and preprocessing data
"""
adata_modality_1 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_RNA.h5ad")
adata_modality_2 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_ADT.h5ad")

adata_modality_1, adata_modality_2 = preprocessing(adata_modality_1, adata_modality_2, 'SPOTS')

sc.pp.pca(adata_modality_1)
sc.pp.pca(adata_modality_2)


"""
Running MultiMAP
"""
adata = MultiMAP.Integration([adata_modality_1, adata_modality_2], ['X_pca', 'X_pca'])
mid_index = adata.shape[0] // 2
adata1 = adata[:mid_index]
adata2 = adata[mid_index:]
combined_X_multimap = np.concatenate((adata1.obsm['X_multimap'], adata2.obsm['X_multimap']), axis=1)
adata = anndata.AnnData(obs=adata1.obs, obsm={'X_multimap': combined_X_multimap})
mclust_R(adata, used_obsm='X_multimap', num_cluster=6)


"""
Restoring and visulizing results
"""
# adata_result = anndata.AnnData()
# adata_result.obs['MultiMAP'] = adata.obs['clusters_mclust']
# adata_result.obsm['MultiMAP'] = adata.obsm['X_multimap']
results = sc.read_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')
# results.obs['MultiMAP'] = adata_result.obs['MultiMAP'].values
# results.obsm['MultiMAP'] = adata_result.obsm['MultiMAP']
# results.write_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
colors = [
    '#fdf0d5', '#f9c74f', '#83c5be', '#99582a', '#3f5e66', '#ee6055'
]

sc.pl.embedding(results, basis='spatial', color='MultiMAP', ax=ax, s=150, show=False, palette=colors)
ax.set_title(f'')
ax.set_xlabel('')
ax.set_ylabel('')
# remove legend
ax.get_legend().remove()
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()