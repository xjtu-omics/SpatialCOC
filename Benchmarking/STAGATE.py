"""
Loading packages
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scanpy as sc
import STAGATE
from scipy import sparse
import os
os.environ['R_HOME'] = 'E:/R-4.3.1'
os.environ['R_USER'] = 'E:/anaconda/lib/site-packages/rpy2'
from anndata import AnnData

import sys
sys.path.insert(0, 'D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Model/')
from preprocess import preprocessing

"""
Loading and preprocessing data
"""
adata_modality_1 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_RNA.h5ad")
adata_modality_2 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_ADT.h5ad")

adata_modality_1.var_names_make_unique()
adata_modality_2.var_names_make_unique()

adata_modality_1, adata_modality_2 = preprocessing(adata_modality_1, adata_modality_2, 'SPOTS')

combined_X = np.concatenate((adata_modality_1.X.toarray(), adata_modality_2.X), axis=1)

adata = AnnData(X=combined_X)
adata.obsm['spatial'] = adata_modality_1.obsm['spatial']


"""
Running STAGATE
"""
STAGATE.Cal_Spatial_Net(adata, rad_cutoff=2)
STAGATE.Stats_Spatial_Net(adata)

adata.X = sparse.csr_matrix(adata.X)
adata = STAGATE.train_STAGATE(adata, alpha=0)

sc.pp.neighbors(adata, use_rep='STAGATE')
sc.tl.umap(adata)
adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=6)


"""
Visulizing and storing results
"""
# Visulizing
import matplotlib.pyplot as plt
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
## replicate 1
colors = [
    '#fdf0d5', '#f9c74f', '#ee6055', '#99582a', '#3f5e66', '#83c5be'
]
## replicate 2
colors = [
    '#3f5e66', '#83c5be', '#99582a', '#f9c74f', '#fdf0d5', '#ee6055'
]
sc.pl.embedding(adata, basis='spatial', color='mclust', ax=ax, s=150, show=False, palette=colors)
ax.set_title(f'')
ax.set_xlabel('')
ax.set_ylabel('')
# remove legend
ax.get_legend().remove()
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()

# Storing
results = sc.read_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')
results.obs['STAGATE'] = adata.obs['mclust'].values
results.obsm['STAGATE'] = adata.obsm['STAGATE']
results.write_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')