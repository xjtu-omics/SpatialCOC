"""
Loading packages
"""
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import os
os.environ['R_HOME'] = 'E:/R-4.3.1'
os.environ['R_USER'] = 'E:/anaconda/lib/site-packages/rpy2'
import sys
sys.path.insert(0, 'D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Model/')
from preprocess import preprocessing

import matplotlib.pyplot as plt

from SpatialGlue.preprocess import pca, construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

"""
Loading and preprocessing data, taking mouse spleen dataset for example
"""
adata_modality_1 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_RNA.h5ad")
adata_modality_2 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_ADT.h5ad")

## Preprocessing
adata_modality_1, adata_modality_2 = preprocessing(adata_modality_1, adata_modality_2, 'SPOTS')


"""
Running SpatialGlue
"""
data_type = 'SPOTS'
adata_modality_1.obsm['feat'] = pca(adata_modality_1, n_comps=adata_modality_2.n_vars-1)
adata_modality_2.obsm['feat'] = pca(adata_modality_2, n_comps=adata_modality_2.n_vars-1)

data = construct_neighbor_graph(adata_modality_1, adata_modality_2, datatype=data_type)

# define model

model = Train_SpatialGlue(data, datatype=data_type)

# train model
output = model.train()

adata = adata_modality_1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()
adata.obsm['alpha'] = output['alpha']
adata.obsm['alpha_omics1'] = output['alpha_omics1']
adata.obsm['alpha_omics2'] = output['alpha_omics2']

from SpatialGlue.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
clustering(adata, key='SpatialGlue', add_key='SpatialGlue', n_clusters=6, method=tool, use_pca=True)


"""
Visualizing and storing results
"""
# Visualizing
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
colors = [
    '#fdf0d5', '#f9c74f', '#83c5be', '#99582a', '#3f5e66', '#ee6055'
]

sc.pl.embedding(adata, basis='spatial', color='SpatialGlue', ax=ax, s=150, show=False, palette=colors)
ax.set_title(f'')
ax.set_xlabel('')
ax.set_ylabel('')
# remove legend
ax.get_legend().remove()
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()

## Restoring
# results = sc.read_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')
# results.obs['SpatialGlue'] = adata.obs['SpatialGlue']
# results.obsm['SpatialGlue'] = adata.obsm['SpatialGlue']
# results.write_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')