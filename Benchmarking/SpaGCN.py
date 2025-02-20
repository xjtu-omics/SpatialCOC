"""
Loading packages
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import SpaGCN as spg
import anndata as ad
import os
import numpy as np
import matplotlib.pyplot as plt
import anndata
os.environ['R_HOME'] = 'E:/R-4.3.1'
os.environ['R_USER'] = 'E:/anaconda/lib/site-packages/rpy2'
import random, torch
import sys
sys.path.append(r'D:/study/learning\spatial_transcriptome/papers\spatial_multi_omics-main')
from Model.utils import mclust_R
from Model.preprocess import preprocessing

"""
Loading and preprocessing data
"""
adata_modality_1 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_RNA.h5ad")
adata_modality_2 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_ADT.h5ad")

adata_modality_1, adata_modality_2 = preprocessing(adata_modality_1, adata_modality_2, 'SPOTS')

adata = anndata.AnnData(X=np.concatenate((adata_modality_1.X, adata_modality_2.X), axis=1))
adata.obsm['spatial'] = adata_modality_1.obsm['spatial']


"""
Running SpaGCN
"""
x_pixel=adata.obsm["spatial"][:, 0].tolist()
y_pixel=adata.obsm["spatial"][:, 1].tolist()
adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
p=10
#Find the l value given p
l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
#If the number of clusters known, we can use the spg.search_res() fnction to search for suitable resolution(optional)
n_clusters=6
#Set seed
r_seed=t_seed=n_seed=100
#Seaech for suitable resolution
res=spg.search_res(adata, adj, l, n_clusters, start=0.3, step=0.025, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
clf=spg.SpaGCN()
clf.set_l(l)
#Set seed
random.seed(r_seed)
torch.manual_seed(t_seed)
np.random.seed(n_seed)
#Run
clf.train(adata, adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
y_pred, prob=clf.predict()
adata.obs["pred"]= y_pred
adata.obs["pred"]=adata.obs["pred"].astype('category')


"""
Visualizing and storing results
"""
## visualization of SpaGCN
colors = [
    '#fdf0d5', '#f9c74f', '#83c5be', '#ee6055', '#99582a', '#3f5e66'
]
colors = [
    '#fdf0d5', '#f9c74f', '#83c5be', '#ee6055', '#99582a', '#3f5e66'
]

fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
sc.pl.embedding(adata, basis='spatial', color='pred', ax=ax, s=180, show=False, palette=colors)
ax.set_title(f'')
ax.set_xlabel('')
ax.set_ylabel('')
# remove legend
ax.get_legend().remove()
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Visualization/Mouse_Spleen/Replicate1/SpaGCN.png", dpi=500)
plt.savefig("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Visualization/Mouse_Spleen/Replicate1/SpaGCN.eps", dpi=500)

## Storing
# results = sc.read_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')
# results

# results.obs['SpaGCN'] = adata.obs["pred"].values
# results.write_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')