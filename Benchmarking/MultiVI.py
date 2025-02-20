"""
Loading packages
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import scanpy as sc
import numpy as np
import anndata as ad
import os
import numpy as np
import matplotlib.pyplot as plt
import scvi
print("Last run with scvi-tools version:", scvi.__version__)

os.environ['R_HOME'] = 'E:/R-4.3.1'
os.environ['R_USER'] = 'E:/anaconda/lib/site-packages/rpy2'

import sys
sys.path.insert(0, 'D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Model/')
from preprocess import preprocessing
from utils import mclust_R


"""
Loading and preprocessing data
"""
adata_modality_1 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_RNA.h5ad")
adata_modality_2 = sc.read_h5ad("D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/data/Mouse_Spleen_1/adata_ADT.h5ad")

adata_RNA, adata_Protein = preprocessing(adata_modality_1, adata_modality_2, 'SPOTS')


"""
Running MultiVI
"""
adata_RNA.var['modality'] = 'RNA'
adata_Protein.var['modality'] = 'Protein'

X = np.hstack([np.array(adata_RNA.X), np.array(adata_Protein.X)])
cell_name = list(adata_RNA.obs_names)
gene_name = list(adata_RNA.var_names) + list(adata_Protein.var_names)
modality = ['RNA'] * adata_RNA.n_vars + ['Protein'] * adata_Protein.n_vars

obs = pd.DataFrame(index=cell_name)
var = pd.DataFrame(index=gene_name)
adata_RNA_Protein = ad.AnnData(X=X, obs=obs, var=var)

adata_RNA_Protein.var['modality'] = modality
adata_RNA_Protein.obsm['spatial'] = adata_RNA.obsm['spatial']

adata_RNA_Protein.var_names_make_unique()

n = int(adata_RNA_Protein.n_obs) 
adata_RNA = adata_RNA_Protein[:n].copy()
adata_paired = adata_RNA_Protein[n:2*n].copy()
adata_Protein = adata_RNA_Protein[2*n:].copy()

adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired, adata_RNA, adata_Protein)
adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")

model = scvi.model.MULTIVI(
    adata_mvi,
    n_genes=(adata_mvi.var["modality"] == "RNA").sum(),
    n_regions=(adata_mvi.var["modality"] == "Protein").sum(),
)

nan_count = np.isnan(adata_mvi.X).sum()
print(f"Number of NaN values: {nan_count}")

model.train(lr=1e-6, weight_decay=1e-6, batch_size=512)

MULTIVI_LATENT_KEY = "X_multivi"
adata_mvi.obsm[MULTIVI_LATENT_KEY] = model.get_latent_representation()

## mclust algorithm doesn't work here, thus leiden algorithm is used instead
sc.pp.neighbors(adata_mvi, use_rep=MULTIVI_LATENT_KEY)
sc.tl.leiden(adata_mvi, key_added="clusters_leiden", resolution=0.03)


"""
Visualizing and storing the results
"""
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
colors = [
    '#fdf0d5', '#f9c74f', '#83c5be', '#99582a', '#3f5e66', '#ee6055'
]

sc.pl.embedding(adata_mvi, basis='spatial', color="clusters_leiden", ax=ax, s=150, show=False, palette=colors)
ax.set_title(f'')
ax.set_xlabel('')
ax.set_ylabel('')
# remove legend
ax.get_legend().remove()
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()

results = sc.read_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')
results
results.obs['MultiVI'] = adata_mvi.obs['clusters_leiden'].values
results.obsm['MultiVI'] = adata_mvi.obsm['X_multivi']
results.write_h5ad('D:/study/learning/spatial_transcriptome/papers/spatial_multi_omics-main/Results/Mouse_Spleen_Replicate1.h5ad')