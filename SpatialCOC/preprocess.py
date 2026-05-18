#!/usr/bin/env python
"""
File: preprocess.py
Description:
    Preprocessing utilities for spatial multi-omics data.

Copyright (C) 2026 XJTU-Yelab
License: GNU GPLv3
Version: 1.0.0
Created on: 2026-03-25
Last modified: 2026-03-25
Author(s):
    - Mingxuan Li (3123154029@stu.xjtu.edu.cn)
"""

"""
================
Loading Packages
================
"""
import scanpy as sc
import numpy as np
import scipy
import anndata
import sklearn

import warnings
warnings.filterwarnings('ignore')

"""
=================
Utility Functions
=================
"""
def clr_normalize_each_cell(adata, inplace=True):
    """
    Apply Centered Log Ratio (CLR) normalization to protein expression data.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing protein expression data to normalize.
    inplace : bool, default=True
        Whether to modify the input AnnData object in place (True) or return a copy (False).

    Returns
    -------
    adata : AnnData
        The normalized AnnData object.
    """
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

def tfidf(X):
    """
    Apply TF-IDF (Term Frequency-Inverse Document Frequency) normalization to chromatin peak accessibility data.

    Parameters
    ----------
    X : ndarray or sparse matrix
        The input data matrix where rows represent cells and columns represent features.

    Returns
    -------
    X_tfidf : ndarray or sparse matrix
        The TF-IDF normalized data matrix.
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def lsi(adata, n_components=20, use_highly_variable=None, **kwargs):
    """
    Apply Latent Semantic Indexing (LSI) to reduce the dimensionality of chromosome accessibility data.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to perform LSI on.
    n_components : int, optional (default=20)
        The number of components to retain.
    use_highly_variable : bool, optional (default=None)
        If True, uses only highly variable genes for LSI. If None, checks if 'highly_variable' is in adata.var.

    Returns
    -------
    None
        The LSI results are stored in adata.obsm['X_lsi'].
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:,1:]


def _mark_highly_variable_genes(adata, n_top_genes=3000):
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        return
    except ImportError as exc:
        print(f"seurat_v3 highly_variable_genes unavailable, falling back to variance ranking: {exc}")

    X = adata.X
    if scipy.sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        variances = mean_sq - mean ** 2
    else:
        variances = np.asarray(X).var(axis=0)

    n_top = min(n_top_genes, adata.n_vars)
    top_idx = np.argsort(variances)[-n_top:]
    adata.var["highly_variable"] = False
    adata.var.iloc[top_idx, adata.var.columns.get_loc("highly_variable")] = True

"""
==========
Public API
==========
"""
def preprocessing(adata_modal_1, adata_modal_2, data_type):
    """
    Preprocesses data for different data types.

    Parameters:
    - adata_modal_1: AnnData object for the first modality (RNA).
    - adata_modal_2: AnnData object for the second modality (Protein or ATAC).
    - data_type: Type of data, one of 'Stereo-CITE-seq', 'SPOTS', 'Spatial-epigenome-transcriptome'.

    Returns:
    - adata_modal_1: Preprocessed AnnData object for the first modality.
    - adata_modal_2: Preprocessed AnnData object for the second modality.
    """

    valid_data_types = ['Stereo-CITE-seq', 'SPOTS', 'Spatial-epigenome-transcriptome']
    if data_type not in valid_data_types:
        print("Invalid data type provided. Please provide one of the following data types:\n 'Stereo-CITE-seq' for mouse thymus slices, \n 'SPOTS' for mouse spleen slices, \n 'Spatial-epigenome-transcriptome' for mouse brain slices.")
        return None, None

    adata_modal_1.var_names_make_unique()
    adata_modal_2.var_names_make_unique()

    if data_type == 'Stereo-CITE-seq':
        sc.pp.filter_genes(adata_modal_1, min_cells=10)
        sc.pp.filter_cells(adata_modal_1, min_genes=80)

        sc.pp.filter_genes(adata_modal_2, min_cells=50)

        adata_modal_2 = adata_modal_2[adata_modal_1.obs_names].copy()

        _mark_highly_variable_genes(adata_modal_1, n_top_genes=3000)
        sc.pp.normalize_total(adata_modal_1, target_sum=1e4)
        sc.pp.log1p(adata_modal_1)

        adata_modal_1 =  adata_modal_1[:, adata_modal_1.var['highly_variable']]
        adata_modal_2 = clr_normalize_each_cell(adata_modal_2)

    if data_type == 'SPOTS':
        sc.pp.filter_genes(adata_modal_1, min_cells=10)

        _mark_highly_variable_genes(adata_modal_1, n_top_genes=3000)
        sc.pp.normalize_total(adata_modal_1, target_sum=1e4)
        sc.pp.log1p(adata_modal_1)
        sc.pp.scale(adata_modal_1)

        adata_modal_1 =  adata_modal_1[:, adata_modal_1.var['highly_variable']]

        adata_modal_2 = adata_modal_2[adata_modal_1.obs_names].copy()
        adata_modal_2 = clr_normalize_each_cell(adata_modal_2)
        sc.pp.scale(adata_modal_2)

    if data_type == 'Spatial-epigenome-transcriptome':
        sc.pp.filter_genes(adata_modal_1, min_cells=10)
        sc.pp.filter_cells(adata_modal_1, min_genes=200)

        _mark_highly_variable_genes(adata_modal_1, n_top_genes=3000)
        sc.pp.normalize_total(adata_modal_1, target_sum=1e4)
        sc.pp.log1p(adata_modal_1)
        sc.pp.scale(adata_modal_1)

        adata_modal_1 =  adata_modal_1[:, adata_modal_1.var['highly_variable']]

        adata_modal_2 = adata_modal_2[adata_modal_1.obs_names].copy()
        lsi(adata_modal_2, use_highly_variable=False, n_components=51)
        
    print(data_type, "data preprocessing have done!")
    print(f"Dimensions after preprocessed adata_modal_1: {adata_modal_1.shape}")
    print(f"Dimensions after preprocessing adata_modal_2: {adata_modal_2.shape}")
    
    return adata_modal_1, adata_modal_2
