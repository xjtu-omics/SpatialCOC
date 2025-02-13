import scanpy as sc
import numpy as np
import scipy
import anndata
import sklearn
from typing import Optional
import os
import random
import torch
from torch.backends import cudnn

import warnings
warnings.filterwarnings('ignore')

def clr_normalize_each_cell(adata, inplace=True):
    """
    Normalizes count vector for each cell, i.e., for each row of .X using the Centered Log Ratio (CLR) method, suitable for protein modality.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to normalize.
    inplace : bool, optional (default=True)
        If True, modifies the input AnnData object in place.

    Returns
    -------
    adata : anndata.AnnData
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
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

def tfidf(X):
    """
    Applies TF-IDF normalization following the Seurat v3 approach.

    Parameters
    ----------
    X : ndarray or sparse matrix
        The input data matrix.

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

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    """
    Performs Latent Semantic Indexing (LSI) analysis following the Seurat v3 approach.

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

        sc.pp.highly_variable_genes(adata_modal_1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_modal_1, target_sum=1e4)
        sc.pp.log1p(adata_modal_1)

        adata_modal_1 =  adata_modal_1[:, adata_modal_1.var['highly_variable']]
        adata_modal_2 = clr_normalize_each_cell(adata_modal_2)

    if data_type == 'SPOTS':
        sc.pp.filter_genes(adata_modal_1, min_cells=10)

        sc.pp.highly_variable_genes(adata_modal_1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_modal_1, target_sum=1e4)
        sc.pp.log1p(adata_modal_1)
        sc.pp.scale(adata_modal_1)

        adata_modal_1 =  adata_modal_1[:, adata_modal_1.var['highly_variable']]
        adata_modal_2 = clr_normalize_each_cell(adata_modal_2)
        sc.pp.scale(adata_modal_2)

    if data_type == 'Spatial-epigenome-transcriptome':
        sc.pp.filter_genes(adata_modal_1, min_cells=10)
        sc.pp.filter_cells(adata_modal_1, min_genes=200)

        sc.pp.highly_variable_genes(adata_modal_1, flavor="seurat_v3", n_top_genes=3000)
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

def fix_seed(seed=2024):
    """
    Fixes the seed for reproducibility across different random number generators.

    Parameters
    ----------
    seed : int, optional (default=2024)
        The seed value to use.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def replace_extreme_values(arr, n=0.005):
    """
    Replaces extreme values in the expression data to optimize visualization effects.

    Parameters
    ----------
    arr : ndarray
        The input array.

    n : float, optional (default=0.005)
        The percentage of extreme values to be replaced.

    Returns
    -------
    arr_copy : ndarray
        The array with extreme values replaced.
    """
    # Ensure arr is a 1D array
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")

    # Calculate the number to replace
    num_to_replace = max(int(len(arr) * n), 1)
    
    # Sort the array
    sorted_arr = np.sort(arr)
    
    # Find the value at the n% percentile
    value_n_percentile = sorted_arr[num_to_replace - 1]
    
    # Find the value at the (100-n)% percentile
    value_100_n_percentile = sorted_arr[-num_to_replace]
    
    # Find the indices for the bottom n% and top n% values
    min_indices = np.argpartition(arr, num_to_replace)[:num_to_replace]
    max_indices = np.argpartition(arr, -num_to_replace)[-num_to_replace:]
    
    # Create a copy of the original array
    arr_copy = arr.copy()
    
    # Replace the extreme values
    arr_copy[min_indices] = value_n_percentile
    arr_copy[max_indices] = value_100_n_percentile
    
    return arr_copy