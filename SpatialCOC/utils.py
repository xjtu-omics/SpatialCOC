#!/usr/bin/env python
"""
File: utils.py
Description:
    Utility functions for SpatialCOC.

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
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.sparse import issparse
import os
import anndata as ad
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import random
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

"""
=================
Clustering Method
=================
"""
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='X_pca', random_seed=2024):
    """
    Performs clustering using mclust_R, which requires setting up the R environment.

    Parameters
    ----------
    adata : object
        The data object containing the observation data.

    num_cluster : int
        The number of clusters.

    modelNames : str, optional (default='EEE')
        The model names for mclust clustering.

    used_obsm : str, optional (default='X_pca')
        The observation slot to use for clustering.

    random_seed : int, optional (default=2024)
        The random seed for reproducibility.

    Returns
    -------
    adata : object
        The data object with clustering results added.
    """
    np.random.seed(random_seed)
    force_sklearn = os.environ.get("SPATIALCOC_FORCE_SKLEARN_CLUSTERING") == "1"
    try:
        if force_sklearn:
            raise RuntimeError("SPATIALCOC_FORCE_SKLEARN_CLUSTERING=1")
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri

        robjects.r.library("mclust")
        rpy2.robjects.numpy2ri.activate()
        r_random_seed = robjects.r['set.seed']
        r_random_seed(random_seed)
        rmclust = robjects.r['Mclust']

        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
        mclust_res = np.array(res[-2])
    except Exception as exc:
        print(f"mclust_R fallback to sklearn GaussianMixture because R mclust is unavailable: {exc}")
        X = np.asarray(adata.obsm[used_obsm])
        try:
            mclust_res = GaussianMixture(
                n_components=num_cluster,
                covariance_type="full",
                random_state=random_seed,
            ).fit_predict(X) + 1
        except Exception:
            mclust_res = KMeans(
                n_clusters=num_cluster,
                random_state=random_seed,
                n_init=20,
            ).fit_predict(X) + 1

    adata.obs['clusters_mclust'] = mclust_res
    adata.obs['clusters_mclust'] = adata.obs['clusters_mclust'].astype('int')
    adata.obs['clusters_mclust'] = adata.obs['clusters_mclust'].astype('category')

    return adata

"""
=========================
Simulated Data Generation
=========================
"""
def generate_noise(X, mean, std, mode, dropout_rate):
    """
    Generates noise to simulate data Dropout and random Gaussian perturbations.

    Parameters
    ----------
    X : ndarray or sparse matrix
        The input data.

    mean : float
        The mean of the Gaussian noise.

    std : float
        The standard deviation of the Gaussian noise.

    mode : str
        The mode of noise generation, either 'gaussian' or 'dropout'.

    dropout_rate : float
        The dropout rate for the 'dropout' mode.

    Returns
    -------
    X_noised : ndarray
        The noisy data.
    """
    # Convert sparse matrix to dense matrix
    X_dense = X.toarray() if issparse(X) else X
    # Create a copy of X_noised and add noise to non-zero elements
    X_noised = X_dense.copy()
    if mode == 'gaussian':
        # Create a noise matrix
        gaussian_noise = np.random.normal(mean, std, X_dense.shape)
        # Add Gaussian noise to non-zero elements
        X_noised[X_dense != 0] += gaussian_noise[X_dense != 0]
        # Set elements in X_noised that are less than 0 to 1e-10
        X_noised = np.where(X_noised < 0, 1e-10, X_noised)
    elif mode == 'dropout':
        dropout_mask = np.ones(X_dense.shape, dtype=bool)
        # Indices of non-zero elements
        non_zero_indices = X_dense != 0
        # Generate a random array with the length equal to the number of non-zero elements
        random_array = np.random.rand(np.count_nonzero(X_dense))
        # Set non-zero elements to False if the random value is less than dropout_rate
        dropout_mask[non_zero_indices] = (random_array >= dropout_rate)
        X_noised = X_dense * dropout_mask
    # Calculate the power of the original signal (the average of the sum of squares of all elements)
    signal_power = np.mean(X_dense ** 2)

    # Calculate the power of the noise (the average of the sum of squares of all elements in the noise matrix)
    # First, calculate the noise matrix
    noise_matrix = X_noised - X_dense
    # Then, calculate the noise power
    noise_power = np.mean(noise_matrix ** 2)

    # To avoid division by zero, ensure that the noise power is at least a very small positive number
    noise_power = np.maximum(noise_power, np.finfo(float).eps)

    # Calculate the Signal-to-Noise Ratio (SNR)
    snr = 10 * np.log10(signal_power / noise_power)
    print(snr)
    return X_noised

def reorder_categories(adata, method, new_order):
    """
    Reorder the categories of a given method in the AnnData object's.obs.
    
    Parameters:
    adata: AnnData object
    method: str, the name of the method to reorder
    new_order: list, the new order of categories
    """
    if not pd.api.types.is_categorical_dtype(adata.obs[method]):
        adata.obs[method] = pd.Categorical(adata.obs[method])
    adata.obs[method] = adata.obs[method].cat.reorder_categories(new_order)

def expand_anndata(adata, used_rep='obsm'):
    """
    Expand the original AnnData object to a new AnnData object.
    
    Parameters:
    adata (AnnData): The original AnnData object.
    
    Returns:
    AnnData: The expanded AnnData object.
    """
    if used_rep == 'obsm':
        level_0_data = adata.X
        level_1_data = adata.obsm['level_1']
        level_2_data = adata.obsm['level_2']
        level_3_data = adata.obsm['level_3']
    elif used_rep == 'uns':
        level_0_data = adata.uns['INR_level_0']
        level_1_data = adata.uns['INR_level_1']
        level_2_data = adata.uns['INR_level_2']
        level_3_data = adata.uns['INR_level_3']
    else:
        raise ValueError("used_rep must be either 'obsm' or 'uns'")

    # Vertically stack the data
    new_X = np.vstack([level_0_data, level_1_data, level_2_data, level_3_data])

    # Create a new AnnData object
    new_adata = ad.AnnData(X=new_X)
    
    # Drop the 'batch' column from the original obs
    adata.obs = adata.obs.drop(columns=['batch'])
    
    # Create a 'noise' field for each part of the data
    noise_values = np.repeat([0, 1, 2, 3], adata.n_obs)
    
    # Copy the original obs data and expand it
    new_obs = pd.concat([adata.obs] * 4, ignore_index=True)
    new_obs['noise_level'] = noise_values

    new_adata.obs = new_obs
    new_adata.obsm['spatial'] = np.vstack([adata.obsm['spatial']] * 4)
    return new_adata

"""
==========
Evaluation
==========
"""
def calculate_chaos(coord, labels):
    """
    Calculates the spatial chaos score, with lower values being better.

    Parameters
    ----------
    coord : ndarray
        The coordinates of the data points.

    labels : ndarray
        The labels of the data points.

    Returns
    -------
    chaos_values : list
        The chaos values for each cluster.
    """
    # Get unique class labels
    unique_labels = np.unique(labels)
    
    # Initialize the list of CHAOS values
    chaos_values = []
    
    # Iterate over each class
    for label in unique_labels:
        # Filter out the coordinates of the current class
        coords_label = coord[labels == label]
        
        # If there is only one point in the class, skip the calculation
        if len(coords_label) < 2:
            chaos_values.append(0)  # Or you can set a specific value as needed
            continue
        
        # Calculate the Euclidean distance between all points in the current class
        dist_matrix = squareform(pdist(coords_label, 'euclidean'))
        
        # Construct a 1NN graph, keeping only the edges with the smallest distance
        adjacency_matrix = np.zeros_like(dist_matrix)
        
        # Find the nearest neighbor for each point
        for i in range(len(coords_label)):
            nearest_neighbor = np.argsort(dist_matrix[i])[1]  # The second smallest element is the nearest neighbor
            adjacency_matrix[i, nearest_neighbor] = dist_matrix[i, nearest_neighbor]
        
        # Calculate the CHAOS value for the current class
        # Consider only actual connection edges
        actual_edges = adjacency_matrix[adjacency_matrix != 0]
        if actual_edges.size > 0:
            chaos_label = np.sum(actual_edges) / len(actual_edges)
        else:
            chaos_label = 0  # If there are no actual connection edges, the CHAOS value is 0
        
        # Add the CHAOS value of the current class to the list
        chaos_values.append(chaos_label)
    
    # Return the CHAOS values for each class
    return chaos_values

def metrics_clustering(X, labels):
    """
    Calculates clustering metrics.

    Parameters
    ----------
    X : ndarray
        The data points.

    labels : ndarray
        The labels of the data points.

    Returns
    -------
    clustering_metrics : list
        A list containing the silhouette score, Davies-Bouldin index, and Calinski-Harabasz index.
    """
    # Calculate the silhouette score, which is between [-1, 1], the larger the better
    silhouette_avg = silhouette_score(X, labels)
    # Calculate the Davies-Bouldin index, the lower the better
    db_index = davies_bouldin_score(X, labels)
    # Calculate the Calinski-Harabasz index, the higher the better
    ch_index = calinski_harabasz_score(X, labels)
    clustering_metrics = [silhouette_avg, db_index, ch_index]
    return clustering_metrics

"""
======
Others
======
"""
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if os.environ.get("SPATIALCOC_DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)

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
    # Replace the extreme values that are 2*n in number
    # Calculate the number to replace
    num_to_replace = int(len(arr) * n)
    # Sort the array
    sorted_arr = np.sort(arr)
    # Find the value at the n% percentile
    value_n_percentile = sorted_arr[num_to_replace - 1]  # The index for the n% percentile is num_to_replace - 1
    # Find the value at the (100-n)% percentile
    value_100_n_percentile = sorted_arr[-(num_to_replace + 1)]

    # Find the indices for the bottom n% and top n% values
    min_indices = np.argpartition(arr, num_to_replace)[:num_to_replace]
    max_indices = np.argpartition(arr, -num_to_replace)[-num_to_replace:]

    # Create a copy of the original array
    arr_copy = arr.copy()
    # Replace the extreme values
    arr_copy[min_indices] = value_n_percentile
    arr_copy[max_indices] = value_100_n_percentile

    return arr_copy

def rotate_spatial_coordinates(spatial_coords, angle_degrees):
    """
    Rotate spatial coordinates clockwise by a specified angle.

    Parameters
    ----------
    spatial_coords: Array of spatial coordinates with shape (n_cells, 2).
    angle_degrees: Rotation angle (clockwise) in degrees.

    Returns
    -------
    Rotated spatial coordinates array.
    """
    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), np.sin(angle_radians)],
        [-np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return np.dot(spatial_coords, rotation_matrix.T)
