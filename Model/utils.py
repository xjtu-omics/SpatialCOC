import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import check_array
from scipy.sparse import issparse, lil_matrix
import os
import anndata as ad
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

def check_Xs(
        Xs,
        multiview=True,
        enforce_views=None,
        copy=False,
        return_dimensions=False,
):
    r"""
    Checks Xs and ensures it to be a list of 2D matrices.

    Parameters
    ----------
    Xs : nd-array, list
        Input data.

    multiview : boolean, (default=False)
        If True, throws error if just 1 data matrix given.

    enforce_views : int, (default=not checked)
        If provided, ensures this number of modalities in Xs. Otherwise not
        checked.

    copy : boolean, (default=False)
        If True, the returned Xs is a copy of the input Xs,
        and operations on the output will not affect
        the input.
        If False, the returned Xs is a modality of the input Xs,
        and operations on the output will change the input.

    return_dimensions : boolean, (default=False)
        If True, the function also returns the dimensions of the multiview
        dataset. The dimensions are n_views, n_samples, n_features where
        n_samples and n_views are respectively the number of modalities and the
        number of samples, and n_features is a list of length n_views
        containing the number of features of each modality.

    Returns
    -------
    Xs_converted : object
        The converted and validated Xs (list of data arrays).

    n_views : int
        The number of modalities in the dataset. Returned only if
        ``return_dimensions`` is ``True``.

    n_samples : int
        The number of samples in the dataset. Returned only if
        ``return_dimensions`` is ``True``.

    n_features : list
        List of length ``n_views`` containing the number of features in
        each modality. Returned only if ``return_dimensions`` is ``True``.
    """
    if not isinstance(Xs, list):
        if not isinstance(Xs, np.ndarray):
            msg = f"If not list, input must be of type np.ndarray,\
                not {type(Xs)}"
            raise ValueError(msg)
        if Xs.ndim == 2:
            Xs = [Xs]
        else:
            Xs = list(Xs)

    n_views = len(Xs)
    if n_views == 0:
        msg = "Length of input list must be greater than 0"
        raise ValueError(msg)

    if multiview:
        if n_views == 1:
            msg = "Must provide at least two data matrices"
            raise ValueError(msg)
        if enforce_views is not None and n_views != enforce_views:
            msg = "Wrong number of modalities. Expected {} but found {}".format(
                enforce_views, n_views
            )
            raise ValueError(msg)

    Xs = [check_array(X, allow_nd=False, copy=copy) for X in Xs]

    if not len(set([X.shape[0] for X in Xs])) == 1:
        msg = "All modalities must have the same number of samples"
        raise ValueError(msg)

    if return_dimensions:
        n_samples = Xs[0].shape[0]
        n_features = [X.shape[1] for X in Xs]
        return Xs, n_views, n_samples, n_features
    else:
        return Xs


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
    # the location of R (used for the mclust clustering)
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    os.environ['R_HOME'] = 'E:\R-4.3.1'
    os.environ['R_USER'] = 'E:\anaconda\lib\site-packages\rpy2'

    np.random.seed(random_seed)
    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['clusters_mclust'] = mclust_res
    adata.obs['clusters_mclust'] = adata.obs['clusters_mclust'].astype('int')
    adata.obs['clusters_mclust'] = adata.obs['clusters_mclust'].astype('category')

    return adata

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