# Benchmarking of six methods

To evaluate the performance of SpaKnit, we compare it against six state-of-the-art methods, including spatial multi-omics data integration method (SpatialGlue), single-cell multi-omics data integration methods (Seurat WNN, MultiVI, and MultiMAP), spatial transcriptome methods (STAGATE and SpaGCN).

After completing the uniform preprocessing steps, we proceeded to evaluate each benchmarking method following the instructions provided in their respective vignettes. Below, we describe the details of each method.

- SpatialGlue. Feature graphs and spatial graphs are constructed using the *construct_neighbor_graph()* function based on the preprocessed feature and spatial information. Subsequently, the model is trained on the neighborhood graph with default parameters to obtain an integrated latent representation.

- Seurat WNN. The *FindMultiModalNeighbors()* function is used to identify the nearest neighbors of each cell based on a weighted combination of the two modalities. The method then constructs a shared nearest neighbor (SNN) graph and performs clustering, incorporating the computed modality weights for each spot.

- MultiVI. We first align and order the multi-omics data using *organize_multiome_anndatas()* function, consolidating different modalities into a single AnnData object. We then set batch_key="modality" when initializing the dataset with *scvi.model.MULTIVI.setup_anndata()* function. The model is trained using default parameters, and the integrated latent space representation is obtained. 

- MultiMAP. The integration is performed using *MultiMAP.Integration()* function, which generates the integrated latent representation under the default parameter Settings.

- SpaGCN. We construct an adjacency matrix from spatial coordinates using the *SpaGCN.calculate_adj_matrix()* function to calculate. The number of clusters is specified, and the optimal resolution is determined using the *spg.search_res()* function. Finally, the model is built and spatial clustering is performed.

- STAGATE. We first merge the data from the two omics modalities. Then a spatial graph is computed using *STAGATE.Cal_Spatial_Net()* function with rad_cutoff=10. The model is trained using *STAGATE.train_STAGATE()* with alpha=0, generating integrated feature representations.