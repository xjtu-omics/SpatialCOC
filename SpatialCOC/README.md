# Model Description

> This is an introduction to the individual files in this folder, see the detailed description of the function in each file.

|   File Name   |                         Description                          |
| :-----------: | :----------------------------------------------------------: |
|    INR.py     | This file is used to implement the first module of SpatialCOC: Spatial Continuous Mapping (SCM) Module. This module takes the shared spatial coordinates as input and reconstructs a continuous representation of each omics. |
|   model.py    | This file is used to implement the second module of SpatialCOC: Cross-Omics Correction (COC) Module. This module focuses on capturing the nonlinear correlations among omics modalities while eliminating modality-specific noise. |
| preprocess.py | This file encompasses the preprocessing methods for spatial multi-omics data derived from various techniques. By subjecting the data to a unified preprocessing protocol, a standardized foundation is established for the subsequent performance evaluation of different methods. |
|   utils.py    | This file contains some useful functions, including the "mclust" clustering method, noise generation. |

## ✨Requirements

> Please install the following packages to ensure that SpatialCOC works correctly.

- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.24.0,<2.0.0
- scipy>=1.10.0
- scikit-learn>=1.3.0
- pandas>=2.0.0
- scanpy>=1.9.0
- anndata>=0.10.0
- tqdm>=4.65.0
- matplotlib>=3.7.0
- opencv-python>=4.8.0
- igraph>=1.0.0
- leidenalg>=0.11.0
