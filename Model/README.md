# Model Description

> This is an introduction to the individual files in this folder, see the detailed description of the function in each file.

|   File Name   |                         Description                          |
| :-----------: | :----------------------------------------------------------: |
|    INR.py     | This file is used to implement the first module of SpaKnit: Implicit Neural Representation (INR) Module. This module takes the shared spatial coordinates as input and reconstructs a continuous representation of each omics. |
|   model.py    | This file is used to implement the second module of SpaKnit: Deep Canonically Correlated Auto-Encoder (DCCAE) Module. This module focuses on capturing the nonlinear correlations among omics modalities while eliminating modality-specific noise. |
| preprocess.py | This file encompasses the preprocessing methods for spatial multi-omics data derived from various techniques. By subjecting the data to a unified preprocessing protocol, a standardized foundation is established for the subsequent performance evaluation of different methods. |
|   utils.py    | This file contains some useful functions, including the "mclust" clustering method, noise generation. |

## ✨Requirements

> Please install the following packages to ensure that SpaKnit works correctly.

- ﻿python==3.11.5
- torch>=2.1.2
- torchvision>=0.16.2
- numpy==1.24.3
- scipy==1.11.3
- scikit-learn>=1.6.0
- pandas==2.1.1
- scanpy==1.9.5
- anndata>=0.11.1
- tqdm==4.65.0
- matplotlib==3.8.0
- opencv-python==4.9.0.80
- rpy2==3.5.14
- R==4.3.1
