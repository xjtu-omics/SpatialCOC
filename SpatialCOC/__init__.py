#!/usr/bin/env python
"""
# Author: Mingxuan Li
# File Name: __init__.py
# Description: SpatialCOC package for spatial multi-omics data integration and analysis
"""

__author__ = "Mingxuan Li"
__email__ = "3123154029@stu.xjtu.edu.cn"
__version__ = "1.0.0"

# Import main classes and functions
from .preprocess import preprocessing
from .utils import (
    mclust_R,
    calculate_chaos,
    metrics_clustering,
    fix_seed,
    reorder_categories
)
from .INR import SCM
from .model import COC

__all__ = [
    ## Preprocessing
    "preprocessing",
    ## Utils
    "mclust_R",
    "calculate_chaos",
    "metrics_clustering",
    "fix_seed",
    "reorder_categories",
    ## Model
    "SCM",
    "COC"
]
