#!/usr/bin/env python
"""
Setup script for SpatialCOC package.
SpatialCOC: Spatial Omics Data Integration and Analysis
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="SpatialCOC",
    version="1.0.0",
    author="Mingxuan Li",
    author_email="3123154029@stu.xjtu.edu.cn",
    description="SpatialCOC package for spatial omics data integration and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xjtu-omics/SpatialCOC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GPL-3.0 License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
        ],
    },
    include_package_data=True,
    zip_safe=False,
)