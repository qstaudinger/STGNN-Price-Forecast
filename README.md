# Spatio-Temporal Graph Neural Networks for Price Forecasting: Evidence from Gasoline Markets

## Overview

This repository contains the code for a research project on **short-term fuel price forecasting** using **spatio-temporal graph neural networks (STGNNs)**.  
The goal is to predict local price movements in the German retail gasoline market by explicitly modeling **competition between nearby stations**.

Fuel prices change frequently and in coordinated patterns that reflect local strategic behavior rather than pure cost shocks.  
Traditional forecasting models, both statistical and deep learning, treat each station as an isolated time series and miss these interactions.  
This project represents gas stations as **nodes in a spatial graph**, linking them based on proximity so that price reactions can propagate through the network.

The data come from the **German Market Transparency Unit for Fuels (MTS-K)**, observed at 15-minute intervals.  
Highway stations and night hours are excluded to avoid distortions.  
The analysis focuses on **E5, E10, and diesel**, using a geographically restricted subsample from **Thuringia**.

---

## Repository structure

The repository follows the full empirical workflow from raw data to final results.
````
01_Data/
├── 01_Original_Data/          Raw price and station data
├── 02_Processed_Data/         Cleaned data and intermediate outputs
│   ├── 01_OSMR/               OSRM distance and travel time data
│   ├── 02_Analysis/           Intermediate analysis results
│   └── 03_Data_Windows/       PyTorch data windows
│       ├── 01_E5/
│       ├── 02_E10/
│       └── 03_Diesel/
└── 03_Shapefiles/             Geographic shapefiles

02_Notebooks/
├── 01_Preprocessing/          Data preprocessing and OSRM handling
├── 02_Summary_Statistics/     Descriptive statistics and graph construction
├── 03_Window_Building/        Spatial and temporal window construction
├── 04_Model_Calibration/      Hyperparameter tuning
├── 05_Experiments/            Model experiments and rankings
└── 06_Performance_Comparison/ Benchmark models (ARIMA, SVR, optimized models)

03_Graphs/
├── 01_Descriptive_Statistics/ Exploratory figures
├── 02_Graphs/                 Graph and neighborhood visualizations
├── 03_Experiments/            Experiment result plots
├── 04_Model_Evaluation/       Model evaluation figures
└── 05_Performance_Comparison/ Benchmark comparison plots

04_Results/
├── 01_Temporary/              Intermediate results and logs
│   ├── 01_Logs/
│   ├── 02_Trial_Results/
│   ├── 03_Outputs/
│   ├── 04_Models/             Stored Models
│   └── 05_Metrics/            Aggregated performance metrics
└── 02_Tables/                 Final result tables


# └── 05_Metrics/                Aggregated performance metrics

05_PBS/
├── 01_SH/                     Shell scripts for cluster execution
│   ├── 04_Model_Calibration/
│   ├── 05_Experiments/
│   └── 06_Performance_Comparison/
└── 02_Python/                 Python scripts mirroring the notebook workflow
````
Notes:
- Numeric prefixes indicate execution order.
- Temporary and final outputs are kept separate for reproducibility.
- Raw data from Tankerkönig and Yahoo Finance are not included due to size and licensing restrictions.
- All notebooks are self-contained once data paths are configured.


---

## Requirements & Installation

To run the notebooks, ensure you have **Python 3.8+** and **Jupyter Notebook** or **JupyterLab** installed.

Install dependencies via:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch networkx tqdm yfinance geopandas shapely jupyterlab lightgbm statsmodels joblib
```

All models are implemented in Python 3.10 and trained on a compute cluster.

The STGNN model is implemented in PyTorch 2.5.1 and trained on a single NVIDIA GeForce GTX 1080 Ti (11 GB VRAM) using 4 CPU cores per run. Baseline models are trained on CPU only using their respective libraries.
