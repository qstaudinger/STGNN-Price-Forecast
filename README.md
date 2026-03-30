# Spatio-Temporal Graph Neural Networks for Price Forecasting: Evidence from Gasoline Markets

## Overview

This repository contains the code for a research project on **short-term fuel price forecasting** using **spatio-temporal graph neural networks (STGNNs)**.  
The goal is to predict **local price movements** in the German retail gasoline market by explicitly modeling **competition between nearby stations**.

Fuel prices change frequently and often follow coordinated patterns that reflect **local strategic behavior** rather than pure cost shocks.  
Traditional forecasting models, both **statistical** and **deep learning**, treat each station as an isolated time series and fail to capture these interactions.  
This project represents gas stations as nodes in a **spatial graph**, linking them based on geographic proximity so that **price reactions propagate through the network**.

The data are obtained from the **German Market Transparency Unit for Fuels (MTS-K)** and observed at **15-minute intervals**.  
The analysis focuses on **E5 fuel prices** using a geographically restricted subsample from **Thuringia**.

In addition, the project includes a series of **experiments** to assess the contribution of individual **model components** and compares the proposed approach against several **baseline models**.

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


### OSRM Server Setup (Routing)

Routing requires a locally running OSRM server (Linux only, not supported on HPC systems).

1. Download OpenStreetMap data for Germany (`.osm.pbf`)
2. Preprocess it using OSRM
3. Start the server:

```bash
docker run -t -i -p 5000:5000 -v "$PWD:/data" osrm/osrm-backend \
  osrm-routed --algorithm mld /data/germany.osrm
````





## Reproducibility Guide

Most parts of the pipeline can be executed independently. Intermediate results (e.g. `.pt` files) are included in the repository, so full recomputation is not required.

### Workflow Overview


Tuning (Shell) → Select Parameters (Notebook) → Training (Shell) → Evaluation (Notebook)


### General Notes

- Scripts are located in `/02_Code`, but not all steps are executed directly as notebooks.
- Computationally intensive tasks are handled via shell scripts in `/05_Compute/01_Shellscripts`.
- These shell scripts:
  - contain **absolute paths** → update them before execution
  - should be converted using `dos2unix` after modification

- Notebooks are used for preprocessing, analysis, and evaluation  
- Shell scripts are used for tuning, training, and experiments  

---

## Setup

### Path Configuration

In `/02_Code/Setup.py`, define all relevant paths:
- raw data
- processed data
- results
- figures

---


## 1. Data Processing

### 1.1 Raw Data Processing (Notebook, CPU)

01_01_Data_Processing.ipynb

Note: Requires very high memory (up to ~500 GB RAM).

### 1.2 Routing + OSRM Integration

01_02_OSRM_Processing.ipynb

This step is handled within the notebook:

- Export routing inputs for OSRM
- Run OSRM and the routing script externally
- Import results back into the notebook
- The exact procedure is documented inside the notebook.

Routing script:

```bash
05_Compute/02_Python/01_Preprocessing/route_candidates.py
````

## 2. Graph Construction (Optional)


02_01_Descriptive_Statistics.ipynb

02_02_Graph_Construction.ipynb


---

## 3. Window Generation


03_01_Window_Building_Dist.ipynb

03_02_Window_Building_Time.ipynb


---

## 4. Model Calibration

All model implementations are defined in `/02_Code` and are executed via shell scripts.

### 4.1 Hyperparameter Tuning (Shell)


05_Compute/01_Shellscripts/04_Model_Calibration/01_Tune_Mode/04_01_Model_Calibration_Tune_TH_E5.sh


### 4.2 Select Best Hyperparameters (Notebook)


04_02_Model_Calibration_Results.ipynb


Aggregates tuning results and stores the selected parameters for training.

### 4.3 Training (Shell)


05_Compute/01_Shellscripts/04_Model_Calibration/02_Train_Mode/04_01_Model_Calibration_Train_TH_E5.sh
05_Compute/01_Shellscripts/04_Model_Calibration/02_Train_Mode/04_01_Model_Calibration_Train_HE_E5.sh


### 4.4 Evaluation (Notebook)


04_03_Model_Calibration_Accuracy.ipynb


---

## 5. Experiments

### 5.1 Run Experiments (Shell)

Experiments are executed via job arrays. Each script contains multiple subruns.

Examples:

```bash
qsub -J 1-8 05_Compute/01_Shellscripts/05_Experiments/02_Architecture/Thuringia_E5.sh
qsub -J 1-21 05_Compute/01_Shellscripts/05_Experiments/01_Hyperparameters/Thuringia_E5.sh
````

### 5.2 Evaluation (Notebook)
05_01_Experiments_Evaluation_HP.ipynb  
05_02_Experiments_Evaluation_Architecture.ipynb

---

## 6. Baseline Models

Baseline model implementations are located in /02_Code and follow the same workflow.

### 6.1 Hyperparameter Tuning (Shell)
05_Compute/01_Shellscripts/06_Performance_Comparison/01_Tune_Mode/01_E5/06_11_Performance_Comparison_SARIMA.sh  
05_Compute/01_Shellscripts/06_Performance_Comparison/01_Tune_Mode/01_E5/06_12_Performance_Comparison_SARIMAX.sh  
05_Compute/01_Shellscripts/06_Performance_Comparison/01_Tune_Mode/01_E5/06_13_Performance_Comparison_Gradient.sh

### 6.2 Select Best Hyperparameters (Notebook)
06_21_Performance_Comparison_OptModels.ipynb

### 6.3 Training (Shell)

Example:

05_Compute/01_Shellscripts/06_Performance_Comparison/02_Train_Mode/01_Thuringia/01_E5/06_11_Performance_Comparison_SARIMA.sh

### 6.4 Evaluation (Notebook)
06_22_Performance_Comparison_Results.ipynb  
06_23_Performance_Comparison_Evaluation.ipynb
