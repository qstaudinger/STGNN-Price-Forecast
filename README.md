# Spatio-Temporal Graph Neural Networks for Price Forecasting: Evidence from Gasoline Markets

## Overview

This repository contains the code for a research project on **short-term fuel price forecasting** using **spatio-temporal graph neural networks (STGNNs)**.  
The goal is to predict local price movements in the German retail gasoline market by explicitly modeling **competition between nearby stations**.

Fuel prices change frequently and in coordinated patterns that reflect local strategic behavior rather than pure cost shocks.  
Traditional forecasting models—both statistical and deep learning—treat each station as an isolated time series and miss these interactions.  
This project represents gas stations as **nodes in a spatial graph**, linking them based on proximity so that price reactions can propagate through the network.

The data come from the **German Market Transparency Unit for Fuels (MTS-K)**, observed at 15-minute intervals.  
Highway stations and night hours are excluded to avoid distortions.  
The analysis focuses on **E5, E10, and diesel**, using a geographically restricted subsample from **Thuringia**.

---

## Repository Structure

### 01_Data_Pipeline.ipynb
Builds a robust data pipeline integrating fuel price data from **Tankerkönig** and oil price series via **yfinance**.  
Performs validation, cleaning, and standardization, then exports reproducible datasets and prepares a **Thuringia sample**.

### 02_Summary.ipynb
Provides an **exploratory overview** of the cleaned data.  
Summarizes key price dynamics, validates data quality, and ensures representativeness before modeling.

### 03_GNN_Edge_Construction.ipynb
Constructs **graph structures** capturing spatial and relational dependencies.  
Defines edges based on distance, brand, and price correlation, forming the basis for STGNN experiments.

### 04_STGNN_Draft.ipynb
Implements a preliminary **spatio-temporal graph neural network** for short-term forecasting  
and evaluates alternative architectures for temporal and spatial encoding.

---

## Requirements & Installation
## Requirements & Installation

To run the notebooks, ensure you have **Python 3.8+** and **Jupyter Notebook** or **JupyterLab** installed.

Install dependencies via:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torch-geometric networkx tqdm yfinance geopandas shapely jupyterlab
```

## Notes

- Raw data from Tankerkönig and Yahoo Finance are not included due to size and licensing restrictions.
- All notebooks are self-contained once data paths are configured.
- The project is research-oriented and not designed for production deployment.

