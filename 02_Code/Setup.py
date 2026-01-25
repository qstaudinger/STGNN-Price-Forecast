##############################
# 1. Global Import
##############################

from __future__ import annotations

# Standard library
import argparse
import gc
import glob
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.nn import (
    ChebConv,
    GATConv,
    GatedGraphConv,
    SAGEConv,
)
from torch_geometric.utils import subgraph

# Progress bar
from tqdm.notebook import tqdm


##############################
# 2. Set Up Directories
##############################

# --- Roots ---
HOME_ROOT    = Path("/home/qusta100/STGNN")
SCRATCH_ROOT = Path("/gpfs/scratch/qusta100/STGNN")

# --- Data (scratch) ---
DATA_SCRATCH = SCRATCH_ROOT / "01_Data"

# --- Graphs (home) ---
GRAPHS_DIR = HOME_ROOT / "03_Graphs"

# --- Results (home) ---
RESULTS_DIR = HOME_ROOT / "04_Results"


##############################
# 3. Show Working Directory
##############################

# Confirm the current working directory
print("Current working directory:", os.getcwd())