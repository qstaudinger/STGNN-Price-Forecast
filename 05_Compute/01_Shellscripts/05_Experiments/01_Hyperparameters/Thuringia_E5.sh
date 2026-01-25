#!/bin/bash
#PBS -l select=1:ncpus=2:ngpus=1:mem=50gb
#PBS -l walltime=71:00:00
#PBS -A CollGNN
#PBS -N 05_01_Experiments_Evaluation_HP_Thuringia_E5
#PBS -r y
#PBS -o /home/qusta100/STGNN/04_Results/01_Temporary/01_Logs/05_Experiments/05_01_Experiments_Evaluation_HP/01_Thuringia/E5_HP.out
#PBS -e /home/qusta100/STGNN/04_Results/01_Temporary/01_Logs/05_Experiments/05_01_Experiments_Evaluation_HP/01_Thuringia/E5_HP.err

set -eo pipefail
cd "$PBS_O_WORKDIR"

module purge
module load jq/1.5
source /software/conda/mambaforge/24.3.0/etc/profile.d/conda.sh
conda activate /gpfs/project/qusta100/pytorchgpu

BASE="/home/qusta100/STGNN/04_Results/01_Temporary/04_Configurations/04_Model_Calibration/01_E5/best_hyperparams.json"

  
# WindowSize Day 
## Value: 16
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/02_WindowSize_Day/16.json"
jq '.win_day = 16 | .length_encoder = 16' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/02_WindowSize_Day/16"
  
## Value: 32
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/02_WindowSize_Day/32.json"
jq '.win_day = 32 | .length_encoder = 16' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/02_WindowSize_Day/32"

## Value: 64
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/02_WindowSize_Day/64.json"
jq '.win_day = 16 | .length_encoder = 16' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/02_WindowSize_Day/64"

## Value: 96
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/02_WindowSize_Day/96.json"
jq '.win_day = 96 | .length_encoder = 12' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/02_WindowSize_Day/96"
  
  
# WindowSize Week
## Value: 96
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/03_WindowSize_Week/96.json"
jq '.win_week = 96' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/03_WindowSize_Week/096"
  
## Value: 192
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/03_WindowSize_Week/192.json"
jq '.win_week = 192' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/03_WindowSize_Week/192"
  
## Value: 288
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/03_WindowSize_Week/288.json"
jq '.win_week = 288' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/03_WindowSize_Week/288"
  
## Value: 480
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/03_WindowSize_Week/480.json"
jq '.win_week = 480' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/03_WindowSize_Week/480"

## Value: 672
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/03_WindowSize_Week/672.json"
jq '.win_week = 672' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/03_WindowSize_Week/672"
  
  
# Chebyshev FilterSize
## Value: 1
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/04_Chebyshev_FilterSize/1.json"
jq '.cheb_K = 1' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/04_Chebyshev_FilterSize/1"
  
## Value: 2
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/04_Chebyshev_FilterSize/2.json"
jq '.cheb_K = 2' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/04_Chebyshev_FilterSize/2"
  
## Value: 3
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/04_Chebyshev_FilterSize/3.json"
jq '.cheb_K = 3' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/04_Chebyshev_FilterSize/3"
  
## Value: 4
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/04_Chebyshev_FilterSize/4.json"
jq '.cheb_K = 4' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/04_Chebyshev_FilterSize/4"
  
## Value: 5
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/04_Chebyshev_FilterSize/5.json"
jq '.cheb_K = 5' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/04_Chebyshev_FilterSize/5"
  
  
# GCN Layers
## Value: 1
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/05_GCN_Layers/1.json"
jq '.cheb_layers = 1' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/05_GCN_Layers/1"
  
## Value: 2
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/05_GCN_Layers/2.json"
jq '.cheb_layers = 2' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/05_GCN_Layers/2"

## Value: 3
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/05_GCN_Layers/3.json"
jq '.cheb_layers = 3' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/05_GCN_Layers/3"
  
## Value: 4
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/05_GCN_Layers/4.json"
jq '.cheb_layers = 4' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/05_GCN_Layers/4"
  
## Value: 5
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/05_GCN_Layers/5.json"
jq '.cheb_layers = 5' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/05_GCN_Layers/5"
  
## Value: 6
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/05_GCN_Layers/6.json"
jq '.cheb_layers = 6' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/05_GCN_Layers/6"
  
  
    
# Attentional Heads
## Value: 2
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/06_Attentional_Heads/2.json"
jq '.attn_heads = 2' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/06_Attentional_Heads/2"
  
  
## Value: 4
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/06_Attentional_Heads/4.json"
jq '.attn_heads = 4' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/06_Attentional_Heads/4"
  
## Value: 6
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/06_Attentional_Heads/6.json"
jq '.attn_heads = 6' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/06_Attentional_Heads/6"
  
## Value: 8
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/06_Attentional_Heads/8.json"
jq '.attn_heads = 8' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/06_Attentional_Heads/8"
  
  
# Temporal Decoder Attention Layers
## Value: 1
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/07_Temporal_Decoder_Attention_Layers/1.json"
jq '.temporal_attn_layers = 1' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/07_Temporal_Decoder_Attention_Layers/1"
  
  
## Value: 2
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/07_Temporal_Decoder_Attention_Layers/2.json"
jq '.temporal_attn_layers = 2' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/07_Temporal_Decoder_Attention_Layers/2"
  
  
## Value: 3
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/07_Temporal_Decoder_Attention_Layers/3.json"
jq '.temporal_attn_layers = 3' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/07_Temporal_Decoder_Attention_Layers/3"
  
  
## Value: 4
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/07_Temporal_Decoder_Attention_Layers/4.json"
jq '.temporal_attn_layers = 4' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/07_Temporal_Decoder_Attention_Layers/4"
  
  
## Value: 5
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/07_Temporal_Decoder_Attention_Layers/5.json"
jq '.temporal_attn_layers = 5' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/07_Temporal_Decoder_Attention_Layers/5"
  
## Value: 6
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/07_Temporal_Decoder_Attention_Layers/6.json"
jq '.temporal_attn_layers = 6' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/07_Temporal_Decoder_Attention_Layers/6"


# WindowSize Spatial
## Value: 2
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/01_WindowSize_Spatial/2.json"
jq '.win_hour = 2' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/01_WindowSize_Spatial/02"
  
## Value: 4
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/01_WindowSize_Spatial/4.json"
jq '.win_hour = 4' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/01_WindowSize_Spatial/04"
  
  
## Value: 8
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/01_WindowSize_Spatial/8.json"
jq '.win_hour = 8' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/01_WindowSize_Spatial/08"
  
    
## Value: 16
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/01_WindowSize_Spatial/16.json"
jq '.win_hour = 16' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/01_WindowSize_Spatial/16"
  
    
## Value: 32
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/01_Experiments_Evaluation_HP/01_E5/01_WindowSize_Spatial/32.json"
jq '.win_hour = 32' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/01_E5/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/01_Experiments_Evaluation_HP/01_Thuringia/01_E5/01_WindowSize_Spatial/32"
  