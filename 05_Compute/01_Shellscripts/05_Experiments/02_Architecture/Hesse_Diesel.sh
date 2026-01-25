#!/bin/bash
#PBS -l select=1:ncpus=2:ngpus=1:mem=50gb
#PBS -l walltime=71:00:00
#PBS -A CollGNN
#PBS -N 05_01_Experiments_Evaluation_Acrhicture_Hesse_Diesel
#PBS -r y
#PBS -o /home/qusta100/STGNN/04_Results/01_Temporary/01_Logs/05_Experiments/05_02_Experiments_Evaluation_Architecture/02_Hesse/Diesel_HP.out
#PBS -e /home/qusta100/STGNN/04_Results/01_Temporary/01_Logs/05_Experiments/05_02_Experiments_Evaluation_Architecture/02_Hesse/Diesel_HP.err

set -eo pipefail
cd "$PBS_O_WORKDIR"

module purge
module load jq/1.5
source /software/conda/mambaforge/24.3.0/etc/profile.d/conda.sh
conda activate /gpfs/project/qusta100/pytorchgpu

BASE="/home/qusta100/STGNN/04_Results/01_Temporary/04_Configurations/04_Model_Calibration/03_Diesel/best_hyperparams.json"

# use_temp_day off
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/02_Experiments_Evaluation_Architecture/03_Diesel/01_use_temp_day/use_temp_day_off.json"
jq '{hyperparams: ., flags: {use_temp_day: false}}' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_Hesse/03_Diesel/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/02_Experiments_Evaluation_Architecture/02_Hesse/03_Diesel/01_use_temp_day"
  
# use_temp_week off
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/02_Experiments_Evaluation_Architecture/03_Diesel/02_use_temp_week/use_temp_week.json"
jq '{hyperparams: ., flags: {use_temp_week: false}}' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_Hesse/03_Diesel/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/02_Experiments_Evaluation_Architecture/02_Hesse/03_Diesel/02_use_temp_week"
  
# use_temp_day and use_temp_week off
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/02_Experiments_Evaluation_Architecture/03_Diesel/03_use_temp_day_week/use_temp_day_week.json"
jq '{hyperparams: ., flags: {use_temp_day: false, use_temp_week: false}}' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_Hesse/03_Diesel/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/02_Experiments_Evaluation_Architecture/02_Hesse/03_Diesel/03_use_temp_day_week"
  
  
# use_spatial off
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/02_Experiments_Evaluation_Architecture/03_Diesel/04_use_spatial/use_spatial.json"
jq '{hyperparams: ., flags: {use_spatial: false}}' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_Hesse/03_Diesel/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/02_Experiments_Evaluation_Architecture/02_Hesse/03_Diesel/04_use_spatial"
  
  
# use_attention off
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/02_Experiments_Evaluation_Architecture/03_Diesel/05_use_attention/use_attention.json"
jq '{hyperparams: ., flags: {use_attention: false}}' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_Hesse/03_Diesel/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/02_Experiments_Evaluation_Architecture/02_Hesse/03_Diesel/05_use_attention"
  
  
# use_time off
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/02_Experiments_Evaluation_Architecture/03_Diesel/06_use_time/use_time.json"
jq '{hyperparams: ., flags: {use_time: false}}' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_Hesse/03_Diesel/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/02_Experiments_Evaluation_Architecture/02_Hesse/03_Diesel/06_use_time"
  
# use_condat off
NEWCFG="STGNN/04_Results/01_Temporary/04_Configurations/05_Experiments/02_Experiments_Evaluation_Architecture/01_E5/07_use_condat/use_condat.json"
jq '{hyperparams: ., flags: {use_condat: false}}' "$BASE" > "$NEWCFG"
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$NEWCFG" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_Hesse/03_Diesel/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/02_Experiments_Evaluation_Architecture/02_Hesse/03_Diesel/07_use_condat"
  
  
# use distance
python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "$BASE" \
  --seeds 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_Hesse/03_Diesel/01_Dist/data_Dist_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/05_Experiments/02_Experiments_Evaluation_Architecture/02_Hessse/03_Diesel/08_use_distance"
  