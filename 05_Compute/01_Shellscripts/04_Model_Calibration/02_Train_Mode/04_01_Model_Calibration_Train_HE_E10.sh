#!/bin/bash
#PBS -l select=1:ncpus=2:ngpus=1:mem=10gb
#PBS -l walltime=12:00:00
#PBS -A CollGNN
#PBS -N 04_01_Model_Calibration_E10_Hesse
#PBS -r y
#PBS -o /home/qusta100/STGNN/04_Results/01_Temporary/03_Outputs/04_01_Model_Calibration_E10_Hesse.o
#PBS -e /home/qusta100/STGNN/04_Results/01_Temporary/03_Outputs/04_01_Model_Calibration_E10_Hesse.e

set -eo pipefail
cd "$PBS_O_WORKDIR"

module purge
source /software/conda/mambaforge/24.3.0/etc/profile.d/conda.sh
conda activate /gpfs/project/qusta100/pytorchgpu

exec >  "/home/qusta100/STGNN/04_Results/01_Temporary/01_Logs/04_Model_Calibration/02_Train/02_Hesse/03_E10/train.out"
exec 2> "/home/qusta100/STGNN/04_Results/01_Temporary/01_Logs/04_Model_Calibration/02_Train/02_Hesse/03_E10/train.err"

jupyter nbconvert \
  --to script \
  STGNN/02_Code/04_Model_Calibration/04_01_Model_Calibration_HP.ipynb \
  --output 04_01_Model_Calibration_HP \
  --output-dir STGNN/05_Compute/02_Python/04_Model_Calibration


python STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode train \
  --config-path "/home/qusta100/STGNN/04_Results/01_Temporary/04_Configurations/04_Model_Calibration/03_E10/best_hyperparams.json" \
  --seeds 42 43 44 45 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_Hesse/03_E10/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/05_Predictions/04_Model_Calibration/02_Hesse/03_E10"