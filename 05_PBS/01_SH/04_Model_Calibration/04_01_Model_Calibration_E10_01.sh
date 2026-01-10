#!/bin/bash
#PBS -l select=1:ncpus=2:ngpus=1:mem=10gb
#PBS -l walltime=12:00:00
#PBS -A CollGNN
#PBS -N 04_01_Model_Calibration_E10_01
#PBS -J 0-999%50
#PBS -r y
#PBS -o STGNN/04_Results/01_Temporary/03_Outputs/04_01_Model_Calibration_E10_01.o
#PBS -e STGNN/04_Results/01_Temporary/03_Outputs/04_01_Model_Calibration_E10_01.e

set -eo pipefail
cd "$PBS_O_WORKDIR"

module purge
source /software/conda/mambaforge/24.3.0/etc/profile.d/conda.sh
conda activate /gpfs/project/qusta100/pytorchgpu

IDX=${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-0}}
TRIAL=$((IDX + 0))

exec >  "STGNN/04_Results/01_Temporary/01_Logs/04_Model_Calibration/04_01_Model_Calibration_HP/02_E10/${TRIAL}.out"
exec 2> "STGNN/04_Results/01_Temporary/01_Logs/04_Model_Calibration/04_01_Model_Calibration_HP/02_E10/${TRIAL}.err"

jupyter nbconvert \
  --to script \
  STGNN/02_Notebooks/04_Model_Calibration/04_01_Model_Calibration_HP.ipynb \
  --output 04_01_Model_Calibration_HP \
  --output-dir STGNN/05_PBS/02_Python/04_Model_Calibration


python STGNN/05_PBS/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --trial "$TRIAL" \
  --base-seed 42 \
  --data-path "STGNN/01_Data/02_Processed_Data/03_Data_Windows/02_E10/02_Time/data_Time_5N.pt" \
  --outdir "STGNN/04_Results/01_Temporary/02_Trial_Results/04_Model_Calibration/04_01_Model_Calibration_HP/02_E10"
            