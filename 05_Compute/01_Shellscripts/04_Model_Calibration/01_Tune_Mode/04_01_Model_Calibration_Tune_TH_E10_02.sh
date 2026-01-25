#!/bin/bash
#PBS -l select=1:ncpus=2:ngpus=1:mem=10gb
#PBS -l walltime=12:00:00
#PBS -A CollGNN
#PBS -N 04_01_Model_Calibration_E10_02
#PBS -J 0-999%50
#PBS -r y
#PBS -o /home/qusta100/STGNN/04_Results/01_Temporary/03_Outputs/04_01_Model_Calibration_E10_02.o
#PBS -e /home/qusta100/STGNN/04_Results/01_Temporary/03_Outputs/04_01_Model_Calibration_E10_02.e

set -eo pipefail
cd "$PBS_O_WORKDIR"

module purge
source /software/conda/mambaforge/24.3.0/etc/profile.d/conda.sh
conda activate /gpfs/project/qusta100/pytorchgpu

IDX=${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-0}}
TRIAL=$((IDX + 1000))

exec >  "/home/qusta100/STGNN/04_Results/01_Temporary/01_Logs/04_Model_Calibration/01_Tune/02_E10/${TRIAL}.out"
exec 2> "/home/qusta100/STGNN/04_Results/01_Temporary/01_Logs/04_Model_Calibration/01_Tune/02_E10/${TRIAL}.err"

jupyter nbconvert \
  --to script \
  /home/qusta100/STGNN/02_Code/04_Model_Calibration/04_01_Model_Calibration_HP.ipynb \
  --output 04_01_Model_Calibration_HP \
  --output-dir /home/qusta100/STGNN/05_Compute/02_Python/04_Model_Calibration


python /home/qusta100/STGNN/05_Compute/02_Python/04_Model_Calibration/04_01_Model_Calibration_HP.py \
  --mode tune \
  --trial "$TRIAL" \
  --base-seed 42 \
  --data-path "/gpfs/scratch/qusta100/STGNN/01_Data/02_Processed_Data/03_Data_Windows/01_Thuringia/02_E10/02_Time/data_Time_5N.pt" \
  --outdir "/home/qusta100/STGNN/04_Results/01_Temporary/02_Trial_Results/04_Model_Calibration/02_E10"