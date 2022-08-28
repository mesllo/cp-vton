#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm/train/gmm/quick/output/gmm_train_quick.out

# experiment parameters
STAGE=GMM
MODE=train
PAIRS="$MODE"_pairs.txt
DATASET=viton_quick
EXP_NAME="$DATASET"_warpmod3
KEEP_STEP=50
DECAY_STEP=50
SAVE_COUNT=10

# Load GPU drivers
module purge
module load 2021
module load Miniconda3/4.9.2

#Make scratch directory
mkdir "$TMPDIR"/bonelloj

# Moving dataset to scratch
echo "Moving dataset to scratch! Time taken will be outputted below:"
time cp -r /project/prjsbonelloj/vton/datasets/"$DATASET" "$TMPDIR"/bonelloj

#Create output directory on scratch
mkdir -p "$TMPDIR"/output

# This loads the anaconda virtual environment with our packages
source /home/bonelloj/.bashrc
conda activate vton37

# go to app directory
cd /project/prjsbonelloj/vton/repos/cp-vton

echo
echo "Starting experiment! Time taken will be outputted after execution."
echo

# Run the actual experiment
time python train.py --dataroot "$TMPDIR"/bonelloj/"$DATASET" --name $EXP_NAME --stage $STAGE --datamode $MODE --data_list $PAIRS --workers 4 --save_count $SAVE_COUNT --shuffle --keep_step $KEEP_STEP --decay_step $DECAY_STEP --dataset $DATASET

# Copy output folder back from scratch
cp -r "$TMPDIR"/output /project/prjsbonelloj/vton/datasets

echo
echo 'Job done!'