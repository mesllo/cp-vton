#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm/train/tom/full/output/tom_train_efficientnetb3_baseline_warpmod1_1a.out

# experiment parameters
STAGE=TOM
MODE=train
PAIRS="$MODE"_pairs.txt
RELEVANCE_LEVEL=baseline
ID=1   # ID of model (use to distinguish between tom models using same gmm model)
FINETUNE_CONFIG=NONE    # choose from ALL, MID, DEEP, NONE
PLF_LAYERS=ALL      # choose from SHALLOW, MID, DEEP, ALL, DEFAULT
PLN=efficientnet_b3
DATASET=viton
GMM_PATH=gmm_final.pth
GMM_MODEL=warpedclothes1
WARP_DIR="$DATASET"_"$GMM_MODEL"_"$GMM_PATH"
EXP_NAME="$STAGE"_"$MODE"_"$RELEVANCE_LEVEL"_id"$ID"_finetune"$FINETUNE_CONFIG"_layers"$PLF_LAYERS"_"$PLN"_"$DATASET"_"$GMM_MODEL"
KEEP_STEP=100000
DECAY_STEP=100000
SAVE_COUNT=50000

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
time python train.py --pretrained --dataroot "$TMPDIR"/bonelloj/"$DATASET" --name $EXP_NAME --stage $STAGE --datamode $MODE --data_list $PAIRS --warp_dir $WARP_DIR --workers 4 --save_count $SAVE_COUNT --shuffle --keep_step $KEEP_STEP --decay_step $DECAY_STEP --rel $RELEVANCE_LEVEL --fin $FINETUNE_CONFIG --plf $PLF_LAYERS --pln $PLN --dataset $DATASET

# Copy output folder back from scratch
cp -r "$TMPDIR"/output /project/prjsbonelloj/vton/datasets

echo
echo 'Job done!'