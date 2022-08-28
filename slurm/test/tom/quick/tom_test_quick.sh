#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm/test/tom/quick/output/tom_test_quick.out

# experiment parameters
STAGE=TOM
MODE=test
PAIRS="$MODE"_pairs.txt
RELEVANCE_LEVEL=baseline
FINETUNE_CONFIG=NONE    # choose from ALL, MID, DEEP
PLF_LAYERS=DEFAULT      # choose from SHALLOW, MID, DEEP, ALL, DEFAULT
PLN=vgg19
DATASET=viton_quick
GMM_PATH=gmm_final.pth
GMM_MODEL=warpedclothes1
WARP_DIR="$DATASET"_"$GMM_MODEL"_"$GMM_PATH"
TOM_PATH=tom_final.pth
TOM_DIR="$STAGE"_train_"$RELEVANCE_LEVEL"_finetune"$FINETUNE_CONFIG"_layers"$PLF_LAYERS"_"$PLN"_"$DATASET"_"$GMM_MODEL"
EXP_NAME="$STAGE"_"$MODE"_"$RELEVANCE_LEVEL"_finetune"$FINETUNE_CONFIG"_layers"$PLF_LAYERS"_"$PLN"_"$DATASET"_"$GMM_MODEL"_"$TOM_PATH"

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
time python test.py --dataroot "$TMPDIR"/bonelloj/"$DATASET" --name $EXP_NAME --stage $STAGE --datamode $MODE --data_list $PAIRS --checkpoint checkpoints/$TOM_DIR/$TOM_PATH --warp_dir $WARP_DIR --workers 4 --rel $RELEVANCE_LEVEL --fin $FINETUNE_CONFIG --plf $PLF_LAYERS --pln $PLN --dataset $DATASET

# Copy output folder back from scratch
cp -r "$TMPDIR"/output /project/prjsbonelloj/vton/datasets

echo
echo 'Job done!'