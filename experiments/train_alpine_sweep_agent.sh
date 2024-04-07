#!/bin/bash

#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/run_alpine_sweep_agent.%j.log

# FIXME: This script is deprecated!

source /curc/sw/anaconda3/latest
conda activate hyperparameter-sensitivity

# Trains a sweep agent on the other server, with a different disk.

# Fixes path issue.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/adwi9965/software/anaconda/envs/yoyodyne/lib/

# SET AS ENV VARS FROM CALLING SCRIPT
# TASK
# LANGUAGE
# ARCH
# SWEEP_ID
# WANDB_PROJECT

readonly ROOT="/projects/adwi9965/yoyodyne"
readonly DATA="/scratch/alpine/adwi9965/${TASK}"

if [[ "${TASK}" == "sig2021-g2p" ]]
then
    # We only do the medium setting for G2P (high is English-only and very large data)
    readonly TRAIN="${DATA}/medium/${LANGUAGE}_train.tsv"
    readonly DEV="${DATA}/medium/${LANGUAGE}_dev.tsv"
    readonly FEATURES_COL=0
elif [[ "${TASK}" == "sig2017" ]]
then
    # We only do high for inflection
    readonly TRAIN="${DATA}/${LANGUAGE}-train-high"
    readonly DEV="${DATA}/${LANGUAGE}-dev"
else
    echo No task "${TASK}" only "sig2021-g2p" and "sig2017" are implemented
    exit 1;
fi

readonly RESULTS_PATH="/scratch/alpine/adwi9965/yoyodyne-sweeps/results/${TASK}-${LANGUAGE}/${ARCH}"

# Different scripts b/c we have to hardcode sweep info, and want to queue LSTM and TRM in parallel sometimes.
# The 2 scripts are just reserved for hardcoding the respective sweep info.
SCRIPT="run_wandb_sweep_agent.py"
if [[ "${ARCH}" == "transformer" ]]
then
    SCRIPT=run_wandb_transformer_sweep_agent.py
fi

# UPDATE: We name the experiment as a unix timestamp in order to 
# avoid PTL directory clashes when it reads version at the same time.
# This is due to running parallel jobs on slow I/O on our server.
for run in 1 2 3; do
    python scripts/train_wandb_sweep_agent.py \
        --sweep_id "${SWEEP_ID}" \
        --arch "${ARCH}" \
        --target_col 2 \
        --features_col "${FEATURES_COL}" \
        --accelerator gpu \
        --experiment "$(date +%s)" \
        --wandb_project "${WANDB_PROJECT}" \
        --train "${TRAIN}" \
        --dev "${DEV}" \
        --max_epochs 800 \
        --max_batch_size 512 \
        --patience 50 \
        --save_top_k 1 \
        --check_val_every_n_epoch 4 \
        --model_dir "${RESULTS_PATH}" \
        --seed 42 ;
done