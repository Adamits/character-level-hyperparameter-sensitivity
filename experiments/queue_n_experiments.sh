#!/bin/bash

if [[ $# -lt 7 ]] ; then
    echo "ERROR: Expected 7 arguments: K, NODE, TASK, LANGUAGE, ARCH  WANDB_PROJECT"
    exit 1
fi

K=$1
NODE=$2
TASK=$3
LANGUAGE=$4
ARCH=$5
SWEEP_ID=$6
WANDB_PROJECT=$7

if [ "$NODE" = "BLANCA_GPU" ]; then
    for i in $(seq 1 $K); do
        sbatch --export=ALL,TASK=${TASK},LANGUAGE=${LANGUAGE},ARCH=${ARCH},SWEEP_ID=${SWEEP_ID},WANDB_PROJECT=${WANDB_PROJECT} experiments/train_sweep_agent.sh;
    done
elif [ "$NODE" = "ALPINE_GPU" ]; then
    for i in $(seq 1 $K); do
        sbatch --export=ALL,TASK=${TASK},LANGUAGE=${LANGUAGE},ARCH=${ARCH},WANDB_CACHE_DIR=/scratch/alpine/adwi9965/cache/wandb experiments/train_alpine_sweep_agent.sh;
    done
else 
    echo "ERROR: Argument NODE must be one of BLANCA_GPU, or ALPINE_GPU."
    echo "Not $NODE."
    exit 1
fi
