#!/bin/bash

SWEEP_ID="9s2g6n56"
PROJECT_NAME="hungarian-sig-2017-inflection-attentive_lstm"
lang="hungarian"
task="sig-2017-inflection"
arch="attentive_lstm"


sbatch --export=ALL,SWEEP_ID=${SWEEP_ID},PROJECT_NAME=${PROJECT_NAME},task=${task},lang=${lang},arch=${arch} scripts/compute_test.sh