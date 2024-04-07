#!/bin/bash

#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=24gb
#SBATCH --time=6:00:00
#SBATCH --qos=blanca-kann
##SBATCH --qos=preemptable
#SBATCH --constraint=rtx6000|A100|A40|V100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_compute.%j.log

# Works with bash version:
# GNU bash, version 4.2.46(2)-release (x86_64-redhat-linux-gnu)

source /curc/sw/anaconda3/latest
conda activate hyperparameter-sensitivity
# FIXME: Move all results to alpine, and change paths to /scratch/alpine
#       it has a larger cap on how disk space!
# DISK_PARTITION=/scratch/alpine
DISK_PARTITION=/rc_scratch
readonly ROOT="/projects/adwi9965/yoyodyne"

declare -A sweeps
# inflection
sweeps["romanian-sig-2017-inflection-attentive_lstm"]="4yna0dme"
sweeps["romanian-sig-2017-inflection-transformer"]="1u212pv3"
sweeps["haida-sig-2017-inflection-attentive_lstm"]="uxy33v9n"
sweeps["haida-sig-2017-inflection-transformer"]="972nfftl"
sweeps["irish-sig-2017-inflection-attentive_lstm"]="8cavqlmd"
sweeps["irish-sig-2017-inflection-transformer"]="t5z4urum"

sweeps["armenian-sig-2017-inflection-attentive_lstm"]="4gwonwv9"
sweeps["armenian-sig-2017-inflection-transformer"]="cpyl44in"
sweeps["basque-sig-2017-inflection-attentive_lstm"]="iqhxceh9"
sweeps["basque-sig-2017-inflection-transformer"]="z9xbjbyy"
sweeps["bengali-sig-2017-inflection-attentive_lstm"]="gkt8afvu"
sweeps["bengali-sig-2017-inflection-transformer"]="0ncpq5el"
sweeps["czech-sig-2017-inflection-attentive_lstm"]="bfrw01q3"
sweeps["czech-sig-2017-inflection-transformer"]="kuxokdhi"
# FINISHED.
sweeps["dutch-sig-2017-inflection-attentive_lstm"]="4sx1o9ba" 
sweeps["dutch-sig-2017-inflection-transformer"]="webnkynn"
sweeps["hungarian-sig-2017-inflection-attentive_lstm"]="9s2g6n56"
sweeps["hungarian-sig-2017-inflection-transformer"]="k9bhthrj"
sweeps["albanian-sig-2017-inflection-attentive_lstm"]="r98lf0lh"
sweeps["albanian-sig-2017-inflection-transformer"]="tga5csbe"
sweeps["arabic-sig-2017-inflection-attentive_lstm"]="2nvum3vr"
sweeps["arabic-sig-2017-inflection-transformer"]="n5rr4pjz"
sweeps["catalan-sig-2017-inflection-attentive_lstm"]="ash3qt3o"
sweeps["catalan-sig-2017-inflection-transformer"]="b3sniwpy"

# g2p
# FINISHED.
sweeps["dut-sig-2021-g2p-attentive_lstm"]="d2hcrze6"
sweeps["dut-sig-2021-g2p-transformer"]="abjpav1y"
sweeps["hun-sig-2021-g2p-attentive_lstm"]="hn8s4qpp"
sweeps["hun-sig-2021-g2p-transformer"]="3gajztgm"
sweeps["geo-sig-2021-g2p-attentive_lstm"]="3fpdfbrl"
sweeps["geo-sig-2021-g2p-transformer"]="n4roktcx"
sweeps["arm_e-sig-2021-g2p-attentive_lstm"]="e3hejkhy"
sweeps["arm_e-sig-2021-g2p-transformer"]="ggpuzbal"


get_test_acc () {
    SWEEP_ID=$1
    PROJECT_NAME=$2
    lang=$3
    task=$4
    arch=$5

    echo $lang $task $arch
    BEST_RUNS_PATH=/projects/adwi9965/hyperparameter-sensitivity/BEST_RUNS.tsv
    DATA="${DISK_PARTITION}/adwi9965/${task}"
    PREDS_ROOT="/projects/adwi9965/hyperparameter-sensitivity/predictions/${task}-${arch}-${lang}"
    # For a given sweep
    #   1. Gets the runs dataframe using W&B API, and writes it as a tsv
    mkdir -p sweep_results
    mkdir -p $PREDS_ROOT

    if [[ "${task}" == "sig-2021-g2p" ]]
    then
        # Gold, uncovered file to predict and evaluate against
        PREDICT="${DATA}/medium/${lang}_test.tsv"
        FEATURES_COL=0
    elif [[ "${task}" == "sig-2017-inflection" ]]
    then
        # Gold, uncovered file to predict and evaluate against
        PREDICT="${DATA}/${lang}-uncovered-test"
        FEATURES_COL=3
    else
        echo No task ${task}, only "sig-2021-g2p" and "sig-2017-inflection" are implemented
        exit 1;
    fi

    run_path=$(cat $BEST_RUNS_PATH | grep ${lang}-${task}-${arch} | cut -f2)
    # Paths point to rc_scratch, but we've moved storage to alpine.
    # run_path=$(echo $run_path | sed "s/\/rc_scratch/\/scratch\/alpine/g")
    run_id=$(echo $run_path | rev | cut -d'/' -f2 | rev)
    # index="${run_path}/index.pkl"
    model_dir=$(dirname $(dirname $run_path))
    checkpoint="${run_path}/checkpoints"
    # Only one model per checkpoint dir
    model_name=$(ls $checkpoint)
    # Full checkpoint path.
    checkpoint="${checkpoint}/${model_name}"
    # Output file to write predictions
    output="${PREDS_ROOT}/${run_id}"

    echo "Doing inference with args:"
    echo $arch
    echo $run_id
    echo $PREDICT
    echo $output
    echo $checkpoint
    echo $FEATURES_COL
    #   2. Gets test predictions on the model in the dataframe for each run
    yoyodyne-predict \
        --arch $arch \
        --experiment $run_id \
        --predict $PREDICT \
        --output $output \
        --model_dir $model_dir \
        --checkpoint $checkpoint \
        --batch_size 100 \
        --features_col $FEATURES_COL \
        --target_col 0 ;

    #   3. Evaluates the test predictions
    accuracy=$(python scripts/evaluate.py --gold ${PREDICT} --predicted $output)
    accs_filepath="sweep_results/test_preds.tsv" #$(echo $output_filepath | sed "s/\.tsv/_test_acc.tsv/g")
    echo -e "${lang}\t${arch}\t${task}\t${run_path}\t${accuracy}" >> $accs_filepath;
}
# Inflection
#for task in sig-2017-inflection; do
#    for lang in irish; do #romanian haida irish dutch hungarian albanian arabic catalan; do 
#        for arch in attentive_lstm transformer; do
#            project_name="${lang}-${task}-${arch}"
#            sweep_id="${sweeps[$project_name]}"
#            get_test_acc $sweep_id $project_name $lang $task $arch
#        done;
#    done;
#done

# G2P
for task in sig-2021-g2p; do
    for lang in geo arm_e dut hun; do 
        for arch in attentive_lstm transformer; do
            project_name="${lang}-${task}-${arch}"
            sweep_id="${sweeps[$project_name]}"
            get_test_acc $sweep_id $project_name $lang $task $arch
        done;
    done;
done
#   5. Writes a new CSV with preds_paths and test_accs
#      TODO
