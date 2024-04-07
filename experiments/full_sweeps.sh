#!/bin/bash

# Based on what we have in SWEEPS.csv: the new sweeps we have generated.

# RANDOM SAMPLE:
# /rc_scratch/adwi9965/sig-2017-inflection/romanian-train-high
# /rc_scratch/adwi9965/sig-2017-inflection/bulgarian-train-high
# /rc_scratch/adwi9965/sig-2017-inflection/haida-train-high
# /rc_scratch/adwi9965/sig-2017-inflection/irish-train-high
# /rc_scratch/adwi9965/sig-2017-inflection/dutch-train-high
# /rc_scratch/adwi9965/sig-2017-inflection/lower-sorbian-train-high
# /rc_scratch/adwi9965/sig-2017-inflection/catalan-train-high
# /rc_scratch/adwi9965/sig-2017-inflection/swedish-train-high
# /rc_scratch/adwi9965/sig-2017-inflection/macedonian-train-high
#/rc_scratch/adwi9965/sig-2017-inflection/welsh-train-high


declare -A sweeps
# inflection
sweeps["catalan-sig-2017-inflection-attentive_lstm"]="ash3qt3o"
sweeps["catalan-sig-2017-inflection-transformer"]="b3sniwpy"
sweeps["romanian-sig-2017-inflection-attentive_lstm"]="4yna0dme"
sweeps["romanian-sig-2017-inflection-transformer"]="1u212pv3"
sweeps["haida-sig-2017-inflection-attentive_lstm"]="uxy33v9n"
sweeps["haida-sig-2017-inflection-transformer"]="972nfftl"
sweeps["irish-sig-2017-inflection-attentive_lstm"]="8cavqlmd"
sweeps["irish-sig-2017-inflection-transformer"]="t5z4urum"
sweeps["dutch-sig-2017-inflection-attentive_lstm"]="4sx1o9ba" 
sweeps["dutch-sig-2017-inflection-transformer"]="webnkynn"
sweeps["hungarian-sig-2017-inflection-attentive_lstm"]="9s2g6n56"
sweeps["hungarian-sig-2017-inflection-transformer"]="k9bhthrj"
sweeps["albanian-sig-2017-inflection-attentive_lstm"]="r98lf0lh"
sweeps["albanian-sig-2017-inflection-transformer"]="tga5csbe"
sweeps["arabic-sig-2017-inflection-attentive_lstm"]="2nvum3vr"
sweeps["arabic-sig-2017-inflection-transformer"]="n5rr4pjz"
# g2p
sweeps["dut-sig-2021-g2p-attentive_lstm"]="d2hcrze6"
sweeps["dut-sig-2021-g2p-transformer"]="abjpav1y"
sweeps["hun-sig-2021-g2p-attentive_lstm"]="hn8s4qpp"
sweeps["hun-sig-2021-g2p-transformer"]="3gajztgm"
sweeps["arm_e-sig-2021-g2p-attentive_lstm"]="e3hejkhy"
sweeps["arm_e-sig-2021-g2p-transformer"]="ggpuzbal"
sweeps["geo-sig-2021-g2p-attentive_lstm"]="3fpdfbrl"
sweeps["geo-sig-2021-g2p-transformer"]="n4roktcx"

# K
# NODE
# TASK
# LANGUAGE
# ARCH
# SWEEP_ID
# WANDB_PROJECT
# Inflection

# Inflection
for task in sig-2017-inflection; do
    for lang in catalan romanian haida irish dutch hungarian albanian arabic; do
        for arch in attentive_lstm transformer; do
            project_name=${lang}-${task}-${arch}
            sweep_id="${sweeps[$project_name]}"
            bash experiments/queue_n_experiments.sh \
                200 \
                BLANCA_GPU \
                ${task} \
                ${lang} \
                ${arch} \
                ${sweep_id} \
                ${project_name}
        done
    done
done

# G2P
for task in sig-2021-g2p; do
    for lang in hun dut arm_e geo; do
        for arch in attentive_lstm transformer; do
            project_name=${lang}-${task}-${arch}
            sweep_id="${sweeps[$project_name]}"
            bash experiments/queue_n_experiments.sh \
                20 \
                BLANCA_GPU \
                ${task} \
                ${lang} \
                ${arch} \
                ${sweep_id} \
                ${project_name}
        done
    done
done
