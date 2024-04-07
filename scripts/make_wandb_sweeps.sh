#!/bin/bash

task=sig-2021-g2p
for lang in albanian arabic armenian basque bengali catalan czech romanian haida irish lower-sorbian swedish macedonian welsh; do
   for arch in attentive_lstm transformer; do
       python scripts/make_wandb_sweep.py \
           --project ${lang}-${task}-${arch} \
           --sweep ${lang}-${task}-${arch}-sweep \
           --arch $arch \
           --outpath SWEEPS.csv
   done
done

# task=sig-2021-g2p
for lang in geo hbs_latn jpn_hira kor vie_hanoi bul fre; do
    for arch in attentive_lstm transformer; do
         python scripts/make_wandb_sweep.py \
             --project ${lang}-${task}-${arch} \
             --sweep ${lang}-${task}-${arch}-sweep \
             --arch $arch \
             --outpath SWEEPS.csv
    done
done
