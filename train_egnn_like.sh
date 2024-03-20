#!/bin/bash

# Script to train an EGNN-like model on the QM9 dataset. Run using `source train_egnn_like.sh` 

# The following are missing from the EGNN specification:
# 1. EGNN uses a different data split
# 2. There MAY be slight differences between EGNN and EMPSN layers, this needs to be checked

conda activate ten
python src/main_qm9.py --lifters "atom:0" "supercell:1" \
                       --target_name "alpha" \
                       --neighbor_type "any_adjacency" \
                       --connectivity "self" \
                       --post_pool_filter 0 \
                       --epochs 30 \
                       --batch_size 96 \
                       --weight_decay 1e-16 \
                       --lr 5e-4 \
                       --num_layers 7 \
                       --num_hidden 128 \
                       --model_name "ten" \
                       --dim 1 \
                       --num_samples 1000