#!/bin/bash

# Dataset generation script for experiment_10

# DEFINE EXP ARGUMENTS
LIFTERS=(atom:0 supercell:1)
NEIGHBOR_TYPES="max"
CONNECTIVITY="self"
VISIBLE_DIMS=(0)
INITIAL_FEATURES="hetero"
DIM=1

# Command to generate dataset
python src/create_dataset.py --lifters "${LIFTERS[@]}" \
                             --neighbor_types "$NEIGHBOR_TYPES" \
                             --connectivity "$CONNECTIVITY" \
                             --visible_dims "${VISIBLE_DIMS[@]}" \
                             --initial_features "$INITIAL_FEATURES" \
                             --dim "$DIM" \
                            