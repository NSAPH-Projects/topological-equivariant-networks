#!/bin/bash

# Dataset generation script for experiment_6

# DEFINE EXP ARGUMENTS
LIFTERS=(atom:0 bond:1 supercell:2)
NEIGHBOR_TYPES="max"
CONNECTIVITY="self"
VISIBLE_DIMS=(0 1)
INITIAL_FEATURES="node"
DIM=2

# Command to generate dataset
python src/create_dataset.py --lifters "${LIFTERS[@]}" \
                             --neighbor_types "$NEIGHBOR_TYPES" \
                             --connectivity "$CONNECTIVITY" \
                             --visible_dims "${VISIBLE_DIMS[@]}" \
                             --initial_features "$INITIAL_FEATURES" \
                             --dim "$DIM" \
                            