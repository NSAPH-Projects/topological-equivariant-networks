#!/bin/bash

# Dataset generation script for experiment_9

# DEFINE EXP ARGUMENTS
LIFTERS=(atom:0 bond:1 ring:2 functional_group:2)
NEIGHBOR_TYPES="['+1', '-1', 'min', 'max']"
CONNECTIVITY="all_to_all"
VISIBLE_DIMS=(0 1 2)
INITIAL_FEATURES="hetero"
DIM=2

# Command to generate dataset
python src/create_dataset.py --lifters "${LIFTERS[@]}" \
                             --neighbor_types "$NEIGHBOR_TYPES" \
                             --connectivity "$CONNECTIVITY" \
                             --visible_dims "${VISIBLE_DIMS[@]}" \
                             --initial_features "$INITIAL_FEATURES" \
                             --dim "$DIM" \
                            