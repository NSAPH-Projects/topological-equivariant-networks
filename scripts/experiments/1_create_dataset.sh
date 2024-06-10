#!/bin/bash

# isolated ranks: atoms + bonds

# DEFINE EXP ARGUMENTS
LIFTERS=("atom:0" "bond:1" "supercell:2")
NEIGHBOR_TYPES="max"
CONNECTIVITY="self"
VISIBLE_DIMS=(0 1)
INITIAL_FEATURES="hetero"
DIM=2


# Train EGNN in parallel for each TARGET_NAME
python src/create_dataset.py --lifters "${LIFTERS[@]}" \
                             --neighbor_types "$NEIGHBOR_TYPES" \
                             --connectivity "$CONNECTIVITY" \
                             --visible_dims "${VISIBLE_DIMS[@]}" \
                             --initial_features "$INITIAL_FEATURES" \
                             --dim "$DIM" \
                       
                        
                        
                        


