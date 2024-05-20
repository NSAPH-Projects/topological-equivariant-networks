#!/bin/bash

# DEFINE EXP ARGUMENTS
LIFTERS=("atom:0" "bond:1" "supercell:2")
DIM=2
VISIBLE_DIMS=(0 1)
INITIAL_FEATURES="node"
NEIGHBOR_TYPES="max"
CONNECTIVITY="self"
EPOCHS=350
NUM_HIDDEN=

# Constants
BATCH_SIZE=96
WEIGHT_DECAY=1e-16
MIN_LR=0
NUM_LAYERS=7
MODEL_NAME="ten"
SPLITS="egnn"
CHECKPOINT_DIR="/content/drive/MyDrive/checkpoints"


# Define the predefined target names lists
TARGET_NAMES_0=("alpha")
TARGET_NAMES_1=("mu" "alpha" "homo" "lumo" "gap" "r2")
TARGET_NAMES_2=("zpve" "U0" "U" "H" "G" "Cv")

# Check if an argument is provided; if not, use 0 as the default option
OPTION=${1:-0}

# Select the target names list based on the provided option
case $OPTION in
  0)
    TARGET_NAMES=("${TARGET_NAMES_0[@]}")
    ;;
  1)
    TARGET_NAMES=("${TARGET_NAMES_1[@]}")
    ;;
  2)
    TARGET_NAMES=("${TARGET_NAMES_2[@]}")
    ;;
  *)
    echo "Invalid option. Please provide 0, 1, or 2."
    exit 1
    ;;
esac

for TARGET_NAME in "${TARGET_NAMES[@]}"
do
    # Check if target is one of the first three
    if [[ "$target" == "gap" || "$target" == "homo" || "$target" == "lumo" ]]; then
        LR="1e-3"
    else
        LR="5e-4"
    fi

    echo "Launching job for target: $TARGET_NAME with LR: $LR"

    # Train EGNN in parallel for each TARGET_NAME
    python src/main_qm9.py --lifters "${LIFTERS[@]}" \
                           --dim "$DIM" \
                           --visible_dims "${VISIBLE_DIMS[@]}" \
                           --initial_features "$INITIAL_FEATURES" \
                           --target_name "$TARGET_NAME" \
                           --neighbor_types "$NEIGHBOR_TYPES" \
                           --connectivity "$CONNECTIVITY" \
                           --epochs "$EPOCHS" \
                           --batch_size "$BATCH_SIZE" \
                           --weight_decay "$WEIGHT_DECAY" \
                           --lr "$LR" \
                           --min_lr "$MIN_LR" \
                           --num_layers "$NUM_LAYERS" \
                           --num_hidden "$NUM_HIDDEN" \
                           --model_name "$MODEL_NAME" \
                           --splits "$SPLITS" \
                           --normalize_invariants \
                           --clip_gradient \
                           --checkpoint_dir "$CHECKPOINT_DIR" &

    # Wait for 30 minutes before moving to the next iteration
    sleep 1800
done

# Wait for all background jobs to finish
echo "All jobs have been launched."
wait
echo "All jobs have been completed."

