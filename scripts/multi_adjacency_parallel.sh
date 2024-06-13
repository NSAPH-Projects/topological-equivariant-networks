LR="5e-4"

for TARGET_NAME in "alpha" "H"
do
  for LIFTER in "bond" "ring" "functional_group"
  do
    # Train EGNN in parallel for each LIFTER
    python src/main_qm9.py --lifters "atom:0" "$LIFTER:1" "supercell:2" \
                           --dim 2 \
                           --visible_dims 0 \
                           --initial_features "hetero" \
                           --target_name "$TARGET_NAME" \
                           --neighbor_types "+1" "max" \
                           --connectivity "self" \
                           --epochs 100 \
                           --batch_size 96 \
                           --weight_decay 1e-16 \
                           --lr "$LR" \
                           --min_lr 0 \
                           --num_layers 7 \
                           --num_hidden 128 \
                           --model_name "ten" \
                           --splits "egnn" &
  done
  
  # Wait for all background processes from the inner loop to finish before proceeding to the next TARGET_NAME
  wait
done
