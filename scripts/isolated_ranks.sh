TARGET_NAME="alpha"
LR="5e-4"

# Train EGNN
python src/main_qm9.py --lifters "atom:0" "functional_group:1" "supercell:2" \
                       --dim 2 \
                       --visible_dims 0 1 \
                       --initial_features "hetero" \
                       --target_name "$TARGET_NAME" \
                       --neighbor_types "max" \
                       --connectivity "self" \
                       --epochs 100 \
                       --batch_size 96 \
                       --weight_decay 1e-16 \
                       --lr "$LR" \
                       --min_lr 0 \
                       --num_layers 7 \
                       --num_hidden 128 \
                       --model_name "ten" \
                       --compile \
                       --splits "egnn" \
