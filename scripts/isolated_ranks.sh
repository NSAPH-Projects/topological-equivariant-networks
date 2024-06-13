#!/bin/bash
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 0-4:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu              # Partition to submit to
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=32000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# DESCRIPTION
# This is the EGNN-LIKE benchmark, but trained on 100 epochs instead of 1000. 
# It is trained on alpha, with learning rate 5e-4.


# Load modules
module load ncf/1.0.0-fasrc01
module load miniconda3/py310_22.11.1-1-linux_x64-ncf
module load cuda/12.2.0-fasrc01

# Activate conda env
source ~/.bashrc
conda activate ten

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
