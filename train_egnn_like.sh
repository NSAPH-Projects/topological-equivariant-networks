#!/bin/bash
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 0-20:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu              # Partition to submit to
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=32000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Load modules
module load ncf/1.0.0-fasrc01
module load miniconda3/py310_22.11.1-1-linux_x64-ncf
module load cuda/12.2.0-fasrc01

# Activate conda env
source ~/.bashrc
conda activate ten

# Train EGNN
python src/main.py --dataset "qm9" \
                   --lifters "atom:0" "supercell:1" \
                   --target_name "$TARGET_NAME" \
                   --neighbor_types "+1" \
                   --merge_neighbors \
                   --connectivity "self" \
                   --visible_dims 0 \
                   --epochs 1000 \
                   --batch_size 96 \
                   --weight_decay 1e-16 \
                   --lr "$LR" \
                   --num_layers 7 \
                   --num_hidden 128 \
                   --model_name "ten" \
                   --dim 1 \
                   --compile \
                   --splits "egnn" \