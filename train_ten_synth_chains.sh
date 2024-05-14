#!/bin/bash
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 0-60:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu              # Partition to submit to
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=32000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Scenario 1 on Slack

# Load modules
# module load ncf/1.0.0-fasrc01
# module load miniconda3/py310_22.11.1-1-linux_x64-ncf
# module load cuda/12.2.0-fasrc01

# # Activate conda env
# source ~/.bashrc
# conda activate ten

# Train EGNN
python src/main.py \
    --dataset synthetic_chains \
    --task_type classification \
    --lifters "atom:0" "synth1:1" \
    --chain_length 4 \
    --max_path_length 1 \
    --visible_dims 0 1 \
    --dim 1 \
    --num_layers 1 \
    --connectivity "self" \
    --neighbor_types +1 \
    --epochs 200 \
    --batch_size 2 \
    --weight_decay 1e-16 \
    --lr 5e-4 \
    --num_hidden 128 \
    --model_name ten