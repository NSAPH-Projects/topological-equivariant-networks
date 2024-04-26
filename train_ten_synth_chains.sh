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
module load ncf/1.0.0-fasrc01
module load miniconda3/py310_22.11.1-1-linux_x64-ncf
module load cuda/12.2.0-fasrc01

# Activate conda env
source ~/.bashrc
conda activate ten

# Train EGNN
python src/main_qm9.py --lifters "path:c"  \
                       --max_path_length 3 \
                       --connectivity "self" \
                       --visible_dims 0 1 2 3 \
                       --neighbor_types "-1" \
                       --epochs 1000 \
                       --batch_size 2 \
                       --weight_decay 1e-16 \
                       --lr "$LR" \
                       --num_layers 1 \
                       --num_hidden 128 \
                       --model_name "ten" \
                       --dim 3 \
                       --splits "egnn" \