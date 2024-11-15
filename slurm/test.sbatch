#!/bin/bash
#SBATCH -t 00:00:10
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem 16G
#SBATCH -p gpu
#SBATCH -o slurm/logs/%j.out
#SBATCH -e slurm/logs/%j.err
#SBATCH --mail-type=ALL

source activate etnn
module load cuda

export PYTORCH_JIT=1
export HYDRA_FULL_ERROR=1

srun python main_pm25.py \
    baseline=topological \
    arch=arch-cc2-2 \
    lr=high \
    training.max_epochs=1 \
    force_restart=true
