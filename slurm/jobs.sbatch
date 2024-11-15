#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem 16G
#SBATCH -p gpu
#SBATCH -o slurm/logs/%j.out
#SBATCH -e slurm/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --array=1-4

source activate etnn
module load cuda

export PYTORCH_JIT=1
export HYDRA_FULL_ERROR=1


for i in $(seq $((4 * SLURM_ARRAY_TASK_ID)) $((4 * (SLURM_ARRAY_TASK_ID + 1)) - 1)); do
  seed=$((42 * i))
  for arch in arch-cc2-2 arch-cc2-1; do
    for lr in low high; do
      for baseline in linear mlp-large topological gnn egnn-equivariant invariant equivariant-ho equivariant-virtual; do
        echo "baseline=$baseline, seed=$seed, arch=$arch"
        (
          python main_pm25.py \
            baseline=$baseline \
            seed=$seed \
            arch=$arch \
            lr=$lr \
            ckpt_prefix=cc3-fix \
            force_restart=true
        ) &
        sleep 3
      done
    done
    wait
  done
done