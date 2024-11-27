#!/bin/bash
#SBATCH -c 16                           # Number of cores (-c)
#SBATCH -t 0-3:00                       # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sapphire                     # Partition to submit to
#SBATCH --mem=32000                     # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Dataset generation script for 3
mkdir -p job_outputs

# Load modules
module load ncf/1.0.0-fasrc01
module load miniconda3/py310_22.11.1-1-linux_x64-ncf
module load cuda/12.2.0-fasrc01

# Activate conda env
source ~/.bashrc
conda activate etnn

# DEFINE EXP ARGUMENTS
LIFTERS=(atom:0 bond:1 ring:2 supercell:3)
NEIGHBOR_TYPES="max"
CONNECTIVITY="self"
VISIBLE_DIMS=(0 1 2)
INITIAL_FEATURES="hetero"
DIM=3

# Command to generate dataset
python src/create_dataset.py --lifters "${LIFTERS[@]}" \
                             --neighbor_types "$NEIGHBOR_TYPES" \
                             --connectivity "$CONNECTIVITY" \
                             --visible_dims "${VISIBLE_DIMS[@]}" \
                             --initial_features "$INITIAL_FEATURES" \
                             --dim "$DIM" \
                            