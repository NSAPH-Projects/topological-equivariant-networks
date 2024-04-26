#!/bin/bash

# Default values
clip_gradient=False

# Parse command-line options
while getopts c: option
do
    case "${option}"
    in
    c) clip_gradient=${OPTARG};;
    esac
done

# Remove the parsed options from the command-line arguments
shift $((OPTIND -1))

# Get the script name from the first argument
script_name=$1
shift

# List of target names
default_targets=("gap" "homo" "lumo" "alpha" "mu" "Cv" "G" "H" "r2" "U" "U0" "zpve")

# If no targets are provided, use the default list
if [ $# -eq 0 ]; then
    targets=("${default_targets[@]}")
else
    targets=("$@")
fi

# Learning rates
lr1="1e-3"
lr2="5e-4"

# Loop over targets
for target in "${targets[@]}"
do
    # Check if target is one of the first three
    if [[ "$target" == "gap" || "$target" == "homo" || "$target" == "lumo" ]]; then
        lr=$lr1
    else
        lr=$lr2
    fi
    # Call the specified script with the current target
    sbatch --export=ALL,TARGET_NAME=$target,LR=$lr,CLIP_GRADIENT=$clip_gradient $script_name
done