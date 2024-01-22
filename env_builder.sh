#!/bin/bash

# Script Name: env_builder.sh
# Description: 
#   This script automates the process of setting up a Conda environment
#   and installing a package from a Git repository. It checks for Conda,
#   creates or activates the specified environment, clones the repository,
#   and installs the package in editable mode.
#
# Usage: source env_builder.sh
# Requirements:
#   - Conda must be installed and initialized in your system.
#   - Git must be installed and accessible from the command line.
#   - Internet access is required to clone the repository.
#
# Notes:
#   - Ensure the script has executable permissions: chmod +x setup_conda_env.sh

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Conda and retry."
    exit 1
fi

# Activate or create Conda environment
echo "Creating environment 'ten'"
conda env create -f environment.yaml
conda activate ten

# Install TopoNetX
echo "Installing 'TopoNetX'"
REPO_URL="https://github.com/pyt-team/TopoNetX"
git clone "$REPO_URL"
REPO_DIR=$(basename "$REPO_URL" .git)
cd "$REPO_DIR"
git pull
pip install -e '.[all]'
cd ..

echo "Installation completed."