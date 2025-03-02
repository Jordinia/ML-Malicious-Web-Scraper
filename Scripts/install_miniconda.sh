#!/usr/bin/bash

# Define variables
HOME_DIR="$HOME"

echo "Changing to home directory..."
cd "$HOME_DIR" || { echo "Error: Failed to change directory to $HOME_DIR"; exit 1; }

sudo apt-get update
sudo apt-get install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash ./Miniconda3-latest-Linux-x86_64.sh
source miniconda3/bin/activate
echo "Setup completed successfully."