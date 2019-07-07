#!/bin/bash
#SBATCH --job-name=scanava
#SBATCH --gres=gpu:1
#SBATCH --mem=50Gb
#SBATCH --partition=gpu
#SBATCH --output=scanava.%j.out
#SBATCH --error=scanava.%j.err

# Change to root project directory
# RUN_SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# PROJECT_DIR="$(dirname "$RUN_SCRIPTS_DIR")"
PROJECT_DIR="/home/sehgal.n/syn-pytorch-pose"
cd $PROJECT_DIR
echo "Current Dir: $PROJECT_DIR"

# Run train script
CUDA_VISIBLE_DEVICES=0 python2 ./example/main.py \
--dataset scanava \
--arch hg \
--stack 2 \
--block 1 \
--features 256 \
--checkpoint ./checkpoint/scanava/test  \
--anno-path data/scanava/scanava_labels.npy \
--image-path /scratch/sehgal.n/datasets/synthetic \

