#!/bin/bash
#SBATCH --job-name=mpii
#SBATCH --gres=gpu:1
#SBATCH --mem=50Gb
#SBATCH --partition=gpu
#SBATCH --output=mpii.%j.out
#SBATCH --error=mpii.%j.err

# Change to root project directory
RUN_SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR="$(dirname "$RUN_SCRIPTS_DIR")"
cd $PROJECT_DIR

# Run train script
CUDA_VISIBLE_DEVICES=0 python2 ./example/debug.py \
--dataset mpii \
--arch hg \
--stack 2 \
--block 1 \
--features 256 \
--checkpoint ./checkpoint/mpii/debug \
--anno-path data/mpii/mpii_annotations.json \
--image-path /scratch/sehgal.n/datasets/mpii/images \
-d

