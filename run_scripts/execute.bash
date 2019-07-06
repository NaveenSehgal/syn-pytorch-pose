#!/bin/bash
#SBATCH --job-name=audio
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --partition=gpu
#SBATCH --output=audio_model.%j.out
#SBATCH --error=audio_model.%j.err

python3 train_model.py --model model4
