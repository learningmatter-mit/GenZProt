#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --time=20:00:00
#SBATCH -o run_train.sh.log-%j
#SBATCH --gres=gpu:volta:1

module load cuda/11.0
python train_model.py -load_json modelparams/multi.json
