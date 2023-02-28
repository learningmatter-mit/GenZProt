#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --time=02:00:00
#SBATCH -c 20
#SBATCH -o run_test.sh.log-%j
#SBATCH --gres=gpu:volta:1

module load cuda/11.0

ped_id=00055
MPATH=./m1_01-19_123
python test_model.py -load_model_path $MPATH -test_data $ped_id -test_epoch 66