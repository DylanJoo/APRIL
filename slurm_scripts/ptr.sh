#!/bin/sh
#SBATCH --job-name=ptr
#SBATCH --partition gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate april

# root
cd ~/APRIL/src

python3 run_ptr.py
