#!/bin/sh
#SBATCH --job-name=pt-api
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=%x-%j.out


# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate april

# root
cd ~/APRIL/src

python3 run_pt_prob.py
