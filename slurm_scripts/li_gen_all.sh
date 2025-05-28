#!/bin/sh
#SBATCH --job-name=li-all
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate april

# root
cd ~/APRIL/src

python3 run_li_gen_prob.py
python3 run_li_gen_list.py
