#!/bin/sh
#SBATCH --job-name=llmrerank
#SBATCH --partition v100
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

for code in unittest/listwise*.py;do 
    run=${code##*/}
    output=${run/\.py/.log}
    echo "Running $run"
    python3 $code > $output 2>&1
done

for code in unittest/pairwise*.py;do 
    run=${code##*/}
    output=${run/\.py/.log}
    echo "Running $run"
    python3 $code > $output 2>&1
done

for code in unittest/setwise*.py;do 
    run=${code##*/}
    output=${run/\.py/.log}
    echo "Running $run"
    python3 $code > $output 2>&1
done
