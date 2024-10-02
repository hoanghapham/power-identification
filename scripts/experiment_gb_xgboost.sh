#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J gb_xgboost
#SBATCH -o gb_xgboost.out
#SBATCH -e gb_xgboost.err
#SBATCH --gres=gpu:1

conda activate /proj/uppmax2020-2-2/hapham/envs/power-identification
cd /proj/uppmax2020-2-2/hapham/power-identification


# Change path to python script
python experiments/classic_ml/experiment_gb_xgboost.py
