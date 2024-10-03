#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J word_embeddings
#SBATCH -o word_embeddings.out
#SBATCH -e word_embeddings.err
#SBATCH --gres=gpu:1

conda activate /proj/uppmax2024-2-13/hapham/envs/power-identification
cd /proj/uppmax2024-2-13/hapham/power-identification


# Change path to python script
python experiments/feature_engineering/experiment_word_embeddings.py
