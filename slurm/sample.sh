#!/bin/bash -l

#SBATCH -A project_name
#SBATCH -M node_name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J sample
#SBATCH -o sample.out
#SBATCH -e sample.err
#SBATCH --gres=gpu:1

source activate /path/to/conda/env

# output_dir is to store results
python path/to/pyfile.py -o $output_dir
