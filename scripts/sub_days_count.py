#!/bin/bash

#SBATCH -p cluster
#SBATCH --job-name "analysis"
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=3-00:00:00

source $HOME/load_mambaforge.source

conda activate ocean_CPU

python days_cats.py
