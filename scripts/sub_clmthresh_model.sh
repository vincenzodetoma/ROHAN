#!/bin/bash

#SBATCH -p vm64
#SBATCH --job-name "anal"
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=vincenzo.detoma@artov.ismar.cnr.it
#SBATCH --mail-type=FAIL

source $HOME/load_mambaforge.source

conda activate ocean_CPU

set -xv

python climthresh_model.py
