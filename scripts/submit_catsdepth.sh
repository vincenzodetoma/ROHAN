#!/bin/bash


#SBATCH -p vm64
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=vincenzodetoma.vdt@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=cats_depth
set -x

source $HOME/.bashrc
source $HOME/load_mambaforge.source

conda activate ocean_CPU

python cat_vs_depth.py || exit -13