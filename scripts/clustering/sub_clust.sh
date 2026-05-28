#!/bin/bash


#SBATCH -p vm64
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mail-user=vincenzodetoma.vdt@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=clustering
#SBATCH --time=10-00:00:00
set -xv

source $HOME/.bashrc
source $HOME/load_mambaforge.source

conda activate oceanTorch

varname=$1
year_end=$2
preproc=$3

python cluster_events.py ${varname} ${year_end} ${preproc} || exit -36

