#!/bin/bash


#SBATCH -p vm64
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=vincenzodetoma.vdt@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --job-name=mhw_det


source $HOME/.bashrc
WhoAmI=`hostname`
source $HOME/load_mambaforge.source
if [[ ${WhoAmI} = *'hpc01'* ]];then
        echo "Activating oceanTorch"
        conda activate oceanTorch
fi
if [[ ${WhoAmI} = *'vm01'* ]];then
        echo "Activating skywalker02"
        conda activate skywalker02
fi
which python

expname=$1
levidx=$2
case=$3

export PYTHONUNBUFFERED=1

PYTHONUNBUFFERED=1 python -u rean_v1.py --filename=thetao_${expname}_d1993_2019_levidx${levidx}_${case}.nc --sst_name=thetao_${expname} || exit -36

