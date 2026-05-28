#!/bin/bash


#SBATCH -p vm64
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=vincenzodetoma.vdt@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --job-name=mhw_det

source $HOME/.bashrc
WhoAmI=`hostname`
source $HOME/load_mambaforge.source
if [[ ${WhoAmI} = *'hpc01'* ]];then
        echo "Activating ocean_CPU"
        conda activate oceanTorch
fi
if [[ ${WhoAmI} = *'vm01'* ]];then
        echo "Activating skywalker02"
        conda activate skywalker02
fi
which python

echo "job running with a total number of task: " ${SLURM_NTASKS}

python main.py --data_dir=/store/data_store/ESA-CCI_SST_V3/ --filename=pippo --sst_name=analysed_sst --DetPeriod='["1980-01-01","2024-05-31"]' --n_work=${SLURM_NTASKS} --thre=${SLURM_CPUS_PER_TASK} --mem_lim='6GB' --mem_per_chunk='128MB'
