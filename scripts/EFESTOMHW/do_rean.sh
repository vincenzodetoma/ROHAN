#!/bin/bash

WhoAmI=`hostname`
source $HOME/load_mambaforge.source
if [[ ${WhoAmI} = *'hpc01'* ]];then
        echo "Activating oceanTorch"
        conda activate oceanTorch
        exe="sbatch"
        list_proc="squeue"
        user="vdetoma"
        nameproc="mhw_det"
        status="awk '{ print $5 }'"
fi
if [[ ${WhoAmI} = *'vm01'* ]];then
        echo "Activating skywalker02"
        conda activate skywalker02
        exe="sh"
        list_proc="ps -ef"
        user="4745"
        nameproc="python"
        status="rean_v1"
fi

exps=("ESACCISST") # "cglo" "foam" "glor" "oras")
levidxs=("1") # "9" "15" "19" "25" "28" "31")
case=("r1x1") # "r1x1_detrend")
for e in ${exps[@]};do
	for l in ${levidxs[@]};do
		for c in ${case[@]};do
			echo ${list_proc} ${user} ${nameproc} ${status}
			myS=( $(`${list_proc} | grep ${user} | grep ${nameproc} | grep ${status}`) )
	                echo ${myS}
			while [ ${#myS[@]} != 0 ]
        	        do
                	        sleep 10
                        	myS=( $(`${list_proc} | grep ${user} | grep ${nameproc} | grep ${status}`) )
                	done
                	echo "Doing experiment ${e}, level ${l}, case ${c}"
			${exe} sub_rean.sh $e $l $c #> $e$l$c.log 2>&1 
		done
	done
done
