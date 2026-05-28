#!/bin/bash

set -xv

expnames=("cglo" "foam" "glor" "oras")
year_end=("2019")
methods=("original" "detrended")
for e in ${expnames[@]};do
	for y in ${year_end[@]};do
		for m in ${methods[@]};do
			sbatch sub_clust.sh ${e} ${y} ${m}
		done
	done
done
