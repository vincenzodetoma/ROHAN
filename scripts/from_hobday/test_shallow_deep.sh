#!/bin/bash

set -x 

which python 
cd mhw_detector/

variables=(
    "duration"
    "rate_decline"
    "rate_onset"
    "category"
    "intensity_cumulative"
    "intensity_var"
)

for v in "${variables[@]}"; do
    echo "Processing variable: $v"
    PYTHONUNBUFFERED=1 python percentage_shallow_deep.py \
        --varname="$v"
done