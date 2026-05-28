#!/bin/bash

exps=(
    "ESACCISST"
    "cglo"
    "foam"
    "glor"
    "oras"
)

levidxs=(
    "1"
    #"9"
    #"15"
    #"19"
    #"25"
    #"28"
    #"31"
)

areas=(
    "NorthWestAtlantic"
    "SantaBarbara"
    "GreatBarrierReef"
    "TasmanSea"
    "MediterraneanSea"
    "NortheastPacBlob"
    "WestAust"
    "NorthernHemisphere"
    "SouthernHemisphere"
)

for exp_idx in "${!exps[@]}"; do
    exp="${exps[$exp_idx]}"
    if [ "$exp" == "ESACCISST" ]; then
        levidx_list=("${levidxs[0]}")
    else
        levidx_list=("${levidxs[@]}")
    fi
    for levidx in "${levidx_list[@]}"; do
        for area in "${areas[@]}"; do
            python make_mhw_anomaly_video.py \
                --fps=100 \
                --dpi=300 \
                --area="$area" \
                --dataset="$exp" \
                --level=levidx${levidx}
        done
    done
done
