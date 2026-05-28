#!/bin/bash

exps=(
    "ESACCISST"
    #"cglo" 
    #"foam" 
    #"glor" 
    #"oras"
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
    "Global None None None None output/Global_mhw_mask.nc output/Global_mhw_metrics.nc output/Global_mhw_climatology.nc"
    #"NorthWestAtlantic 35 50 -80 -50 output/NorthWestAtlantic_mhw_mask.nc output/NorthWestAtlantic_mhw_metrics.nc output/NorthWestAtlantic_mhw_climatology.nc"
    #"SantaBarbara 30 37 -125 -115 output/SantaBarbara_mhw_mask.nc output/SantaBarbara_mhw_metrics.nc output/SantaBarbara_mhw_climatology.nc"
    #"GreatBarrierReef -25 -5 135 155 output/GreatBarrierReef_mhw_mask.nc output/GreatBarrierReef_mhw_metrics.nc output/GreatBarrierReef_mhw_climatology.nc"
    #"TasmanSea -50 -35 140 160 output/TasmanSea_mhw_mask.nc output/TasmanSea_mhw_metrics.nc output/TasmanSea_mhw_climatology.nc"
    #"MediterraneanSea 30 50 0 30 output/MediterraneanSea_mhw_mask.nc output/MediterraneanSea_mhw_metrics.nc output/MediterraneanSea_mhw_climatology.nc"
    #"NortheastPacBlob 15 70 -180 -100 output/NortheastPacBlob_mhw_mask.nc output/NortheastPacBlob_mhw_metrics.nc output/NortheastPacBlob_mhw_climatology.nc"
    #"WestAust -40 -20 90 120 output/WestAust_mhw_mask.nc output/WestAust_mhw_metrics.nc output/WestAust_mhw_climatology.nc"
    #"NorthernHemisphere 0 90 -180 180 output/NorthernHemisphere_mhw_mask.nc output/NorthernHemisphere_mhw_metrics.nc output/NorthernHemisphere_mhw_climatology.nc"
    #"SouthernHemisphere -90 0 -180 180 output/SouthernHemisphere_mhw_mask.nc output/SouthernHemisphere_mhw_metrics.nc output/SouthernHemisphere_mhw_climatology.nc"
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
            read name lat_min lat_max lon_min lon_max mask_out metrics_out clim_out <<< $area
            echo "Processing area: $name, exp: $exp, levidx: $levidx"
            echo "Latitude range: $lat_min to $lat_max"
            echo "Longitude range: $lon_min to $lon_max"
            mask_out_mod="${mask_out%.nc}_${exp}_levidx${levidx}.nc"
            metrics_out_mod="${metrics_out%.nc}_${exp}_levidx${levidx}.nc"
            clim_out_mod="${clim_out%.nc}_${exp}_levidx${levidx}.nc"
            python main.py \
                --file_path="../../data/thetao_${exp}_d1993_2019_levidx${levidx}_r1x1.nc" \
                --var_name="thetao_${exp}" \
                --threshold_percentile=90 \
                --min_duration=5 \
                --max_gap=2 \
                --output_path="$mask_out_mod" \
                --verbose=True \
                --clim_start="1993-01-01" \
                --clim_end="2019-12-31" \
                --lat_min="$lat_min" \
                --lat_max="$lat_max" \
                --lon_min="$lon_min" \
                --lon_max="$lon_max" \
                --window="5" \
                --smoothing=True \
                --smoothwind="15" \
                --save_metrics=True \
                --metrics_output_file="$metrics_out_mod" \
                --save_climatology=True \
                --climatology_output_file="$clim_out_mod" \
                --plot_sample=False
            mv output/labels.nc output/labels_${name}_${exp}_levidx${levidx}.nc
            mv output/joined_labels.nc output/joined_labels_${name}_${exp}_levidx${levidx}.nc
        done
    done
done
