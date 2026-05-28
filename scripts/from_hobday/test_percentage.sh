
exps=(
    #"ESACCISST"
    #"cglo"
    "ensemble" 
    #"foam" 
    #"glor" 
    #"oras"
)
vmins=(
    #5
    #-15
    -15
    #-15
    #-15
    #-15
)
vmaxs=(
    #65
    #15
    15
    #15
    #15
    #15
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
    #"NorthWestAtlantic 35 50 -80 -50 output/NorthWestAtlantic_mhw_Atlas.nc output/NorthWestAtlantic_mhw_numEvents.nc output/NorthWestAtlantic_mhw_climatology.nc"
    #"SantaBarbara 30 37 -125 -115 output/SantaBarbara_mhw_Atlas.nc output/SantaBarbara_mhw_numEvents.nc output/SantaBarbara_mhw_climatology.nc"
    #"GreatBarrierReef -25 -5 135 155 output/GreatBarrierReef_mhw_Atlas.nc output/GreatBarrierReef_mhw_numEvents.nc output/GreatBarrierReef_mhw_climatology.nc"
    #"TasmanSea -50 -35 140 160 output/TasmanSea_mhw_Atlas.nc output/TasmanSea_mhw_numEvents.nc output/TasmanSea_mhw_climatology.nc"
    #"MediterraneanSea 30 50 0 30 output/MediterraneanSea_mhw_Atlas.nc output/MediterraneanSea_mhw_numEvents.nc output/MediterraneanSea_mhw_climatology.nc"
    #"NortheastPacBlob 15 70 -180 -100 output/NortheastPacBlob_mhw_Atlas.nc output/NortheastPacBlob_mhw_numEvents.nc output/NortheastPacBlob_mhw_climatology.nc"
    #"WestAust -40 -20 90 120 output/WestAust_mhw_Atlas.nc output/WestAust_mhw_numEvents.nc output/WestAust_mhw_climatology.nc"
    #"NorthernHemisphere 0 90 -180 180 output/NorthernHemisphere_mhw_Atlas.nc output/NorthernHemisphere_mhw_numEvents.nc output/NorthernHemisphere_mhw_climatology.nc"
    #"SouthernHemisphere -90 0 -180 180 output/SouthernHemisphere_mhw_Atlas.nc output/SouthernHemisphere_mhw_numEvents.nc output/SouthernHemisphere_mhw_climatology.nc"
    "Global None None None None output/Global_mhw_Atlas.nc output/Global_mhw_numEvents.nc output/Global_mhw_climatology.nc"
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
            PYTHONUNBUFFERED=1 python mhw_detector/percentage.py \
                --filename="${mask_out_mod}" \
                --varname="duration" \
                --filename_sst="${clim_out_mod}" \
                --varname_sst="mhw_intensity" \
                --expname="${exp}" \
                --levidx="${levidx}" \
		        --vmin="${vmins[$exp_idx]}" \
		        --vmax="${vmaxs[$exp_idx]}" \
		        --projection="miller"\
                --metric='mean' || exit -12
        done
    done
done
