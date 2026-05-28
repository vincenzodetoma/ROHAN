#!/bin/bash

areas=(
    "WestAust"
    "NorthWestAtlantic"
    "SantaBarbara"
    "GreatBarrierReef"
    "TasmanSea"
    "MediterraneanSea"
    "NortheastPacBlob"
    #"NorthernHemisphere"
    #"SouthernHemisphere"
)

for area in "${areas[@]}"; do
    echo "Generating violin plots for area: $area"
    PYTHONUNBUFFERED=1 python gruber_metrics.py --area="$area" --plot=True --use_tiles=False || exit -136
done
