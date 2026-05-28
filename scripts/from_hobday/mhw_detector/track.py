#%%
import sys
sys.path.append('../')
from ocetrac.ocetrac.tracker import Tracker
import xarray as xr
import numpy as np
import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import cartopy.crs as ccrs
import fire
import datetime
import sys
import pandas as pd
import numpy as np
import xarray as xr

import cmocean
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

import ocetrac

from ocetrac.ocetrac.utils import cesm2_lens_utils
from ocetrac.ocetrac.utils import cesm_anomalies
from ocetrac.ocetrac.measures.shape_measures import ShapeMeasures
from ocetrac.ocetrac.measures.motion_measures import MotionMeasures
from ocetrac.ocetrac.measures.temporal_measures import get_initial_detection_time, get_duration
from ocetrac.ocetrac.measures.intensity_measures import calculate_intensity_metrics
from ocetrac.ocetrac.measures.plotting import plot_displacement
from ocetrac.ocetrac.measures.utils import lons_to_360, get_object_masks, run_shape_measures, run_motion_measures, run_temporal_measures, run_intensity_measures, process_objects_and_calculate_measures

def main(input_dir="../output", output_dir="../output/gifs", fps=10, dpi=150, area='Global', dataset="cglo", level="levidx19"):
    mask_files = sorted(glob.glob(os.path.join(input_dir, area+"_mhw_climatology_"+dataset+"_"+level+".nc")))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("mask files", mask_files)


    event_areas = {
        "WestAust"          : slice("2010-11-01", "2011-07-01"),
        "NortheastPacBlob"  : slice("2014-05-01", "2016-08-01"),
        "MediterraneanSea"  : slice("2006-04-01", "2006-11-01"),
        "TasmanSea"         : slice("2015-07-01", "2016-11-01"),
        "GreatBarrierReef"  : slice("2015-11-15", "2016-07-15"),
        "SantaBarbara"      : slice("2015-06-15", "2016-01-15"),
        "NorthWestAtlantic" : slice("2011-11-01", "2012-08-01"),
        "NorthernHemisphere": slice(None,None),
        "SouthernHemisphere": slice(None,None),
        "Global"            : slice(None, None),
        }

    for mask_file in mask_files:
        area_name = os.path.basename(mask_file).replace("_mhw_climatology", "").replace(".nc", "")
        print(f"Area Name {area_name}")
        area_shortname = area_name.replace("_"+dataset+"_"+level, "")
        print(f"Area shortName {area_shortname}")
        time_selection = event_areas[area_shortname]
        print("time selection", time_selection.start, time_selection.stop)
        time_coord_start = time_selection.start if time_selection.start is not None else "1993-01-01"
        time_coord_end = time_selection.stop if time_selection.stop is not None else "2019-12-31"
        print(f"start {time_coord_start}, end {time_coord_end}")
        ds = xr.open_dataset(mask_file).assign_coords({'time': pd.date_range("1993-01-01", "2019-12-31", freq='1D')}).sel(time=time_selection).assign_coords({'time': pd.date_range(time_coord_start, time_coord_end, freq='D')})
        # Sostituisci 'mhw_mask' con il nome effettivo della variabile se diverso
        #varname = [v for v in ds.data_vars if 'mask' in v][0]
        ds = lons_to_360(ds, coord='longitude')
        ds = ds.rename({'longitude':'lon', 'latitude': 'lat'})
        varname = "mhw_intensity"
        print(f"Variable name: {varname}")
        mhw_intensity = ds[varname].squeeze() if "ESACCI" not in area_name else ds[varname]  # (time, lat, lon)
        mask_bool = ds['exceed_bool']
        mask_region = xr.ones_like(mhw_intensity)
        obj_Tracker = Tracker(da=mhw_intensity, mask=mask_region, radius=3, min_size_quartile=0.75, timedim='time', xdim='lon', ydim='lat', positive=True)
        blobs = obj_Tracker.track()
        blobs = blobs
        mo = obj_Tracker._morphological_operations()
        
        # Defining our region (Pacific Ocean)
        upper_lat = 90
        lower_lat = -90
        left_lon = 0
        right_lon = 360

        anomalies_NP = mhw_intensity.sel(lon=slice(left_lon, right_lon),
                                         lat=slice(lower_lat,upper_lat))
        mhw_objs_NP = blobs.sel(lon=slice(left_lon, right_lon),
                                lat=slice(lower_lat,upper_lat))
        print(np.unique(mhw_objs_NP.data)[:-1])
        event_binary, event_intensity = get_object_masks(blobs, mhw_intensity, object_id=150)
        
        fig, axes = plt.subplots(5, 1, figsize=(4, 8), 
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        axes = axes.flatten()
        print(event_intensity)

        for i, ax in enumerate(axes[:11]):  # Only plot first 5
            contour = event_intensity.sel(lat=slice(-65, 65))[i,:,:].plot.contourf(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmocean.cm.balance,
                levels=17,
                vmin=-4,
                vmax=4,
                add_colorbar=False,
                extend='neither'
            )

            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

            time_str = str(event_intensity.time[i].values)[:10]
            ax.set_title(time_str, fontsize=10, pad=4)

        cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Intensity', fontsize=10)

        plt.tight_layout()
        plt.show()
        
if __name__=='__main__':
    main()
        