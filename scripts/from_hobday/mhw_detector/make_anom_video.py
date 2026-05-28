#%%
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
from dask.diagnostics import ProgressBar
ProgressBar().register()

def main(input_dir="../output", output_dir="../output/gifs", fps=10, dpi=150, area='WestAust', dataset="cglo", level="levidx31"):
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
        "Global"            : slice("2014-05-01", "2016-08-01"),
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
        varname = "mhw_intensity"
        print(f"Variable name: {varname}")
        mask = ds[varname].squeeze() if "ESACCI" not in area_name else ds[varname]  # (time, lat, lon)
        mask = mask.where(ds['exceed_bool']!=0)
        print(mask)

        times = mask.time.values
        print(f"times: {times}")

        import make_video_single_point_profile as sp
        latpoint, lonpoint = sp.single_point(area=area, dataset=dataset, time_selection=time_selection)

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()

        # Add the fixed green point with black edge
        point_artist = ax.scatter(lonpoint[0], latpoint[0], color='green', edgecolor='black', s=40, zorder=10, transform=ccrs.PlateCarree())

        plt.title(f"MHW intensity anomaly: \n {area_name}")

        vmin = 0
        vmax = 3
        im = ax.imshow(mask.isel(time=0), origin='lower', transform=ccrs.PlateCarree(),
                       vmin=vmin, vmax=vmax, cmap='hot', extent=[
                           float(mask.longitude.min()), float(mask.longitude.max()),
                           float(mask.latitude.min()), float(mask.latitude.max())
                       ])
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, aspect=30)
        cbar.set_label("Intensity Anomaly (°C)")

        def update(frame):
            point_artist.set_offsets([lonpoint[frame], latpoint[frame]])
            im.set_data(mask.isel(time=frame))
            if "ESACCI" in area_name:
                ax.set_title(f"{area_shortname} \n {dataset} \n {str(np.datetime_as_string(times[frame], unit='D'))}")
            else:
                ax.set_title(f"{area_shortname} \n {dataset} \n level: {ds.depth.values} \n {str(np.datetime_as_string(times[frame], unit='D'))}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(times), blit=True)
        video_path = os.path.join(output_dir, f"mhw_anomaly_{area_name}.mp4")
        gif_path = video_path.replace(".mp4", ".gif")
        print(f"fps: {fps}")
        ani.save(gif_path, writer=PillowWriter(fps=int(float(fps))), dpi=dpi)
        print(f"GIF Salvata: {gif_path}")
        plt.close(fig)

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) >2:
        fire.Fire(main)
    else:
        main()