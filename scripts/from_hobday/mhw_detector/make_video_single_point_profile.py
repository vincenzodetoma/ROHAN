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
import dask
from dask.diagnostics import ProgressBar
ProgressBar().register()

def single_point(input_dir="../output", output_dir="../output/gifs", fps=10, dpi=150, area='Global', dataset="cglo", level="levidx*", time_selection=slice(None, None)):
    mask_files = sorted(glob.glob(os.path.join(input_dir, area+"_mhw_climatology_"+dataset+"_"+level+".nc")))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("mask files", mask_files)
        
    time_selection = time_selection
    time_coord_start = time_selection.start if time_selection.start is not None else "1993-01-01"
    time_coord_end = time_selection.stop if time_selection.stop is not None else "2019-12-31"
    print(f"start {time_coord_start}, end {time_coord_end}")
    ds = xr.open_mfdataset(mask_files, concat_dim='depth', combine='nested').assign_coords({'time': pd.date_range("1993-01-01", "2019-12-31", freq='1D')}).sel(time=time_selection).assign_coords({'time': pd.date_range(time_coord_start, time_coord_end, freq='D')})
    ds = ds.sortby(ds.depth).compute()
    # Sostituisci 'mhw_mask' con il nome effettivo della variabile se diverso
    #varname = [v for v in ds.data_vars if 'mask' in v][0]
    varname = "mhw_intensity"
    print(f"Variable name: {varname}")
    mask = ds[varname]  # (time, lat, lon)
    mask = mask#.where(ds['exceed_bool']!=0)
    print(mask)
    times = mask.time.values
    print(f"times: {times}")
    fig, ax = plt.subplots(figsize=(6, 8))
    vmin = -10
    vmax = 10
    
    maximum = mask.max('depth').max('time').argmax(dim=['latitude', 'longitude'])
    print("maxidx", maximum)
    
    LatPoints, LonPoints = [], []
    
    latpoint, lonpoint = mask.latitude[maximum['latitude'].values], mask.longitude[maximum['longitude'].values]
    LatPoints.append(latpoint)
    LonPoints.append(lonpoint)
    print("latpoints:", latpoint, lonpoint)

    # Extract the vertical profile time series
    profile_data = mask.sel(latitude=latpoint, longitude=lonpoint, method='nearest')  # (time, depth)
    print(f"profile data {profile_data}")
    depths = profile_data.depth.values
    print(f"depths {depths}")

    # Initialize the line plot
    for tt in range(1, len(profile_data.time)):
        lines, = ax.plot(profile_data.isel(time=tt), depths, color='grey', alpha=0.5)
        
    line, = ax.plot(profile_data.isel(time=0), depths, marker='o', color='green', markeredgecolor='k', linewidth=5)
    ax.axvline(0, color='k')
    ax.set_ylim(depths.max() + 5, depths.min() - 5)  # Invert y-axis: deeper at bottom
    ax.set_xlim(vmin, vmax)
    ax.grid(ls='--', color='grey')
    ax.set_xlabel("MHW Intensity")
    ax.set_ylabel("Depth (m)")
    plt.show()

    title = ax.set_title(f"MHW intensity anomaly profile: \n")  # Will update in `update`

    def update(frame):
        #latpoint, lonpoint = mask.latitude[maximum['latitude'].isel(time=frame).values], mask.longitude[maximum['longitude'].isel(time=frame).values]
        print("latpoints:", latpoint, lonpoint) 
        LatPoints.append(latpoint)
        LonPoints.append(lonpoint)

        # Extract the vertical profile time series  
        profile_data = mask.sel(latitude=latpoint, longitude=lonpoint, method='nearest')  # (time,  depth)
        line.set_xdata(profile_data.isel(time=frame).values)
        title.set_text(
            f"MHW intensity anomaly profile: \n"
            f"lat: {latpoint.values}, lon: {lonpoint.values} | dataset: {dataset} \n"
            f"Time: {np.datetime_as_string(times[frame], unit='D')}"
        )
        return [line, title]

    ani = animation.FuncAnimation(fig, update, frames=len(times), blit=True)
    video_path = os.path.join(output_dir, f"{area}_mhw_anomaly_profile_{dataset}.mp4")
    gif_path = video_path.replace(".mp4", ".gif")
    print(f"fps: {fps}")
    ani.save(gif_path, writer=PillowWriter(fps=int(float(fps))), dpi=dpi)
    print(f"GIF Salvata: {gif_path}")
    plt.close(fig)
    return LatPoints, LonPoints
        
if __name__=='__main__':
    llat, llon = single_point()