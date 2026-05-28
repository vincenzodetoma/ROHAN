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

def main(input_dir="output", output_dir="output/gifs", fps=5, dpi=150, area='GreatBarrierReef', dataset="ESACCISST", level="levidx1"):
    mask_files = sorted(glob.glob(os.path.join(input_dir, area+"_mhw_mask_"+dataset+"_"+level+".nc")))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(mask_files)


    event_areas = {
        "NorthernHemisphere": slice(None,None),
        "SouthernHemisphere": slice(None,None),
        "WestAust"          : slice("2010-11-01", "2011-07-01"),
        "NortheastPacBlob"  : slice("2014-05-01", "2016-08-01"),
        "MediterraneanSea"  : slice("2006-04-01", "2006-11-01"),
        "TasmanSea"         : slice("2015-07-01", "2016-11-01"),
        "GreatBarrierReef"  : slice("2015-11-15", "2016-07-15"),
        "SantaBarbara"      : slice("2015-06-15", "2016-01-15"),
        "NorthWestAtlantic" : slice("2011-11-01", "2012-08-01"),
        }

    for mask_file in mask_files:
        area_name = os.path.basename(mask_file).replace("_mhw_mask", "").replace(".nc", "")
        area_shortname = area_name.replace("_"+dataset+"_"+level, "")
        time_selection = event_areas[area_shortname]
        print("time selection", time_selection)
        ds = xr.open_dataset(mask_file).sel(time=time_selection)
        # Sostituisci 'mhw_mask' con il nome effettivo della variabile se diverso
        varname = [v for v in ds.data_vars if 'mask' in v][0]
        mask = ds[varname].squeeze() if "ESACCI" not in area_name else ds[varname]  # (time, lat, lon)
        print(mask)

        times = mask.time.values

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        plt.title(f"MHW intensity anomaly: \n {area_name}")

        vmin = float(mask.min())
        vmax = float(mask.max())
        im = ax.imshow(mask.isel(time=0), origin='lower', transform=ccrs.PlateCarree(),
                       vmin=-vmax, vmax=vmax, cmap='RdYlBu_r', extent=[
                           float(mask.longitude.min()), float(mask.longitude.max()),
                           float(mask.latitude.min()), float(mask.latitude.max())
                       ])
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, aspect=30)
        cbar.set_label("Intensity Anomaly (°C)")

        def update(frame):
            im.set_data(mask.isel(time=frame))
            if "ESACCI" in area_name:
                ax.set_title(f"{area_shortname} \n {dataset} \n {str(np.datetime_as_string(times[frame], unit='D'))}")
            else:
                ax.set_title(f"{area_shortname} \n {dataset} \n level: {ds.depth.values} \n {str(np.datetime_as_string(times[frame], unit='D'))}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(times), blit=True)
        video_path = os.path.join(output_dir, f"mhw_anomaly_{area_name}.mp4")
        gif_path = video_path.replace(".mp4", ".gif")
        ani.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
        print(f"GIF Salvata: {gif_path}")
        plt.close(fig)

if __name__ == "__main__":
    fire.Fire(main)
