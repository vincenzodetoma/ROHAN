#%%
import os
import glob
import math
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import cartopy.crs as ccrs
import fire

def _extract_dataset_name(path, area, level):
    base = os.path.basename(path)
    prefix = base.replace(f"{area}_mhw_mask_", "").replace(f"_{level}.nc", "")
    return prefix


def _pick_coord_name(da, candidates):
    for name in candidates:
        if name in da.coords:
            return name
    raise ValueError(f"Coordinate non trovata tra {candidates}")


def main(
    input_dir="output",
    output_dir="output/gifs",
    fps=5,
    dpi=150,
    area="Global",
    level="levidx1",
    dataset_prefixes=("ESACCISST", "cglo", "glor", "oras"),
    ncols=2,
    chunks_time=1,
):
    if not os.path.isabs(input_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, input_dir)
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)

    pattern = os.path.join(input_dir, f"{area}_mhw_mask_*_{level}.nc")
    all_files = sorted(glob.glob(pattern))
    prefixes_lower = tuple(p.lower() for p in dataset_prefixes)
    mask_files = [
        f for f in all_files
        if _extract_dataset_name(f, area, level).lower().startswith(prefixes_lower)
    ]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(mask_files)

    event_areas = {
        "WestAust"          : slice("2010-11-01", "2011-07-01"),
        "NorthernHemisphere": slice(None,None),
        "SouthernHemisphere": slice(None,None),
        "NortheastPacBlob"  : slice("2014-05-01", "2016-08-01"),
        "MediterraneanSea"  : slice("2006-04-01", "2006-11-01"),
        "TasmanSea"         : slice("2015-07-01", "2016-11-01"),
        "GreatBarrierReef"  : slice("2015-11-15", "2016-07-15"),
        "SantaBarbara"      : slice("2015-06-15", "2016-01-15"),
        "NorthWestAtlantic" : slice("2011-11-01", "2012-08-01"),
        }

    if not mask_files:
        raise FileNotFoundError(
            f"Nessun file trovato per {area} con prefissi {dataset_prefixes} e livello {level}. Pattern: {pattern}"
        )

    area_shortname = area
    time_selection = event_areas.get(area_shortname, slice(None, None))
    print("time selection", time_selection)

    masks = []
    models = []
    for mask_file in mask_files:
        model_name = _extract_dataset_name(mask_file, area, level)
        ds = xr.open_dataset(mask_file, chunks={"time": chunks_time})
        varname = [v for v in ds.data_vars if "mask" in v][0]
        da = ds[varname]
        if "depth" in da.dims:
            da = da.isel(depth=0, drop=True)
        da = da.sel(time=time_selection)
        masks.append(da)
        models.append(model_name)

    aligned = xr.align(*masks, join="outer", fill_value=np.nan)
    mask_all = xr.concat(aligned, dim="model").assign_coords(model=models)

    if mask_all.sizes.get("time", 0) == 0:
        raise ValueError(
            f"Nessun dato nel range temporale {time_selection} per {area}."
        )

    lon_name = _pick_coord_name(mask_all, ["longitude", "lon"])
    lat_name = _pick_coord_name(mask_all, ["latitude", "lat"])

    times = mask_all.time.values
    vmax = float(mask_all.max(skipna=True).compute())
    if np.isnan(vmax):
        raise ValueError("Vmax non valido: tutti i valori sono NaN.")
    vmin = -vmax

    n_models = len(models)
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(6 * ncols, 4 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    ims = []
    for i, model in enumerate(models):
        ax = axes[i]
        ax.coastlines()
        data0 = mask_all.sel(model=model).isel(time=0)
        im = ax.imshow(
            data0.values,
            origin="lower",
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
            cmap="RdYlBu_r",
            extent=[
                float(data0[lon_name].min()),
                float(data0[lon_name].max()),
                float(data0[lat_name].min()),
                float(data0[lat_name].max()),
            ],
        )
        ax.set_title(model)
        ims.append(im)

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    cbar = fig.colorbar(ims[0], ax=axes[:n_models], orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label("Intensity Anomaly (°C)")
    suptitle = fig.suptitle(f"MHW intensity anomaly: {area_shortname}")

    def update(frame):
        for i, model in enumerate(models):
            data_t = mask_all.sel(model=model).isel(time=frame)
            ims[i].set_data(data_t.values)
        suptitle.set_text(
            f"MHW intensity anomaly: {area_shortname} | {str(np.datetime_as_string(times[frame], unit='D'))}"
        )
        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    video_path = os.path.join(output_dir, f"mhw_anomaly_{area_shortname}_{level}_ensemble.mp4")
    gif_path = video_path.replace(".mp4", ".gif")
    ani.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
    print(f"GIF Salvata: {gif_path}")
    plt.close(fig)

if __name__ == "__main__":
    fire.Fire(main)
