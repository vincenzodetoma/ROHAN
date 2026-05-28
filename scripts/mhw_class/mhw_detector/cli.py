import xarray as xr
import fire
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from .accessors import *
import matplotlib.pyplot as plt
import gc

def run_mhw_detection(
    file_path,
    var_name="sst",
    threshold_percentile=90,
    min_duration=5,
    max_gap=2,
    output_path="mhw_mask.nc",
    verbose=True,
    clim_start=None,
    clim_end=None,
    lat_min=None,
    lat_max=None,
    lon_min=None,
    lon_max=None,
    window=5,
    smoothing=True,
    smoothwind=15,
    save_climatology=False,
    climatology_output_file="mhw_climatology.nc",
    save_metrics=False,
    metrics_output_file="mhw_metrics.nc",
    plot_sample=False
):
    if verbose:
        ProgressBar().register()
        tqdm.write(f"\U0001F680 Loading dataset from: {file_path}")

    if ~((lon_min is not None and lon_max is not None) and (lat_min is not None and lat_max is not None)):
        save_metrics = False
        save_climatology = True
        print("No region specified, using full dataset", "\n",
              "Writing out only mask, climatology and threshold anyway.", "\n",
              f"save_metrics={save_metrics}, save_climatology={save_climatology}"
              )
        
    ds = xr.open_dataset(file_path, chunks={'time':-1, 'latitude': 'auto', 'longitude':'auto'})
    if verbose:
        tqdm.write(f"\U0001F4CD Regionalizing to {lat_min} {lat_max} {lon_min} {lon_max} ")

    if (lon_min is not None and abs(lon_min) > 360) or (lon_max is not None and abs(lon_max) > 360):
        raise ValueError("Longitude values must be within -360 to 360.")

    if (lon_min is not None and lon_min > 180) or (lon_max is not None and lon_max > 180):
        if 'longitude' in ds.coords:
            ds = ds.assign_coords(longitude=((ds.longitude + 360) % 360))
            ds = ds.sortby('longitude')
        else:
            raise ValueError("Longitude coordinate not found in dataset.")
    
    sst = ds[var_name].sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    if verbose and plot_sample:
        if len(sst.latitude)>1 and len(sst.longitude)>1:
            sst.isel(time=0).plot()
        else:
            sst.plot()
        plt.show()

    if verbose:
        tqdm.write(f"\U0001F4CB Running MHW detection on variable: {var_name}")

    results = sst.mhw.detect(
        threshold_percentile=threshold_percentile,
        min_duration=min_duration,
        max_gap=max_gap,
        clim_start=clim_start,
        clim_end=clim_end,
        verbose=verbose,
        return_metrics=save_metrics,
        window=window,
        smoothing=smoothing,
        smoothwind=smoothwind,
        return_climatology=save_climatology
    )
    mhw_mask = results[0]

    if verbose:
        tqdm.write(f"\U0001F4BE Saving MHW mask to: {output_path}")

    mhw_mask.name = "mhw_mask"
    mhw_mask = mhw_mask
    mhw_mask.to_netcdf(output_path)
    del mhw_mask
    gc.collect()
    if save_metrics:
        if verbose:
            tqdm.write(f"\U0001F4BE Saving Gruber metrics to: {metrics_output_file}")
        metrics = results[1]
        metrics.to_netcdf(metrics_output_file)
        del metrics
        gc.collect()
        if save_climatology:
            if verbose:
                tqdm.write(f"\U0001F4BE Saving climatology to: {climatology_output_file}")
            threshold= results[2]
            threshold.name = "mhw_threshold"
            climatology = results[3]
            climatology.name = "mhw_climatology"
            ds_clim = xr.merge([climatology, threshold])
            ds_clim.to_netcdf(climatology_output_file)
            del climatology, threshold, ds_clim
            gc.collect()
    else:
        if verbose:
            tqdm.write("\U0001F4A1 No metrics calculated, saving only MHW mask climatology and threshold.")
        if save_climatology:
            if verbose:
                tqdm.write("\U0001F4A1 Saving climatology to: {climatology_output_file}")
            threshold = results[1]
            threshold.name = "mhw_threshold"
            climatology = results[2]
            climatology.name = "mhw_climatology"
            ds_clim = xr.merge([climatology, threshold])
            ds_clim.to_netcdf(climatology_output_file)
            del climatology, threshold, ds_clim
            gc.collect()
    if verbose:
        tqdm.write("\u2705 Done!")

if __name__ == "__main__":
    fire.Fire(run_mhw_detection)

