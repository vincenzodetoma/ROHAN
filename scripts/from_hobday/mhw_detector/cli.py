#%%
import time
import xarray as xr
import fire
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import sys
sys.path.append('../')  # Adjust the path to your project structure
import matplotlib.pyplot as plt
import pandas as pd
from marineHeatWaves.marineHeatWaves import detect
try:
    # Import relativo (funziona quando esegui come modulo/package)
    from .clustering import *
    from .plotting import *
    from .utils import *
except ImportError:
    # Import assoluto (funziona in notebook o script standalone)
    from clustering import *
    from plotting import *
    from utils import *

def run_mhw_detection(
    file_path="../../../data/thetao_ESACCISST_d1993_2019_levidx1_r1x1.nc",
    var_name="thetao_ESACCISST",
    threshold_percentile=90,
    min_duration=5,
    max_gap=2,
    output_path="mhw_Atlas.nc",
    verbose=True,
    clim_start=1993,
    clim_end=2019,
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
    metrics_output_file="mhw_nEvents.nc",
    plot_sample=True,
    coldSpelldetect=False,
    clustering=False,
    clusterator="HDBSCAN",
    method='both',
    tol_days=5,
    cont_pixels=4,
    eps_space=4,
    eps_time=5,
    min_samples=2,
    returns = False,
):
    if verbose:
        ProgressBar().register()
        tqdm.write(f"\U0001F680 Loading dataset from: {file_path}")

    lat_chunks= 'auto'
    lon_chunks= 'auto'

    if ~((lon_min is not None and lon_max is not None) and (lat_min is not None and lat_max is not None)):
        save_metrics = False
        save_climatology = True
        print("No region specified, using full dataset", "\n",
              "Writing out only mask, climatology and threshold anyway.", "\n",
              f"save_metrics={save_metrics}, save_climatology={save_climatology}"
              )
        
    ds = xr.open_dataset(file_path, 
                         chunks={
                             'time':-1, 
                             'latitude': lat_chunks, 
                             'longitude':lon_chunks
                             }
                         )
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
    
    mask = xr.open_dataset('/store/detomavi/cmems_gloran_lot8/data/thetao_ESACCISST_d1993_2019_levidx1_r1x1.nc', 
                           chunks={
                               'time':-1, 
                               'latitude':  lat_chunks, 
                               'longitude': lon_chunks
                               }, 
                           )['mask'].sel(
        latitude=slice(lat_min, lat_max), 
        longitude=slice(lon_min, lon_max)
    ).assign_coords({'time': ds.time})
    
    
    sst = ds[var_name].sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)).where(mask==1)
    if "depth" in sst.dims:
        print("Removing depth dimension from sst variable")
        sst = sst.squeeze('depth')
    if verbose and plot_sample:
        if len(sst.latitude)>1 and len(sst.longitude)>1:
            plot_map(sst.isel(time=0), title=sst.isel(time=0).time.values, cmap='jet', cbar_label='SST [K]', vmin=sst.isel(time=0).min().values, vmax=sst.isel(time=0).max().values)
        else:
            sst.plot()
        plt.show()

    if verbose:
        tqdm.write(f"\U0001F4CB Running MHW detection on variable: {var_name}")

    t = pd.to_datetime(sst.time).to_series().apply(lambda x: x.toordinal()).values
    print(f"Time range: {t.min()} to {t.max()}")
    
    results = xr.apply_ufunc(
        detect,
        t, 
        sst,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[object, object],
        keep_attrs=True,
        kwargs={
            'climatologyPeriod': [clim_start,clim_end], 
            'pctile': threshold_percentile, 
            'windowHalfWidth': window, 
            'smoothPercentile': smoothing, 
            'smoothPercentileWidth': smoothwind, 
            'minDuration': min_duration, 
            'joinAcrossGaps': True, 
            'maxGap': max_gap, 
            'maxPadLength': False, 
            'coldSpells': coldSpelldetect, 
            'alternateClimatology': False, 
            'Ly': False,
        }
    )
    
    print("MHW detection function applied, computing mhw...")
    mhw = results[0].compute()
    print("Computing climatology...")
    clim = results[1].compute()
    print("extracting dictionaries keys from results...")
    first_dict = mhw.values.flat[0]
    first_dict_keys = list(first_dict.keys())
    first_dict_keys.remove('n_events')
    second_dict = clim.values.flat[0]
    second_dict_keys = list(second_dict.keys())
    
    print('extracting n_events from mhw...')
    n = extract_field(mhw, 'n_events').rename('n_events')
    print('writing n_events to file...')
    n.to_netcdf(metrics_output_file) # change me when you integrate this into main!
    
    print("Creating mhw xarray dataset...")
    ds_mhw = xr.Dataset(
        {
            m:(['event', 'latitude', 'longitude'], extract_event_field(mhw, m).values) for m in first_dict_keys
        }
    ).assign_coords({'latitude': sst.latitude, 'longitude': sst.longitude})
    print('writing mhw dataset to file...')
    if "depth" in ds[var_name].dims:
        ds_mhw.attrs['depth'] = ds[var_name].depth.values
    ds_mhw.to_netcdf(output_path)
    
    print("Creating climatology xarray dataset...")
    ds_clim = xr.Dataset(
        {
            c:(['time', 'latitude', 'longitude'], extract_field_with_time(clim, c, time_coord=sst.time.values).values) for c in second_dict_keys
        }
    ).assign_coords({'latitude': sst.latitude, 'longitude': sst.longitude})
    print('writing climatology dataset to file...')
    if "depth" in ds[var_name].dims:
        ds_clim.attrs['depth'] = ds[var_name].depth.values
    ds_clim.to_netcdf(climatology_output_file)
    
    print("MHW detection completed.")
    
    if clustering:
        print(f"clustering events with method {method}")
        results_clustering = clustering(sst, ds_mhw, tol_days=tol_days, cont_pixels=cont_pixels, eps_space=eps_space, eps_time=eps_time, min_samples=min_samples, clusterator=clusterator)
        if method=='both':
            ds_mhw_grouped = results_clustering['by-hand']
            ds_mhw_grouped_clusters = results_clustering['clustering']
            ds_mhw_grouped_clusters_wtime = results_clustering['clustering_wtime']
        else:
            results_clustering = results_clustering[method]
    
    if returns:
        if clustering:
            if method=='both':
                return t, sst, results, ds_mhw, ds_clim, ds_mhw_grouped, ds_mhw_grouped_clusters, ds_mhw_grouped_clusters_wtime
            else:
                return t, sst, results, ds_mhw, ds_clim, results_clustering
        return t, sst, results, ds_mhw, ds_clim

if __name__ == "__main__":
    tstart = time.time()
    print(f"Start time: {tstart} seconds")
    if len(sys.argv) >2:
        fire.Fire(run_mhw_detection)
    else:
        t, sst, results, ds_mhw, ds_clim = run_mhw_detection(lat_min=-40, lat_max=-20, lon_min=90, lon_max=120, returns=True)
        print("Computing mhw...")
        print("plotting events which have their starting date in 2011...")
        clusterator="HDBSCAN"
        t_days=5
        contpix=10
        epsspace=10
        epstime=5
        min_samples=15
        results_clustering = clustering(sst=sst, ds_mhw=ds_mhw, intensity=ds_clim['mhw_intensity'].values, tol_days=t_days, cont_pixels=contpix, eps_space=epsspace, eps_time=epstime, min_samples=min_samples, clusterator=clusterator)
        ds_mhw_grouped = results_clustering['by-hand']
        ds_mhw_grouped_clusters = results_clustering[f'{clusterator}_clustering']
        ds_mhw_grouped_clusters_wtime = results_clustering[f'{clusterator}_clustering_wtime']
        make_plots(ds_mhw_grouped, 2011, cwrap=2, suptitle=f'by-hand, {contpix}, {t_days}', vmin=0, vmax=2)
        print("Clustering events in space and time...")
        make_plots(ds_mhw_grouped_clusters, 2011, cwrap=2, suptitle=f"{clusterator}, {epsspace}, {epstime}, {min_samples}", vmin=0, vmax=2)
        plot_events_active_in_year(ds_mhw_grouped_clusters_wtime, sst, 2011, 4, suptitle=f"{clusterator}, {epsspace}, {epstime}, {min_samples}", vmin=-1, vmax=1)
        clusterator="DBSCAN"
        results_clustering = clustering(sst=sst, ds_mhw=ds_mhw, intensity=ds_clim['mhw_intensity'].values, tol_days=t_days, cont_pixels=contpix, eps_space=epsspace, eps_time=epstime, min_samples=min_samples, clusterator=clusterator)
        ds_mhw_grouped_clusters = results_clustering[f'{clusterator}_clustering']
        ds_mhw_grouped_clusters_wtime = results_clustering[f"{clusterator}_clustering_wtime"]
        make_plots(ds_mhw_grouped_clusters, 2011, cwrap=2, suptitle=f"{clusterator}, {epsspace}, {epstime}, {min_samples}", vmin=0, vmax=2)
        print("Plotting done.")
        plot_events_active_in_year(ds_mhw_grouped_clusters_wtime, sst, 2011, 4, suptitle=f"{clusterator}, {epsspace}, {epstime}, {min_samples}", vmin=-1, vmax=1)
        print("Done!")
    tend = time.time()
    elapsed = tend - tstart
    print(f"Elapsed time: {elapsed} seconds")
