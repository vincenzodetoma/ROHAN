#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from tqdm import tqdm
from dask.diagnostics import ProgressBar
ProgressBar().register()

def _build_file_list(initfilepath, start_date, end_date):
    if not start_date or not end_date:
        return sorted([os.path.join(initfilepath, f) for f in os.listdir(initfilepath) if f.startswith('grepv2_daily_') and f.endswith('.nc')])

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    if end < start:
        raise ValueError('end_date must be on or after start_date')

    files = []
    day = start
    while day <= end:
        fname = f"grepv2_daily_{day.strftime('%Y%m%d')}.nc"
        fpath = os.path.join(initfilepath, fname)
        if os.path.exists(fpath):
            files.append(fpath)
        day += timedelta(days=1)
    return files

def main(model=['cglo', 'glor', 'oras'], start_date='1993-01-01', end_date='2019-12-31'):
    initfilepath = '/store/data_store/Ocean_Rean/GLOBAL_REANALYSIS_PHY_001_031/global-reanalysis-phy-001-031-grepv2-daily/global'
    print("initfilepath", initfilepath)
    varnames = [f"mlotst_{m}" for m in model]
    file_list = _build_file_list(initfilepath, start_date, end_date)
    print("file count", len(file_list))

    def _preprocess(ds):
        # Drop variables early to reduce metadata and indexing overhead.
        MLDs = [f"mlotst_{m}" for m in model]
        varnames = [v for v in MLDs if v in ds.variables]
        keep = [v for v in varnames if v in varnames]
        return ds[keep]

    datasets = []
    for fpath in tqdm(file_list, desc='open files'):
        ds_part = xr.open_dataset(fpath, chunks='auto')
        ds_part = _preprocess(ds_part)
        datasets.append(ds_part)
    ds = xr.combine_by_coords(datasets, compat='equals', combine_attrs='drop_conflicts')
    ds_mld = ds[varnames].to_array(dim='model').assign_coords({'model': model})
    ds_mld = ds_mld.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds_mld.coords['lon'] = (ds_mld.coords['lon'] + 180) % 360 - 180
    ds_mld = ds_mld.sortby(ds_mld.lon)
    ds_mld_ensmean = ds_mld.mean(dim='model').expand_dims({'model': ['EnsMean']})
    ds_mld = xr.concat([ds_mld, ds_mld_ensmean], dim='model')
    ds_mld = ds_mld
    print("ds_mld", ds_mld)
    return ds_mld

def faceted_plot(ds_seas):
    ds_ref = ds_seas.sel(model='EnsMean')
    ds_bias = ds_seas.sel(model=ds_seas.model != 'EnsMean') - ds_ref

    seasons = list(ds_seas.season.values)
    models = list(ds_seas.model.values)
    bias_cmap = 'coolwarm'
    abs_cmap = 'Spectral_r'

    bias_max = 15
    abs_min = 0
    abs_max = 200

    nrows = len(models)
    ncols = len(seasons)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 3 * nrows / 2),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True,
    )
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, np.newaxis]

    bias_mappable = None
    abs_mappable = None

    for r, model in enumerate(models):
        for c, season in enumerate(seasons):
            ax = axes[r, c]
            if model == 'EnsMean':
                data = ds_ref.sel(season=season)
                mappable = data.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap=abs_cmap,
                    vmin=abs_min,
                    vmax=abs_max,
                    add_colorbar=False,
                )
                abs_mappable = mappable
            else:
                data = ds_bias.sel(model=model, season=season)
                mappable = data.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap=bias_cmap,
                    vmin=-bias_max,
                    vmax=bias_max,
                    add_colorbar=False,
                )
                bias_mappable = mappable

            ax.coastlines('50m')
            ax.set_title(f"{model}, {season}")
            gl = ax.gridlines(draw_labels=False)
            if r == nrows - 1:
                gl.xlabel_style = {'size': 8}
                gl.bottom_labels = True
            if c == 0:
                gl.ylabel_style = {'size': 8}
                gl.left_labels = True

    if bias_mappable is not None:
        fig.colorbar(bias_mappable, ax=axes[:-1, :], orientation='vertical', pad=0.02, label='MLD Bias [m]')
    if abs_mappable is not None:
        fig.colorbar(abs_mappable, ax=axes[-1, :], orientation='vertical', pad=0.02, label='MLD [m]')
    plt.show()

if __name__=='__main__': 
    if os.path.exists('../results/mld_seasonal.nc'):
        print("Loading precomputed seasonal means from results/mld_seasonal.nc")
        ds_seas = xr.open_dataset('../results/mld_seasonal.nc')
    else:
        print("Computing seasonal means from raw data files...")
        ds = main()
        ds_seas = ds.groupby('time.season').mean('time')
        ds_seas.to_netcdf('../results/mld_seasonal.nc')
    faceted_plot(ds_seas['__xarray_dataarray_variable__'])
