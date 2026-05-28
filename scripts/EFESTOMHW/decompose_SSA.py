################################## MODULES IMPORT ###########################
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import detection
from dask.diagnostics import ProgressBar
import dask
import preprocess
import plotting
import gc
dask.config.set(**{'array.slicing.split_large_chunks': True})
ProgressBar().register()
################################ LOADING DATA ###############################
cwd = os.getcwd()
ds_whole = xr.open_dataset(cwd+'/../data/Global/ESACCI_daily_1x1_Global_1982-2021.nc', 
        chunks='auto')
ds_whole = ds_whole.rename({'latitude':'lat', 'longitude':'lon'})
ds_whole = ds_whole.sortby(ds_whole.lat)
AOIs=["TPac"]#, "TPac", "Global"]
for AOI in AOIs:
    if AOI=='TPac':
        lonmin, lonmax, latmin, latmax = 120, 230, -30, 30
        DetPeriod = ["1998-01-01", "2018-12-31"]
    elif AOI=='Med':
        lonmin, lonmax, latmin, latmax = -6, 36, 30, 46
        DetPeriod = ["2019-01-01", "2021-12-31"]
    else:
        AOI='Global'
        lonmin, lonmax, latmin, latmax = 0, 360, -90, 90
        DetPeriod = ["1982-01-01", "2021-12-31"]
    if AOI=='Global' or AOI=='TPac':
        ds_whole.coords['lon'] = (ds_whole.coords['lon']) % 360
        ds = ds_whole.sortby(ds_whole.lon)
    else:
        ds = ds_whole
    #print(ds)
    ds_cut = ds.sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))
    print("loading data...")
    sst_orig = ds_cut['analysed_sst'].chunk({'time':-1})
    sst_in, sstFast, sstSlow, periods = preprocess.xr_SSA_decompose(sst_orig, modes=int(len(sst_orig)/10))
    ds_ssa = xr.Dataset(
            data_vars=dict(
                inputSST = (["time", "lat", "lon" ], sst_in.data),
                fastSST=(["time", "lat", "lon"], sstFast.data),
                sstSlow = (["time", "lat", "lon"], sstSlow.data),
                period=(["lat", "lon"], periods.data),
            ),
            coords=dict(
                lon=(["lon"], sst_orig.lon.values),
                lat=(["lat"], sst_orig.lat.values),
                time=sst_orig.time.values,
            ),
            attrs=dict(description="result from SSA"),
        )
    ds_ssa.to_netcdf(AOI+'_SSA_results.nc')