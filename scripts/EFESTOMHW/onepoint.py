#%%
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
AOIs=["Med", "TPac"]#, "Global"]
for AOI in AOIs:
    if AOI=='TPac':
        lonmin, lonmax, latmin, latmax = 120, 230, -30, 30
        DetPeriod = ["1998-01-01", "2018-12-31"]
        latpoint, lonpoint = 0, 220
    elif AOI=='Med':
        lonmin, lonmax, latmin, latmax = -6, 36, 30, 46
        DetPeriod = ["2019-01-01", "2021-12-31"]
        lonpoint, latpoint = 7.5, 42.5
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
    sst_orig = ds_cut['analysed_sst'].chunk({'time':-1}).sel(lat=latpoint, lon=lonpoint, method='nearest').compute()
    fs = (2*np.pi)/(24*60*60) # Sampling Frequency in Hertz
    lowcut = (2*np.pi)/(400*365*24*60*60) #ENSO lower limit freq
    highcut = (2*np.pi)/(1*365*24*60*60) #ENSO upper limit freq
    # Filter all frequencies inside lowcut, highcut. 
    FiltSST_np = preprocess.butter_bandstop_filter(sst_orig, lowcut, highcut, fs, order=6)
    FiltSST = xr.DataArray(FiltSST_np, coords=sst_orig.coords).rename('butterworth_gt_1y')
    _, fastSST_np, slowSST_np, _ = preprocess.single_pixel_ssa(sst_orig.values, n_components=365)
    fastSST = xr.DataArray(fastSST_np, coords=sst_orig.coords)
    slowSST = xr.DataArray(slowSST_np, coords=sst_orig.coords)
    fastSST = fastSST.rename('FastSST_after_SSA_gt_1y')
    slowSST = slowSST.rename('SlowSST_after_SSA_gt_1y')
    sst_orig.to_netcdf(AOI+'_singlePoint_orig_latlon_'+str(latpoint)+str(lonpoint)+'.nc')
    FiltSST.to_netcdf(AOI+'_singlePoint_butter_latlon_'+str(latpoint)+str(lonpoint)+'.nc')
    fastSST.to_netcdf(AOI+'_singlePoint_fastSSA_latlon_'+str(latpoint)+str(lonpoint)+'.nc')
    slowSST.to_netcdf(AOI+'_singlePoint_slowSSA_latlon_'+str(latpoint)+str(lonpoint)+'.nc')
    
# %%
