#%%
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
import sys
dask.config.set(**{'array.slicing.split_large_chunks': True})
ProgressBar().register()
################################ LOADING DATA ###############################
cwd = os.getcwd()
expname=str(sys.argv[1])
levidx=str(sys.argv[2])
ds_whole = xr.open_mfdataset(cwd+'/../../data/thetao_'+expname+'_daily_*levidx'+levidx+'_r1x1.nc', 
        chunks='auto').chunk({"time": -1, "latitude": "auto", "longitude": "auto"})
ds_whole = ds_whole.rename({'latitude':'lat', 'longitude':'lon'})
ds_whole = ds_whole.sortby(ds_whole.lat)
AOIs=["Global"]#["Med"]#, "Med", "TPac"]
for AOI in AOIs:
    if AOI=='TPac':
        lonmin, lonmax, latmin, latmax = 120, 230, -30, 30
        DetPeriod = ["1982-01-01", "2021-12-31"]
    elif AOI=='Med':
        lonmin, lonmax, latmin, latmax = -6, 36, 30, 46
        DetPeriod = ["1982-01-01", "2021-12-31"]
    else:
        AOI='Global'
        lonmin, lonmax, latmin, latmax = 0, 360, -90, 90
        DetPeriod = ["1993-01-01", "2019-12-31"]
    if AOI=='Global' or AOI=='TPac':
        ds_whole.coords['lon'] = (ds_whole.coords['lon']) % 360
        ds = ds_whole.sortby(ds_whole.lon)
    else:
        ds = ds_whole
    #print(ds)
    ds_cut = ds.sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax)).squeeze('depth')
    print("loading data...")
    sst_orig = ds_cut['thetao_'+expname]#.chunk({'time':-1})
    ############################## Set detection Parameters #####################
    ClimPeriod = ["1993-01-01", "2019-12-31"]
    PcTile, Win, Smooth, SmoothWin, MinDur = 0.9, 5, True, 15, 5
    # Filter all frequencies inside lowcut, highcut. 
    detSST = preprocess.detrend_linear(sst_orig, dim='time')
    detSST = detSST.rename({'detrended_sst'}).compute()#.chunk({'time':-1})
    print(detSST)
    detSST.to_netcdf(cwd+'/../../results/thetao_'+expname+'_daily__1993_2019_levidx'+levidx+'_r1x1_detr.nc')
    del detSST
    detSST = xr.open_dataset(cwd+'/../../results/thetao_'+expname+'_daily__1993_2019_levidx'+levidx+'_r1x1_detr.nc')['detrended_sst']
    print("data filtered!")
    SSTs = [detSST, sst_orig]# detSST]#, FiltSST, FiltSSTO2, SST_ssa1, SST_ssa]
    suffixes = ['detrended', 'original']#, 'detrended']#, '2_7yrs_butt1', 'gt_9m_butt2', 'ssa_onlygravest', 'ssa_less1y']
    idx_suff=0
    for sst in SSTs:
        sst = sst.chunk(dict(time=-1)).compute()
        # Detection and climatology in a single step...!
        clim, thresh, mhw_intensity = detection.detection_complete(var=sst,
                                                                period_clim=ClimPeriod,
                                                                period_detect=DetPeriod,
                                                                pctile=PcTile,
                                                                window=Win,
                                                                smoothing=Smooth,
                                                                smooth_window=SmoothWin,
                                                                minduration=MinDur)
        mhw_intensity = mhw_intensity.transpose('time', 'lat', 'lon')
        mhw_bool = xr.where(mhw_intensity>-999., 1, 0)
        anomalies = sst.groupby('time.dayofyear') - thresh
        DS = xr.Dataset(
            data_vars={
                'clim_'+suffixes[idx_suff] : (["dayofyear", "lat", "lon"], clim.data),
                'thresh_'+suffixes[idx_suff] : (["dayofyear", "lat", "lon"], thresh.data),
                'anom_'+suffixes[idx_suff] : (["time", "lat", "lon"], anomalies.data),
                'mhw_intensity_'+suffixes[idx_suff] : (["time", "lat", "lon"], mhw_intensity.data),
                'mhw_bool_'+suffixes[idx_suff] : (["time", "lat", "lon"], mhw_bool.data),
            },
            coords=dict(
                lon=(["lon"], mhw_intensity.lon.data),
                lat=(["lat"], mhw_intensity.lat.data),
                dayofyear=clim.dayofyear.data,
                time=mhw_intensity.time.data,
            ),
            attrs=dict(description="MHW detection Hobday et al. (2016): "+suffixes[idx_suff],
                    climatology_reference_period=ClimPeriod,
                    detection_period = DetPeriod,
                    pctile=PcTile,
                    climatology_window=Win,
                    smoothing = str(Smooth),
                    smoothing_window=SmoothWin if Smooth==True else None,
                    minimum_duration=MinDur,
                    ),
        )
        # Save the output in a netcdf:
        print("doing computations...")
        ds_out = DS.compute()
        ds_out.to_netcdf(cwd+'/../../results/'+expname+'_'+AOI+'_'+suffixes[idx_suff]+'_'+DetPeriod[0]+'_'+DetPeriod[1]+'.nc')
        ds_out
        del clim, thresh, mhw_intensity, DS, ds_out, anomalies, mhw_bool
        gc.collect()
        idx_suff+=1

# %%
