#%%
################################## MODULES IMPORT ###########################
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import detection
from dask.diagnostics import ProgressBar
import dask
import preprocess
import plotting
import gc
import fire
from dask.distributed import Client
from dask.config import set
from functools import partial 
import utils 
import sys
import time

def main(data_dir='../../data/', filename='ESACCI_daily_1x1_Global_1982-2021.nc', sst_name='analysed_sst',
         AOIs=['Global'], DetPeriod=["1982-01-01", "2021-12-31"], ClimPeriod=["1991-01-01", "2020-12-31"],
         PcTile=0.9, Win=5, Smooth=True, SmoothWin=15, MinDur=5,
         detrend_linear=False, butterworth_o1=False, butterworth_o2=False, SSA=False, only_gravest=False,
         n_work=16):
    dask.config.set(**{'array.slicing.split_large_chunks': True})
    ProgressBar().register()
     # Start distributed client with 16 workers, each using up to 6GB RAM
    #client = Client(n_workers=n_work)
    #print("Dask cluster started with the following configuration:")
    #print(client)
    #print(f"Dask Dashboard: {client.dashboard_link}")
    ################################ LOADING DATA ###############################
    cwd = os.getcwd()
    print("working in directory: ", cwd, flush=True)
    print("loading data from folder: ", cwd+'/'+data_dir, flush=True)
    print("loading data file: ", filename, flush=True)
    print("Detection Period --complete:", DetPeriod, flush=True)
    print("Detection Period: ", DetPeriod[0], DetPeriod[1], flush=True)
    print("Climatology Period: ", ClimPeriod[0], ClimPeriod[1], flush=True)
    time.sleep(10)
    for AOI in tqdm(AOIs, desc="processing AOIs", unit="AOI", colour="blue"):
        if AOI=='TPac':
            lon_bnds, lat_bnds = (120, 230), (-30, 30)
        elif AOI=='Med':
            lon_bnds, lat_bnds = (-6, 36), (30, 46)
        elif AOI=='NWMed':
            lon_bnds, lat_bnds = (0, 15), (40, 45)
        else:
            AOI='Global'
            lon_bnds, lat_bnds = (-180, 180), (-90, 90)
        lonmin, lonmax, latmin, latmax = lon_bnds[0], lon_bnds[1], lat_bnds[0], lat_bnds[1]
        if filename.endswith('.nc'):
            ds_whole = xr.open_dataset(cwd+'/'+data_dir+filename, 
                chunks='auto')
        else:
            ds_whole = []
            filelist = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    filelist.append(os.path.join(root, file))
            filelist = sorted(filelist)
            print("first and last files of filelist \n", filelist[0], "\n", filelist[-1]) 
            print("selecting region", lat_bnds, lon_bnds)
            partial_lonlat = partial(utils._sellonlatbox, lon_bnds=lon_bnds, lat_bnds=lat_bnds) 
            ds_whole = xr.open_mfdataset([f for f in tqdm(filelist,
                                                          desc="loading single files", 
                                                          unit="files", 
                                                          colour="green")], 
                chunks='auto', combine='by_coords', preprocess=partial_lonlat, parallel=True,
                drop_variables=['time_bnds', 'lat_bnds', 'lon_bnds', 'analysed_sst_uncertainty', 'sea_ice_fraction', 'mask'])
        if 'latitude' in ds_whole.dims:
            ds_whole = ds_whole.rename({'latitude':'lat', 'longitude':'lon'})
        print("After if on latitude")
        ds_whole = ds_whole.sortby(ds_whole.lat)
        ds_whole = ds_whole.chunk(dict(time=-1))
        print(ds_whole)
        if AOI=='Global' or AOI=='TPac':
            ds_whole.coords['lon'] = (ds_whole.coords['lon']) % 360
            ds = ds_whole.sortby(ds_whole.lon)
        else:
            ds = ds_whole
            
        ds_cut = ds.sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))
        print("loading data...")
        if 'depth' in ds_cut.dims:
            ds_cut = ds_cut.squeeze('depth')
        sst_orig = ds_cut[sst_name] - 273.15 if 'ESACCISST' in filename else ds_cut[sst_name]
        ############################## Set detection Parameters #####################
        SSTs = [sst_orig]
        initial_suffix = 'detrend' if 'detrended' in filename else 'original'
        suffixes = [initial_suffix]
        if butterworth_o1 or butterworth_o2:
            fs = (2*np.pi)/(24*60*60) # Sampling Frequency in Hertz
            if butterworth_o1:
                lowcut1 = (2*np.pi)/(7*365*24*60*60)
                highcut1 = (2*np.pi)/(2*365*24*60*60) #ENSO lower limit freq
                FiltSST = preprocess.xr_butter_bandstop_filter(sst_orig, lowcut=lowcut1, highcut=highcut1, fs=fs, order=1, lin_detrend=False)
                SSTs.append(FiltSST)
                suffixes.append('2_7yrs_butt1')
            elif butterworth_o2:
                # Filter from 9 months above
                lowcut2 = (2*np.pi)/(400*365*24*60*60) 
                highcut2 = (2*np.pi)/(0.75*365*24*60*60)
                FiltSSTO2 = preprocess.xr_butter_bandstop_filter(sst_orig, lowcut=lowcut2, highcut=highcut2, fs=fs, order=2, lin_detrend=False)
                SSTs.append(FiltSSTO2)
                suffixes.append('gt_9m_butt2')
        if detrend_linear:
            detSST = preprocess.detrend_linear(sst_orig, dim='time')
            SSTs.append(detSST)
            suffixes.append('detrended')
        if SSA:
            if only_gravest:
                sst_in1, sstFast1, sstSlow1, periods1 = preprocess.xr_SSA_decompose(sst_orig, modes=int(len(sst_orig)/10), only_gravest=True)
                SST_ssa1 = sstFast1 + sstSlow1.mean('time')
                SSTs.append(SST_ssa1)
                suffixes.append('ssa_onlygravest')
            else:
                sst_in, sstFast, sstSlow, periods = preprocess.xr_SSA_decompose(sst_orig, modes=int(len(sst_orig)/10), only_gravest=False)
                SST_ssa = sstFast + sstSlow.mean('time')
                SSTs.append(SST_ssa)
                suffixes.append('ssa_less1y')
        for idx_suff, sst in enumerate(SSTs):
            print(sst)
            sst = sst
            # Detection and climatology in a single step...!
            clim, thresh, mhw_intensity, mhw_labels = detection.detection_complete(var=sst,
                                                                                   period_clim=ClimPeriod,  
                                                                                   period_detect=DetPeriod,  
                                                                                   pctile=PcTile,    
                                                                                   window=Win, 
                                                                                   smoothing=Smooth,    
                                                                                   smooth_window=SmoothWin,    
                                                                                   minduration=MinDur)
            mhw_intensity = mhw_intensity.transpose('time', 'lat', 'lon')
            mhw_labels = mhw_labels.transpose('time', 'lat', 'lon')
            DS = xr.Dataset(
                data_vars={
                    'clim_'+suffixes[idx_suff] : (["dayofyear", "lat", "lon"], clim.data),
                    'thresh_'+suffixes[idx_suff] : (["dayofyear", "lat", "lon"], thresh.data),
                    'mhw_intensity_'+suffixes[idx_suff] : (["time", "lat", "lon"], mhw_intensity.data),
                    'mhw_labels_'+suffixes[idx_suff] : (["time", "lat", "lon"], mhw_labels.data),
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
            ds_out.to_netcdf(cwd+'/../../results/'+'detection_'+AOI+'_'+suffixes[idx_suff]+'_'+DetPeriod[0]+'_'+DetPeriod[1]+filename)
            ds_out
            del clim, thresh, mhw_intensity, DS, ds_out, mhw_labels
            gc.collect()
            idx_suff+=1

if __name__ == "__main__":
    ProgressBar().register() 
    fire.Fire(main)
