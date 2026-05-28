#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar
import dask
import pandas as pd
dask.config.set(**{'array.slicing.split_large_chunks': False})
ProgressBar().register()

cwd = os.getcwd()
resdir=cwd+'/../results/'
figdir=cwd+'/../figures/'
model=['cglo', 'foam', 'glor', 'oras']
suffix=['original']#, 'detrended']
levidxs=["1", "9", "15", "19", "25", "28", "31"]
depths=["0.5m", "10m", "30m", "50m", "100m", "150m", "200m"]
for s in suffix:
    j=0
    for l in levidxs:
        ds_list = []
        if l=="1":
            model = ['ESACCISST', 'cglo', 'foam', 'glor', 'oras']
        else:
            model = ['cglo', 'foam', 'glor', 'oras']
        for m in model:
            filename=m+'_Global_'+s+'_1993-01-01_2019-12-31_lev_'+l+'.nc'
            time_range = pd.date_range(start='01/01/1993', end='31/12/2019', freq='D', normalize=True)
            print(time_range)
            ds = xr.open_dataset(resdir+filename, chunks='auto')
            print(filename, ds, '\n')
            ds_list.append(ds)

        ds = xr.concat(ds_list, dim='model').assign_coords({'model': model})
        clim = ds['clim_'+s]
        thresh = ds['thresh_'+s]
        anom = ds['anom_'+s].assign_coords({'time':time_range})
        if resdir+'anomalies_'+s+'_lev_'+l+'.nc' in os.listdir(resdir):
            print("Anomalies file already exists, skipping...")
            continue
        else:
            print("writing out anomalies file for level "+l)
            anom.rename('mhw_anoms').to_netcdf(resdir+'anomalies_'+s+'_lev_'+l+'.nc')
        #cat1=thresh - clim
        #cat2, cat3, cat4 = 2*cat1, 3*cat1, 4*cat1
        #cats = xr.where(anom.groupby('time.dayofyear') - cat1 < 0, 0., np.nan)
        #cats = xr.where(np.logical_and(anom.groupby('time.dayofyear') - cat1 > 0., anom.groupby('time.dayofyear') - cat2 < 0.), 1, cats)
        #cats = xr.where(np.logical_and(anom.groupby('time.dayofyear') - cat2 > 0.,  anom.groupby('time.dayofyear') - cat3 < 0.), 2, cats)
        #cats = xr.where(np.logical_and(anom.groupby('time.dayofyear') - cat3 > 0., anom.groupby('time.dayofyear') - cat4 < 0.), 3, cats)
        #cats = xr.where(anom.groupby('time.dayofyear') - cat4 > 0., 4, cats)
        #cats.rename('mhw_cats').to_netcdf(resdir+'categories_'+s+'_lev_'+l+'.nc')
        #
        #seascats = cats.groupby('time.season').max('time').compute()
#
        #days_moderate = xr.where(cats==1, 1, np.nan).count(dim='time').rename('days_moderate')
        #days_strong = xr.where(cats==2, 1, np.nan).count(dim='time').rename('days_strong')
        #days_severe = xr.where(cats==3, 1, np.nan).count(dim='time').rename('days_severe')
        #days_extreme = xr.where(cats==4, 1, np.nan).count(dim='time').rename('days_extreme')
#
        #d = [days_moderate, days_strong, days_severe, days_extreme]
        #for day in d:
        #    print("writing out", day.name)
        #    day.to_netcdf(resdir+day.name+'_'+s+'_lev_'+l+'.nc')
        #plt.rcParams.update({'font.size':18})
        #fg = seascats.plot(col='season', row='model', cmap='afmhot_r', levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cbar_kwargs={'ticks':[0, 1, 2, 3, 4]}, transform=ccrs.PlateCarree(), subplot_kws={"projection": ccrs.Robinson()}, aspect=ds.dims['lon']/ds.dims['lat'])
        #fg.map(lambda: plt.gca().coastlines())
        #fg.cbar.set_ticklabels(['None', 'Moderate', 'Strong', 'Severe', 'Extreme'])
        #fg.fig.suptitle('MHW maximum category reached during each climatological season on '+s+' T @ '+depths[j]+': 1993-2019', fontweight='bold', x=0.5, y=1.01)
        #plt.savefig(figdir+'SeasonalCategories_'+s+'_lev_'+l+'.png')
        j=j+1

# %%
