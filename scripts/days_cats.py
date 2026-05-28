import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar
ProgressBar().register()

cwd = os.getcwd()
resdir=cwd+'/../results/'
figdit=cwd+'/../figures/'
s='original'

cats = xr.open_dataset(resdir+'categories_'+s+'.nc', chunks='auto')['mhw_cats']

days_moderate = xr.where(cats==1, 1, np.nan).count(dim='time').rename('days_moderate')
days_strong = xr.where(cats==2, 1, np.nan).count(dim='time').rename('days_strong')
days_severe = xr.where(cats==3, 1, np.nan).count(dim='time').rename('days_severe')
days_extreme = xr.where(cats==4, 1, np.nan).count(dim='time').rename('days_extreme')

d = [days_moderate, days_strong, days_severe, days_extreme]
for day in d:
    print("writing out", day.name)
    day.to_netcdf(resdir+day.name+'.nc')


