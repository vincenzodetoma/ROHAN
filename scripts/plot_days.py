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

categories=['moderate', 'strong', 'severe', 'extreme']
days_cats = [xr.open_dataset(resdir+'days_'+c+'.nc', chunks='auto')['days_'+c].rename('days') for c  in categories]

days_cats = xr.concat([ds for ds in days_cats], dim='category').assign_coords({'category':categories})

