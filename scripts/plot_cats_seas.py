#%%
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
figdir=cwd+'/../figures/'
model=['cglo', 'foam', 'glor', 'oras']
suffix=['original', 'detrended']
levidxs=["1", "9", "15", "19", "25", "28", "31"]
depths=["0.5m", "10m", "30m", "50m", "100m", "150m", "200m"]
for s in suffix:
    for j, l in enumerate(levidxs):
        print('plotting case'+s+', depth', depths[j])
        ds = xr.open_dataset(resdir+'categories_'+s+'_lev_'+l+'.nc', chunks='auto')
        cats = ds['mhw_cats'].sel(lat=slice(-60,60))
        seascats = cats.groupby('time.season').max('time').compute()
        plt.rcParams.update({'font.size':18})
        fg = seascats.plot(col='season', row='model', cmap='afmhot_r', levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cbar_kwargs={'ticks':[0, 1, 2, 3, 4]}, transform=ccrs.PlateCarree(), subplot_kws={"projection": ccrs.Robinson()}, aspect=1.5*(ds.dims['lon']/ds.dims['lat']))
        fg.map(lambda: plt.gca().coastlines())
        for i, ax in enumerate(fg.axs.flatten()):
            print(i)
            gl  = ax.gridlines(draw_labels=False)
            if (i%4==0):
                print('left labels plotting')
                gl.left_labels=True
                gl.bottom_labels=True if i==12 else False
            elif (i>12):
                print('bottom labels plotting')
                gl.bottom_labels=True
        fg.cbar.set_ticklabels(['None', 'Moderate', 'Strong', 'Severe', 'Extreme'])
        fg.fig.suptitle('MHW maximum category reached during each climatological season on '+s+' T @ '+depths[j]+': 1993-2019', fontweight='bold', x=0.5, y=1.01)
        #plt.savefig(figdir+'SeasonalCategories_'+s+'_lev_'+l+'.png')
        plt.show()
# %%
