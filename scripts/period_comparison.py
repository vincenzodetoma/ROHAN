#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cartopy.crs as ccrs
import cartopy
from dask.diagnostics import ProgressBar
from shapely import geometry
import dask
import pandas as pd
dask.config.set(**{'array.slicing.split_large_chunks': False})
ProgressBar().register()

cwd = os.getcwd()
datadir=cwd+'/../data/'
resdir=cwd+'/../results/'
figdir=cwd+'/../figures/'
model=['ESACCISST', 'cglo', 'foam', 'glor', 'oras']
suffix=['_original', '_detrended']
levidxs=["1"]#, "9", "15", "19", "25", "28", "31"]
depths=["0.5m"]#, "10m", "30m", "50m", "100m", "150m", "200m"]
datasets = []
#%%
ds_model_red = []
for j, var_name in enumerate(model):
    method_ds = []
    for s in suffix:
        filenames = [resdir+var_name+'_Global'+s+'_1993-01-01_2019-12-31_lev_'+idx+'.nc' for idx in levidxs]
        vars_to_drop = ['analysis_uncertainty', 'sea_ice_fraction', 'mask']
        ds = xr.open_mfdataset([f for f in filenames], combine='nested', concat_dim='depth').assign_coords({'depth':depths}).rename({'clim'+s: 'clim', 'anom'+s: 'anom', 'thresh'+s: 'thresh', 'mhw_intensity'+s: 'mhw_intensity', 'mhw_bool'+s: 'mhw_bool'})
        time_coords = pd.date_range('1993-01-01', '2019-12-31')
        ds = ds.assign_coords({'time':time_coords})
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)
        method_ds.append(ds)
    ds_method = xr.concat([d for d in method_ds], dim='method').assign_coords({'method': suffix})
    ds_model_red.append(ds_method)
REDDS = xr.concat([d.rename() for d in ds_model_red], dim='model').assign_coords({'model': model})
sst = xr.where(REDDS['mhw_intensity']>0, REDDS['mhw_intensity'], np.nan)
sstmean = sst.mean('time')
count_surface = sst.count('time') / len(time_coords)
count_surface = xr.where(count_surface!=0, count_surface, np.nan).compute()
count_surface.to_netcdf(resdir+'surface_percentage_total_global.nc')

# %%
ds = REDDS
fg = count_surface.isel(depth=0).plot.contourf(col='model', row='method', cmap='turbo', levels=[0., 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5], cbar_kwargs={'ticks':[0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475], 'orientation': 'vertical', 'pad': 0.0095, 'label': 'MHW days [%]'}, transform=ccrs.PlateCarree(), subplot_kws={"projection": ccrs.PlateCarree()}, aspect=1.2*(ds.dims['lon']/ds.dims['lat']))
fg.map(lambda: plt.gca().coastlines())
for i, ax in enumerate(fg.axs.flatten()):
    print(i)
    ax.add_feature(cartopy.feature.LAND)
    gl  = ax.gridlines(draw_labels=False)
    if (i%5==0):
        print('left labels plotting')
        gl.ylabels_left=True
        gl.xlabels_bottom=True if i==5 else False
    elif (i>4):
        print('bottom labels plotting')
        gl.xlabels_bottom=True
fg.cbar.set_ticklabels(['0-0.5', '0.5-0.1', '0.1-0.15', '0.15-0.20', '0.20-0.25', '0.25-0.30', '0.30-0.35', '0.35-0.40', '0.40-0.45', '0.45-0.50'])
fg.fig.suptitle('Surface percentage of MHW days', fontweight='bold', x=0.5, y=1.01)
#plt.savefig(figdir+'SeasonalCategories_'+s+'_lev_'+l+'.png')
#plt.show()# %%

# %%
