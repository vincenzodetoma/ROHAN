#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cartopy.crs as ccrs
import cartopy
from dask.diagnostics import ProgressBar
ProgressBar().register()
import pandas as pd
import seaborn as sns
#%%

cwd = os.getcwd()
resdir=cwd+'/../results/'
figdir=cwd+'/../figures/'
model=['ESACCISST', 'cglo', 'foam', 'glor', 'oras']
levidxs=["1", "9", "15", "19", "25", "28", "31"]
depths=["0.5m", "10m", "30m", "50m", "100m", "150m", "200m"]
datasets = []
suffix=['original', 'detrended']
list_vert = ['surface', 'subsurface', 'mixed', 'shallow', 'deep']
list_cats = ['days_moderate', 'days_strong', 'days_severe', 'days_extreme']
colors=['whitesmoke', 'yellow', 'teal', 'darkorange']
cats_ds = []
for j, var_name in enumerate(list_cats):
    met_ds = []
    for s in suffix:
        vert_ds = []
        for v in list_vert:
            ds_model = []
            for m in model:
                if m=='ESACCISST':
                    vert='surface'
                else:
                    vert=v
                ds = xr.open_dataset(resdir+vert+'_events'+'_'+var_name+'_'+s+'.nc').sel({'model':m}).max('depth')
                ds_model.append(ds)
                print(ds)
            ds_model = xr.concat([m for m in ds_model], dim='model').assign_coords({'model':model})
            vert_ds.append(ds_model)
        vert_ds = xr.concat([m for m in vert_ds], dim='vertical_type').assign_coords({'vertical_type': list_vert})
        print(vert_ds)
        met_ds.append(vert_ds)
    met_ds = xr.concat([s for s in met_ds], dim='method').assign_coords({'method':suffix})
    print(met_ds)
    cats_ds.append(met_ds)
cats_ds = xr.concat([c for c in cats_ds], dim='category').assign_coords({'category': list_cats}).sel({'vertical_type': list_vert})
#cats_ds = xr.where(cats_ds['mhw_days']!=0, cats_ds, np.nan)
print(cats_ds)
# %%
#model=['ESACCISST', 'cglo', 'foam', 'glor', 'oras']
#levidxs=["1"]#, "9", "15", "19", "25", "28", "31"]
#depths=["0.5m"]#, "10m", "30m", "50m", "100m", "150m", "200m"]
#datasets = []
#ds_model_red = []
#for j, var_name in enumerate(model):
#    method_ds = []
#    for s in suffix:
#        if var_name=='ESACCISST':
#            levidxs=["1"]#, "9", "15", "19", "25", "28", "31"]
#            depths=["0.5m"]#, "10m", "30m", "50m", "100m", "150m", "200m"]
#        else:
#            levidxs=["1", "9", "15", "19", "25", "28", "31"]
#            depths=["0.5m", "10m", "30m", "50m", "100m", "150m", "200m"]
#        filenames = [resdir+var_name+'_Global_'+s+'_1993-01-01_2019-12-31_lev_'+idx+'.nc' for idx in levidxs]
#        vars_to_drop = ['analysis_uncertainty', 'sea_ice_fraction', 'mask']
#        ds = xr.open_mfdataset([f for f in filenames], combine='nested', concat_dim='depth').assign_coords({'depth':depths}).rename({'clim_'+s: 'clim', 'anom_'+s: 'anom', 'thresh_'+s: 'thresh', 'mhw_intensity_'+s: 'mhw_intensity', 'mhw_bool_'+s: 'mhw_bool'})
#        time_coords = pd.date_range('1993-01-01', '2019-12-31')
#        ds = ds.assign_coords({'time':time_coords})
#        #ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
#        #ds = ds.sortby(ds.lon)
#        method_ds.append(ds)
#    ds_method = xr.concat([d for d in method_ds], dim='method').assign_coords({'method': suffix})
#    ds_model_red.append(ds_method)
#REDDS = xr.concat([d.rename() for d in ds_model_red], dim='model').assign_coords({'model': model})
#sst = xr.where(REDDS['mhw_intensity']>0, REDDS['mhw_intensity'], 0)
#sstmean = sst.mean('time').sel(lat=slice(-60, 60)).mean('depth')
#print(sstmean)
#%%
time = pd.date_range('1993-01-01', '2019-12-31')
N = len(time)
print(N)
#percentage = xr.where(~np.isnan(sstmean), (cats_ds/N).max('category'), np.nan).compute()
percentage = (cats_ds/N).max('category').compute()
# %%
suffix=['original']#, 'detrended']
reduced_vert = ['surface']#, 'subsurface', 'mixed']
model=["ESACCISST"]
for s in suffix:
    to_plot = percentage['mhw_days'].sel({'method':s, 'vertical_type': reduced_vert}).sel({'model':model})
    print(to_plot)
    #to_plot.plot.contourf(col='model', row='vertical_type', cmap='jet', vmin=0, vmax=0.05, levels=21)
    fg = to_plot.plot.contourf(col='model', row='vertical_type', cmap='turbo', levels=[0.0, 0.005, 0.01, 0.015, 0.020, 0.025, 0.030, 0.035, 0.04, 0.045, 0.05], cbar_kwargs={'ticks':[0.0025, 0.0075, 0.0125, 0.0175, 0.0225, 0.0275, 0.0325, 0.0375, 0.0425, 0.0475], 'orientation': 'vertical', 'pad': 0.015, 'label': 'MHW days [%]'}, transform=ccrs.PlateCarree(), subplot_kws={"projection": ccrs.PlateCarree()}, aspect=.87*(cats_ds.dims['lon']/cats_ds.dims['lat']))
    fg.map(lambda: plt.gca().coastlines())
    for i, ax in enumerate(fg.axs.flatten()):
        print(i)
        ax.add_feature(cartopy.feature.LAND)
        gl  = ax.gridlines(draw_labels=False)
        if (i%5==0):
            print('left labels plotting')
            gl.left_labels=True
            gl.bottom_labels=True if i==10 else True
        elif (i>9):
            print('bottom labels plotting')
            gl.bottom_labels=True
    fg.cbar.set_ticklabels(['0-0.5', '0.5-0.1', '0.1-0.15', '0.15-0.20', '0.20-0.25', '0.25-0.30', '0.30-0.35', '0.35-0.40', '0.40-0.45', '0.45-0.50'])
    fg.fig.suptitle('Percentage of Surface MHW days', fontweight='bold', x=0.4, y=1.01)
    plt.show() 
# %%
suffix=['original']#, 'detrended']
reduced_vert = ['shallow', 'deep']
model=['cglo', 'foam', 'glor', 'oras']
for s in suffix:
    to_plot = percentage['mhw_days'].sel({'method':s, 'vertical_type': reduced_vert, 'model': model})
    print(to_plot)
    #to_plot.plot.contourf(col='model', row='vertical_type', cmap='jet', vmin=0, vmax=0.05, levels=21)
    fg = to_plot.plot.contourf(col='vertical_type', row='model', cmap='jet', levels=[0.0, 0.005, 0.01, 0.015, 0.020, 0.025, 0.030, 0.035, 0.04, 0.045, 0.05], cbar_kwargs={'ticks':[0.0025, 0.0075, 0.0125, 0.0175, 0.0225, 0.0275, 0.0325, 0.0375, 0.0425, 0.0475], 'orientation': 'vertical', 'pad': 0.035, 'label': 'MHW days [%]'}, transform=ccrs.PlateCarree(), subplot_kws={"projection": ccrs.PlateCarree()}, aspect=.87*(cats_ds.dims['lon']/cats_ds.dims['lat']))
    fg.map(lambda: plt.gca().coastlines())
    for i, ax in enumerate(fg.axs.flatten()):
        print(i)
        ax.add_feature(cartopy.feature.LAND)
        gl  = ax.gridlines(draw_labels=False)
        if (i%2==0):
            print('left labels plotting')
            gl.left_labels=True
            gl.bottom_labels=True if i==6 else False
        elif (i>6):
            print('bottom labels plotting')
            gl.bottom_labels=True
    fg.cbar.set_ticklabels(['0-0.5', '0.5-0.1', '0.1-0.15', '0.15-0.20', '0.20-0.25', '0.25-0.30', '0.30-0.35', '0.35-0.40', '0.40-0.45', '0.45-0.50'])
    fg.fig.suptitle('Percentage of MHW days', fontweight='bold', x=0.4, y=1.01)
    plt.show()
# %%
DF = []
for v in reduced_vert:
    to_plot_df = percentage['mhw_days'].sel({'vertical_type': v, 'model': model}).rename(v)
    DF.append(to_plot_df.to_dataset())
DF = xr.merge([d for d in DF], compat='override').to_dataframe().reset_index().set_index('model')
#%%
sns.pairplot(DF, hue='vertical_type')
plt.show()
#%%
total_number_MHW_days = xr.where(~np.isnan(percentage['mhw_days'].mean('vertical_type')), (percentage*N).sum('vertical_type'), np.nan).compute()
# %%
to_plot = total_number_MHW_days['mhw_days']
#to_plot.plot.contourf(col='model', row='vertical_type', cmap='jet', vmin=0, vmax=0.05, levels=21)
fg = to_plot.plot.contourf(col='model', row='method', cmap='turbo', levels=[5, 30, 60, 90, 180, 210, 240, 270, 300, 365, 365*2, 365*3, 365*4, 365*5], cbar_kwargs={'orientation': 'vertical', 'pad': 0.015, 'label': 'MHW days [days]'}, transform=ccrs.PlateCarree(), subplot_kws={"projection": ccrs.PlateCarree()}, aspect=.87*(cats_ds.dims['lon']/cats_ds.dims['lat']))
fg.map(lambda: plt.gca().coastlines())
for i, ax in enumerate(fg.axs.flatten()):
    print(i)
    ax.add_feature(cartopy.feature.LAND)
    gl  = ax.gridlines(draw_labels=False)
    if (i%5==0):
        print('left labels plotting')
        gl.left_labels=True
        gl.bottom_labels=True if i==5 else False
    elif (i>5):
        print('bottom labels plotting')
        gl.bottom_labels=True
fg.fig.suptitle('Total Number of MHW days', fontweight='bold', x=0.4, y=1.01)
plt.show()
# %%
