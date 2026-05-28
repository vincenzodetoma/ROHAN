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

cwd = os.getcwd()
resdir=cwd+'/../results/'
figdir=cwd+'/../figures/'
model=['cglo', 'foam', 'glor', 'oras']
suffix=['original', 'detrended']
levidxs=["1", "9", "15", "19", "25", "28", "31"]
depths=["0.5m", "10m", "30m", "50m", "100m", "150m", "200m"]
datasets = []
list_cats = ['days_moderate', 'days_strong', 'days_severe', 'days_extreme']
colors=['whitesmoke', 'yellow', 'teal', 'darkorange']
thresholds = [180, 30, 15, 1]
for j, var_name in enumerate(list_cats):
    thresh = thresholds[j]
    for s in suffix:
        filenames = [resdir+var_name+'_'+s+'_lev_'+idx+'.nc' for idx in levidxs]
        ds = xr.open_mfdataset([f for f in filenames], combine='nested', concat_dim='depth').assign_coords({'depth':depths})
        print(filenames, ds)
        to_plot = ds[var_name].sel(lat=slice(-60,60)).compute()
        to_plot2 = xr.where(to_plot>thresh, 1., np.nan)
        list_depths = xr.concat([xr.where(to_plot2.isel(depth=d)==1., float(depths[d][:-1]), np.nan) for d in range(len(levidxs))], dim='depth').assign_coords({'depth':depths})
        cumdep = list_depths.sum('depth')
        cumdep.to_netcdf(resdir+'cumdep_events_'+var_name+'_'+s+'.nc')
        surf = xr.where(cumdep==0.5, 12.5, 0.)
        subsurf = xr.where(np.logical_and(cumdep%1==0, cumdep!=0), 17.5, 0.)
        mixed = xr.where(np.logical_and(cumdep>0.5,cumdep%1!=0), 37.5, 0.)
        SHALLOW = xr.where(np.logical_and(cumdep!=0, cumdep<190.5), to_plot, np.nan).rename('mhw_days')
        DEEP = xr.where(np.logical_and(cumdep!=0, cumdep>=190.5), to_plot, np.nan).rename('mhw_days')
        SHALLOW.to_netcdf(resdir+'shallow_events_'+var_name+'_'+s+'.nc')
        DEEP.to_netcdf(resdir+'deep_events_'+var_name+'_'+s+'.nc')
        SURF = xr.where(cumdep==0.5, to_plot, np.nan).rename('mhw_days')
        SUBSURF = xr.where(np.logical_and(cumdep%1==0, cumdep!=0), to_plot, np.nan).rename('mhw_days')
        MIXED = xr.where(np.logical_and(cumdep>0.5,cumdep%1!=0), to_plot, np.nan).rename('mhw_days')
        SURF.to_netcdf(resdir+'surface_events_'+var_name+'_'+s+'.nc')
        SUBSURF.to_netcdf(resdir+'subsurface_events_'+var_name+'_'+s+'.nc')
        MIXED.to_netcdf(resdir+'mixed_events_'+var_name+'_'+s+'.nc')
        to_plot = surf + subsurf + mixed
        plt.rcParams.update({'font.size':18})
        fg = to_plot.plot.contourf(col='model', col_wrap=2, colors=colors, levels=[0, 10, 15, 20, 60],
                          transform=ccrs.PlateCarree(), 
                          cbar_kwargs={'ticks':[5., 12.5, 17.5, 40]}, subplot_kws={"projection": ccrs.PlateCarree()}, sharex=True, sharey=True, aspect=1.*(ds.dims['lon']/ds.dims['lat']))
        fg.map(lambda: plt.gca().coastlines())
        for i, ax in enumerate(fg.axs.flatten()):
            print(i)
            ax.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black', facecolor='white')
            gl  = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=True)
            gl.xlabels_top, gl.ylabels_right=False, False
            if (i==0 or i==1):
                gl.xlabels_bottom=False
            if (i==1 or i==3):
                gl.ylabels_left=False
            if i==2:
                gl.xlabels_bottom=False
        fg.cbar.set_ticklabels(['None', 'Surface', 'Subsurface', 'Mixed'])
        fg.fig.suptitle('Vertical Class of '+var_name.replace('days_', '')+' MHWs on '+s+': 1993-2019 ('+str(thresh)+' days threshold)', fontweight='bold', x=0.5, y=1.01)
        plt.savefig(figdir+var_name.replace('days_', '')+'_MHWs_depth_model_'+s+'.png')
        plt.show()
# %%
