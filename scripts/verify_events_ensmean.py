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
dask.config.set(**{'array.slicing.split_large_chunks': False})
ProgressBar().register()
plt.rcParams['font.size'] = 15

cwd = os.getcwd()
resdir=cwd+'/../results/'
figdir=cwd+'/../figures/'
model=['ESACCISST', 'cglo', 'glor', 'oras']
model_ens = ['ESACCISST', 'ensemble_mean', 'cglo', 'glor', 'oras']
suffix=['original', 'detrended']
levidxs=["1"]#, "9", "15", "19", "25", "28", "31"]
depths=["0.5m"]#, "10m", "30m", "50m", "100m", "150m", "200m"]
datasets = []
levs = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
regions = {
    'Western Australia': 
        {"lat": slice(-40,-20), "lon": slice(90,120), 'time': slice('2010-11-01', '2011-07-01'),
         "lat_sub": slice(-32.5, -27.5), "lon_sub": slice(110, 115)},
    'Northeast Pacific Blob':
        {"lat": slice(15, 70), "lon": slice(-180, -100), 'time': slice('2014-05-01', '2016-08-01'),
         "lat_sub": slice(28,55), "lon_sub": slice(-140, -120)},
    'Mediterranean Sea':
        {"lat": slice(30, 50), "lon": slice(0,30), "time": slice("2006-04-01", "2006-11-01"),
         "lat_sub": slice(40, 42.5), "lon_sub": slice(6,8.5)},
    'Tasman Sea':
        {"lat": slice(-50,-35), "lon": slice(140, 160), "time": slice("2015-07-01", "2016-11-01"),
         "lat_sub": slice(-46, -39), "lon_sub": slice(147.5, 157.5)},
    'Great Barrier Reef':
        {"lat": slice(-25,-5), "lon": slice(135, 155), "time": slice("2015-11-15", "2016-07-15"),
         "lat_sub": slice(-15, -5), "lon_sub": slice(140, 145)},
    'Santa Barbara':
        {"lat": slice(30, 37), "lon": slice(-125, -115), "time": slice("2015-06-15", "2016-01-15"),
         "lat_sub": slice(31,34), "lon_sub": slice(-121, -118)},
    'Northwest Atlantic':
        {"lat": slice(35, 50), "lon": slice(-80, -50), "time": slice("2011-11-01", "2012-08-01"),
         "lat_sub": slice(42, 44), "lon_sub": slice(-70, -65)},
    }

# Pre-load all datasets once to avoid redundant I/O
print("Loading all datasets...")
ds_all = {}
ds_cats_all = {}
for var_name in model:
    ds_all[var_name] = {}
    for s in suffix:
        filenames = [resdir+var_name+'_Global_'+s+'_1993-01-01_2019-12-31_lev_'+idx+'.nc' for idx in levidxs]
        ds = xr.open_mfdataset([f for f in filenames], combine='nested', concat_dim='depth').assign_coords({'depth':depths})
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)
        ds_all[var_name][s] = ds

for s in suffix:
    ds_cats = xr.open_mfdataset(resdir+'categories_'+s+'_lev_1.nc')
    ds_cats.coords['lon'] = (ds_cats.coords['lon'] + 180) % 360 - 180
    ds_cats = ds_cats.sortby(ds_cats.lon)
    ds_cats_all[s] = ds_cats

print("Datasets loaded. Processing regions...")

for regname, r in zip(regions.keys(), regions.values()):
    print(f"Processing region: {regname}")
    
    # Store all time series data for all models and suffixes to avoid redundant loads
    ts_data = {}
    mhw_peaks_data = {}
    cats_data = {}
    
    # Pre-compute all time series data once
    for var_name in model:
        ts_data[var_name] = {}
        mhw_peaks_data[var_name] = {}
        cats_data[var_name] = {}
        
        for s in suffix:
            ds = ds_all[var_name][s]
            ds_sel = ds.sel({"lat" : r["lat"], "lon":r["lon"], 'time': r['time']})
            
            # Select spatial subset for time series
            anom_subset = ds_sel['anom_'+s].sel({"lat": r["lat_sub"], "lon": r["lon_sub"]})
            clim_subset = ds_sel['clim_'+s].sel({"lat": r["lat_sub"], "lon": r["lon_sub"]})
            thresh_subset = ds_sel['thresh_'+s].sel({"lat": r["lat_sub"], "lon": r["lon_sub"]})
            
            # Compute weighted means
            weights = np.cos(anom_subset.lat*np.pi/180.)
            anom_ts = anom_subset.weighted(weights).mean(dim=['lat', 'lon'])
            clim_ts = clim_subset.weighted(weights).mean(dim=['lat', 'lon'])
            thresh_ts = thresh_subset.weighted(weights).mean(dim=['lat', 'lon'])
            
            ts_data[var_name][s] = {'anom': anom_ts, 'clim': clim_ts, 'thresh': thresh_ts}
            
            print('Extracting peak for ', var_name, s)
            # Get peak info for maps
            ds_full = ds_sel#.compute()
            anom_event = ds_full['anom_'+s]
            time_idx = anom_ts.idxmax(dim='time')
            strtime = str(time_idx.values[0])
            single_peak = anom_event.sel(time=strtime)
            mhw_peaks_data[var_name][s] = (single_peak, time_idx, strtime)
            
            # Get categories
            ds_cats = ds_cats_all[s]
            ds_cats_sel = ds_cats.sel({"lat" : r["lat"], "lon":r["lon"], 'time': r['time']}).compute()
            single_cat = ds_cats_sel['mhw_cats'].sel({'time':strtime})
            cats_data[var_name][s] = single_cat
    
    # Compute ensemble mean and std once
    ens_data = {}
    for s in suffix:
        anom_list = [ts_data[m][s]['anom'] for m in ['cglo', 'glor', 'oras']]
        clim_list = [ts_data[m][s]['clim'] for m in ['cglo', 'glor', 'oras']]
        thresh_list = [ts_data[m][s]['thresh'] for m in ['cglo', 'glor', 'oras']]
        
        ens_anom = xr.concat(anom_list, dim='model').mean(dim='model')
        ens_anom_std = xr.concat(anom_list, dim='model').std(dim='model')
        ens_clim = xr.concat(clim_list, dim='model').mean(dim='model')
        ens_clim_std = xr.concat(clim_list, dim='model').std(dim='model')
        ens_thresh = xr.concat(thresh_list, dim='model').mean(dim='model')
        ens_thresh_std = xr.concat(thresh_list, dim='model').std(dim='model')
        
        ens_data[s] = {
            'anom': ens_anom, 'anom_std': ens_anom_std,
            'clim': ens_clim, 'clim_std': ens_clim_std,
            'thresh': ens_thresh, 'thresh_std': ens_thresh_std
        }
    
    # Build MHW_Peaks for spatial maps
    MHW_Peaks = []
    stringtimes = []
    xr_cats_method_all = []
    
    for j, var_name in enumerate(model):
        mhw_method = []
        cats_method = []
        
        for s in suffix:
            single_peak, time_idx, strtime = mhw_peaks_data[var_name][s]
            stringtimes.append(strtime)
            mhw_method.append(single_peak.rename({'depth':'ref_time'}).assign_coords({'ref_time': time_idx.values}))
            cats_method.append(cats_data[var_name][s])
        
        xr_mhw_method = xr.concat(mhw_method, dim='method').assign_coords({'method': suffix})
        xr_cats_method = xr.concat(cats_method, dim='method').assign_coords({'method': suffix})
        MHW_Peaks.append(xr_mhw_method)
        xr_cats_method_all.append(xr_cats_method)
    
    # Compute ensemble mean for spatial maps
    ens_mhw_peaks = []
    ens_mhw_std = []
    for s_idx, s in enumerate(suffix):
        mhw_method_ens = [MHW_Peaks[i].isel(method=s_idx) for i in [1, 2, 3]]
        ens_mhw = xr.concat(mhw_method_ens, dim='model', coords='different').mean(dim='model')
        ens_std = xr.concat(mhw_method_ens, dim='model', coords='different').std(dim='model')
        ens_mhw_peaks.append(ens_mhw)
        ens_mhw_std.append(ens_std)
    
    ens_mhw_concat = xr.concat(ens_mhw_peaks, dim='method').assign_coords({'method': suffix})
    ens_std_concat = xr.concat(ens_mhw_std, dim='method').assign_coords({'method': suffix})
    
    # Expand ensemble mean to match shape of individual models (add dummy model dim)
    ens_mhw_concat = ens_mhw_concat.expand_dims(dim={'model': 1})
    MHW_Peaks.append(ens_mhw_concat)
    
    # Prepare titles and plot data
    xr_MHW_Peaks = xr.concat(MHW_Peaks, dim='model', coords='different')
    titles = [model_ens[i] + ", " + stringtimes[i*2][:10] for i in range(len(model))] + ['ensemble_mean']
    xr_MHW_Peaks = xr_MHW_Peaks.assign_coords({'model': titles})
    to_plot = xr_MHW_Peaks.mean('ref_time')
    
    # Get dims for aspect ratio
    ds_sample = ds_all[model[0]][suffix[0]].sel({"lat" : r["lat"], "lon":r["lon"], 'time': r['time']})
    
    print("Creating Figure 1: SST Anomaly Maps...")
    fg = to_plot.plot.contourf(col='model', row='method', transform=ccrs.PlateCarree(), 
                                subplot_kws={'projection': ccrs.PlateCarree()}, 
                                aspect=ds_sample.dims['lon']/ds_sample.dims['lat'], 
                                cbar_kwargs={'label': 'SST Anomaly [K]'}, levels=levs)
    fg.set_titles(template="{value}")
    fg.map(lambda: plt.gca().coastlines())
    
    anom_subset = ds_all[model[0]][suffix[0]].sel({"lat": r["lat_sub"], "lon": r["lon_sub"]})
    for pan_num, (i, ax) in enumerate(zip(fg.name_dicts.flatten(), fg.axs.flatten())):
        geom = geometry.box(minx=anom_subset.lon.min(),maxx=anom_subset.lon.max(),
                           miny=anom_subset.lat.min(), maxy=anom_subset.lat.max())
        ax.add_geometries([geom], crs=cartopy.crs.PlateCarree(), edgecolor='k', facecolor="None", linewidth=3)
        gl = ax.gridlines(draw_labels=False)
        if pan_num==0:
            gl.left_labels=True
        elif pan_num==5:
            gl.left_labels=True
            gl.bottom_labels=True
        elif pan_num>5:
            gl.bottom_labels=True
        
        # Add std contours for ensemble mean
        if (pan_num % len(model_ens)) == (len(model_ens) - 1):
            for ens_std in ens_mhw_std:
                ens_std_plot = ens_std.mean('ref_time')
                ens_std_plot.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='black', 
                                         linewidths=1, linestyles='--', levels=5)
                break
    
    plt.savefig(figdir+'mhw_sst_anomaly.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Creating Figure 2: MHW Categories Maps...")
    cats_to_plot = xr.concat(xr_cats_method_all, dim='model').assign_coords({'model': titles[:-1]}).squeeze()
    fg = cats_to_plot.plot.contourf(col='model', row='method', cmap='afmhot_r', 
                                     levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 
                                     cbar_kwargs={'ticks':[0, 1, 2, 3, 4]}, 
                                     transform=ccrs.PlateCarree(), 
                                     subplot_kws={"projection": ccrs.PlateCarree()}, 
                                     aspect=(ds_sample.dims['lon']/ds_sample.dims['lat']))
    fg.set_titles(template="{value}")
    fg.cbar.set_ticklabels(['None', 'Moderate', 'Strong', 'Severe', 'Extreme'])
    fg.map(lambda: plt.gca().coastlines())
    for pan_num, (i, ax) in enumerate(zip(fg.name_dicts.flatten(), fg.axs.flatten())):
        geom = geometry.box(minx=anom_subset.lon.min(),maxx=anom_subset.lon.max(),
                           miny=anom_subset.lat.min(), maxy=anom_subset.lat.max())
        ax.add_geometries([geom], crs=cartopy.crs.PlateCarree(), edgecolor='k', facecolor="None", linewidth=3)
        gl = ax.gridlines(draw_labels=False)
        if pan_num==0:
            gl.left_labels=True
        elif pan_num==4:
            gl.left_labels=True
            gl.bottom_labels=True
        elif pan_num>4:
            gl.bottom_labels=True
    
    plt.savefig(figdir+'mhw_categories.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Creating Figure 3: Time Series with Error Bars...")
    fig, axes = plt.subplots(len(suffix), len(model_ens), figsize=(20, 10))
    
    for row_idx, s in enumerate(suffix):
        for col_idx in range(len(model) + 1):
            ax = axes[row_idx, col_idx] if len(suffix) > 1 else axes[col_idx]
            
            if col_idx < len(model):
                # Individual models
                var_name = model[col_idx]
                anom_ts = ts_data[var_name][s]['anom']
                clim_ts = ts_data[var_name][s]['clim']
                thresh_ts = ts_data[var_name][s]['thresh']
                
                SST = anom_ts.groupby('time.dayofyear') + thresh_ts
                THRESHOLD = (SST - anom_ts)
                CLIM = (THRESHOLD - (THRESHOLD.groupby('time.dayofyear') - clim_ts))
                
                ax.plot(SST.time.values, SST.values, color='k', label='SST', linewidth=1.5)
                ax.plot(THRESHOLD.time.values, THRESHOLD.values, color='green', label='Threshold', linewidth=1.5)
                ax.plot(CLIM.time.values, CLIM.values, color='blue', label='Climatology', linewidth=1.5)
                ax.set_title(f'{var_name} {s}', fontsize=11, fontweight='bold')
            
            else:
                # Ensemble mean with error bars
                anom_ens = ens_data[s]['anom']
                clim_ens = ens_data[s]['clim']
                clim_std_ens = ens_data[s]['clim_std']
                thresh_ens = ens_data[s]['thresh']
                thresh_std_ens = ens_data[s]['thresh_std']
                
                SST_ens = anom_ens.groupby('time.dayofyear') + thresh_ens
                THRESHOLD_ens = (SST_ens - anom_ens)
                CLIM_ens = (THRESHOLD_ens - (THRESHOLD_ens.groupby('time.dayofyear') - clim_ens))
                
                ax.plot(SST_ens.time.values, SST_ens.values, color='k', label='SST', linewidth=1.5)
                ax.plot(THRESHOLD_ens.time.values, THRESHOLD_ens.values, color='green', label='Threshold', linewidth=1.5)
                ax.plot(CLIM_ens.time.values, CLIM_ens.values, color='blue', label='Climatology', linewidth=1.5)
                
                # Add error bars as shaded regions
                ax.fill_between(CLIM_ens.time.values, 
                               (CLIM_ens - clim_std_ens).values, 
                               (CLIM_ens + clim_std_ens).values, 
                               alpha=0.2, color='blue')
                ax.fill_between(THRESHOLD_ens.time.values, 
                               (THRESHOLD_ens - thresh_std_ens).values, 
                               (THRESHOLD_ens + thresh_std_ens).values, 
                               alpha=0.2, color='green')
                
                ax.set_title(f'ensemble_mean {s}', fontsize=11, fontweight='bold')
            
            ax.set_ylabel('Temperature [K]', fontsize=10)
            ax.set_xlabel('Time', fontsize=10)
            ax.grid(True, alpha=0.3)
            if col_idx == 0:
                ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(figdir+'time_series_ensemble.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Completed region: {regname}")

print("All processing complete!")

# %%
