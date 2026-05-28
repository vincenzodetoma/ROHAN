#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import cartopy.crs as ccrs
import cartopy
from dask.diagnostics import ProgressBar
from shapely import geometry
import dask
dask.config.set(**{'array.slicing.split_large_chunks': False})
ProgressBar().register()
plt.rcParams['font.size'] = 15

def plot_anomaly_ts(SST, THRESHOLD, CLIM, cat2, cat3, cat4, anom_event_ts, var_name, s, regname, ens_std=None):
    print("Checking dimension names of input to plotting function:", SST.dims, THRESHOLD.dims, CLIM.dims, cat2.dims, cat3.dims, cat4.dims, anom_event_ts.dims, ens_std.dims if ens_std is not None else "N/A")
    print("Checking dimensions of input to plotting function:", SST.shape, THRESHOLD.shape, CLIM.shape, cat2.shape, cat3.shape, cat4.shape, anom_event_ts.shape, ens_std.shape if ens_std is not None else "N/A")
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    SST.plot(ax=ax, color='k', label='SST')
    THRESHOLD.plot(ax=ax, color='green', label='thresh')
    CLIM.plot(ax=ax, color='blue', label='clim')
    (CLIM + cat2).plot(ax=ax, ls='-.', color='green', label='2xth')
    (CLIM + cat3).plot(ax=ax, ls='--', color='green', label='3xth')
    (CLIM + cat4).plot(ax=ax, ls=':', color='green', label='4xth')
    if ens_std is not None:
        ax.fill_between(SST.time, (SST - ens_std).squeeze(), (SST + ens_std).squeeze(), color='grey', alpha=0.9, label='EnsStdDev')
    ax.fill_between(SST.time, (THRESHOLD).squeeze(), (SST).squeeze(), where=(anom_event_ts).squeeze()>=0, color='khaki')
    ax.fill_between(SST.time, (CLIM + cat2).squeeze(), (SST).squeeze(), where=(SST >= (CLIM + cat2)).squeeze(), color='darkorange')
    ax.fill_between(SST.time, (CLIM + cat3).squeeze(), (SST).squeeze(), where=(SST >= (CLIM + cat3)).squeeze(), color='orangered')
    ax.fill_between(SST.time, (CLIM + cat4).squeeze(), (SST).squeeze(), where=(SST >= (CLIM + cat4)).squeeze(), color='darkred')
    ax.set_title(f"{var_name} {s} \n {regname}") if var_name!='Ensemble Mean' else ax.set_title(f"EnsMean {s} \n {regname}")
    ax.grid()
    plt.xticks(rotation=90)
    if ens_std is not None:
        ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1), ncol=1)
    fig.savefig(figdir+var_name+'_'+s+'_SST_anomaly_timeseries_'+regname.replace(" ", "_")+'.png', dpi=300)
    plt.show()

cwd = os.getcwd()
resdir=cwd+'/../results/'
figdir=cwd+'/../figures/'
cachedir = Path(cwd) / '..' / 'cache'
cachedir.mkdir(exist_ok=True)

model=['ESACCISST', 'cglo', 'glor', 'oras']
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
for regname, r in zip(regions.keys(), regions.values()):
    print(regname, r)
    regname_underscore = regname.replace(" ", "_")
    cache_file = cachedir / f"{regname_underscore}_results.nc"
    
    # Check if results already exist
    if cache_file.exists():
        print(f"Loading cached results from {cache_file}")
        ds_cached = xr.open_dataset(cache_file)
        xr_MHW_Peaks = ds_cached['MHW_Peaks']
        xr_ANOM_ts_model = ds_cached['ANOM_ts_model']
        xr_CLIM_ts_model = ds_cached['CLIM_ts_model']
        xr_THRESH_ts_model = ds_cached['THRESH_ts_model']
        xr_CAT2_ts_model = ds_cached['CAT2_ts_model']
        xr_CAT3_ts_model = ds_cached['CAT3_ts_model']
        xr_CAT4_ts_model = ds_cached['CAT4_ts_model']
        stringtimes = ds_cached.attrs['stringtimes'].split(',')
        anom_subset = ds_cached['anom_subset']
        xr_cats_method = ds_cached['cats_method']
    else:
        print(f"No cached results found. Computing and caching to {cache_file}")
        # Compute results
        MHW_Peaks = []
        ANOM_ts_model = []
        CLIM_ts_model = []
        THRESH_ts_model = []
        CAT2_ts_model = []
        CAT3_ts_model = []
        CAT4_ts_model = []
        stringtimes = []
        for j, var_name in enumerate(model):
            mhw_method = []
            cats_method = []
            anom_ts_method = []
            clim_ts_method = []
            thresh_ts_method = []
            cat2_ts_method = []
            cat3_ts_method = []
            cat4_ts_method = []
            for s in suffix:
                filenames = [resdir+var_name+'_Global_'+s+'_1993-01-01_2019-12-31_lev_'+idx+'.nc' for idx in levidxs]
                ds = xr.open_mfdataset([f for f in filenames], combine='nested', concat_dim='depth').assign_coords({'depth':depths})
                ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
                ds = ds.sortby(ds.lon)
                ds_cats = xr.open_mfdataset(resdir+'categories_'+s+'_lev_1.nc').sel({"model": model})
                print(ds_cats)
                ds_cats.coords['lon'] = (ds_cats.coords['lon'] + 180) % 360 - 180
                ds_cats = ds_cats.sortby(ds_cats.lon)
                ds_calset = ds_cats.sel({"lat" : r["lat"], "lon":r["lon"], 'time': r['time']}).compute()
                print('opening', filenames)
                ds_sel = ds.sel({"lat" : r["lat"], "lon":r["lon"], 'time': r['time']}).compute()
                anom_event = ds_sel['anom_'+s]
                anom_subset = anom_event.sel({"lat": r["lat_sub"], "lon": r["lon_sub"]})
                anom_event_ts = anom_subset.weighted(np.cos(anom_subset.lat*np.pi/180.)).mean(dim=['lat', 'lon'])
                clim_event_ts = ds_sel['clim_'+s].sel({"lat": r["lat_sub"], "lon": r["lon_sub"]}).weighted(np.cos(anom_subset.lat*np.pi/180.)).mean(dim=['lat', 'lon'])
                thresh_event_ts = ds_sel['thresh_'+s].sel({"lat": r["lat_sub"], "lon": r["lon_sub"]}).weighted(np.cos(anom_subset.lat*np.pi/180.)).mean(dim=['lat', 'lon'])
                SST = anom_event_ts.groupby('time.dayofyear') + thresh_event_ts
                THRESHOLD = (SST - anom_event_ts)
                cat1 = (THRESHOLD.groupby('time.dayofyear') - clim_event_ts) 
                CLIM = (THRESHOLD - cat1) 
                cat2 = 2*cat1
                cat3 = 3*cat1
                cat4 = 4*cat1
                plot_anomaly_ts(SST, THRESHOLD, CLIM, cat2, cat3, cat4, anom_event_ts, var_name, s, regname)
                if var_name=='ESACCISST':
                    time_idx = anom_event_ts.idxmax(dim='time')
                    strtime = str(time_idx.values[0])
                stringtimes.append(strtime)
                single_peak = anom_event.sel(time=strtime)
                mhw_method.append(single_peak.rename({'depth':'ref_time'}).assign_coords({'ref_time': time_idx.values}))
                single_cat = ds_calset['mhw_cats'].sel({'time':strtime})
                cats_method.append(single_cat)
                anom_ts_method.append(anom_event_ts)
                clim_ts_method.append(clim_event_ts)
                thresh_ts_method.append(thresh_event_ts)
                cat2_ts_method.append(cat2)
                cat3_ts_method.append(cat3)
                cat4_ts_method.append(cat4)
            xr_mhw_method = xr.concat(mhw_method, dim='method').assign_coords({'method': suffix})
            xr_cats_method = xr.concat(cats_method, dim='method').assign_coords({'method': suffix})
            xr_anom_ts_method = xr.concat(anom_ts_method, dim='method').assign_coords({'method': suffix})
            xr_clim_ts_method = xr.concat(clim_ts_method, dim='method').assign_coords({'method': suffix})
            xr_thresh_ts_method = xr.concat(thresh_ts_method, dim='method').assign_coords({'method': suffix})
            xr_cat2_ts_method = xr.concat(cat2_ts_method, dim='method').assign_coords({'method': suffix})
            xr_cat3_ts_method = xr.concat(cat3_ts_method, dim='method').assign_coords({'method': suffix})
            xr_cat4_ts_method = xr.concat(cat4_ts_method, dim='method').assign_coords({'method': suffix})
            MHW_Peaks.append(xr_mhw_method)
            ANOM_ts_model.append(xr_anom_ts_method)
            CLIM_ts_model.append(xr_clim_ts_method)
            THRESH_ts_model.append(xr_thresh_ts_method)
            CAT2_ts_model.append(xr_cat2_ts_method)
            CAT3_ts_model.append(xr_cat3_ts_method)
            CAT4_ts_model.append(xr_cat4_ts_method)
        xr_MHW_Peaks = xr.concat(MHW_Peaks, dim='model')
        xr_ANOM_ts_model = xr.concat(ANOM_ts_model, dim='model').assign_coords({'model': model})
        xr_CLIM_ts_model = xr.concat(CLIM_ts_model, dim='model').assign_coords({'model': model})
        xr_THRESH_ts_model = xr.concat(THRESH_ts_model, dim='model').assign_coords({'model': model})
        xr_CAT2_ts_model = xr.concat(CAT2_ts_model, dim='model').assign_coords({'model': model})
        xr_CAT3_ts_model = xr.concat(CAT3_ts_model, dim='model').assign_coords({'model': model})
        xr_CAT4_ts_model = xr.concat(CAT4_ts_model, dim='model').assign_coords({'model': model})
        
        # Save results to cache
        ds_to_save = xr.Dataset({
            'MHW_Peaks': xr_MHW_Peaks,
            'ANOM_ts_model': xr_ANOM_ts_model,
            'CLIM_ts_model': xr_CLIM_ts_model,
            'THRESH_ts_model': xr_THRESH_ts_model,
            'CAT2_ts_model': xr_CAT2_ts_model,
            'CAT3_ts_model': xr_CAT3_ts_model,
            'CAT4_ts_model': xr_CAT4_ts_model,
            'anom_subset': anom_subset,
            'cats_method': xr_cats_method,
        })
        ds_to_save.attrs['stringtimes'] = ','.join(stringtimes)
        ds_to_save.to_netcdf(cache_file)
        print(f"Cached results to {cache_file}")
    
    ds_cached = xr.open_dataset(cache_file)
    ds_sel = ds_cached.sel({"lat" : r["lat"], "lon":r["lon"], 'time': r['time']})
    anom_event = ds_sel['MHW_Peaks']
    anom_subset = anom_event.sel({"lat": r["lat_sub"], "lon": r["lon_sub"]})
    xr_ANOM_ts_model = ds_cached['ANOM_ts_model']
    xr_CLIM_ts_model = ds_cached['CLIM_ts_model']
    xr_THRESH_ts_model = ds_cached['THRESH_ts_model']
    xr_CAT2_ts_model = ds_cached['CAT2_ts_model']
    xr_CAT3_ts_model = ds_cached['CAT3_ts_model']
    xr_CAT4_ts_model = ds_cached['CAT4_ts_model']
    xr_MHW_Peaks = ds_cached['MHW_Peaks']
    xr_cats_method = ds_cached['cats_method']
    # Plotting (always runs)
    print("Plotting results for region:", regname)
    for i, var_name in enumerate(model):
        for s in suffix:
            anom_event_ts = xr_ANOM_ts_model.sel(model=var_name).sel({'method':s})
            clim_ts = xr_CLIM_ts_model.sel(model=var_name).sel({'method':s})
            thresh_ts = xr_THRESH_ts_model.sel(model=var_name).sel({'method':s})
            SST = anom_event_ts.groupby('time.dayofyear') + thresh_ts
            THRESHOLD = (SST - anom_event_ts)
            cat1 = (THRESHOLD.groupby('time.dayofyear') - clim_ts) 
            CLIM = (THRESHOLD - cat1) 
            cat2 = 2*cat1
            cat3 = 3*cat1
            cat4 = 4*cat1
            plot_anomaly_ts(SST, THRESHOLD, CLIM, cat2, cat3, cat4, anom_event_ts, var_name, s, regname)
    
    print("doing ensemble mean plots", model[1:])
    ens_ANOM_ts = xr_ANOM_ts_model.sel(model=model[1:]).mean('model')
    ens_STD_ts = xr_ANOM_ts_model.sel(model=model[1:]).std('model')
    ens_CLIM_ts = xr_CLIM_ts_model.sel(model=model[1:]).mean('model')
    ens_THRESH_ts = xr_THRESH_ts_model.sel(model=model[1:]).mean('model')
    ens_SST_ts = ens_ANOM_ts.groupby('time.dayofyear') + ens_THRESH_ts
    ens_THRESH_ts = (ens_SST_ts - ens_ANOM_ts)
    ens_CAT1 = (ens_THRESH_ts.groupby('time.dayofyear') - ens_CLIM_ts)
    ens_CLIM_ts = (ens_THRESH_ts - ens_CAT1)
    ens_CAT2_ts = xr_CAT2_ts_model.sel(model=model[1:]).mean('model')
    ens_CAT3_ts = xr_CAT3_ts_model.sel(model=model[1:]).mean('model')
    ens_CAT4_ts = xr_CAT4_ts_model.sel(model=model[1:]).mean('model')
    for s in suffix:
        plot_anomaly_ts(ens_SST_ts.sel({'method':s}), ens_THRESH_ts.sel({'method':s}), ens_CLIM_ts.sel({'method':s}), ens_CAT2_ts.sel({'method':s}), ens_CAT3_ts.sel({'method':s}), ens_CAT4_ts.sel({'method':s}), ens_ANOM_ts.sel({'method':s}), 'Ensemble Mean', s, regname, ens_std=ens_STD_ts.sel({'method':s}))
    times = xr_MHW_Peaks.ref_time.values
    titles = [model[i] + f"\n" + stringtimes[i][:10] for i in range(len(model))]
    titles.append(f"Ensemble Mean\n" + stringtimes[0][:10])
    ens_MHW_peak = xr_MHW_Peaks.sel(model=model[1:]).mean('model')
    std_MHW_peak = xr_MHW_Peaks.sel(model=model[1:]).std('model')
    xr_MHW_Peaks = xr.concat([xr_MHW_Peaks, ens_MHW_peak.expand_dims({'model':['Ensemble Mean']})], dim='model')
    xr_MHW_Peaks = xr_MHW_Peaks.assign_coords({'model': titles})
    
    # Reorder to show ESACCI first, then Ensemble Mean, then other models
    model_order = [titles[0], titles[-1]] + titles[1:-1]
    xr_MHW_Peaks = xr_MHW_Peaks.sel(model=model_order)
    
    to_plot = xr_MHW_Peaks.mean('ref_time')
    fg = to_plot.plot.contourf(col='model', row='method', transform=ccrs.PlateCarree(), subplot_kws={'projection': ccrs.PlateCarree()}, aspect=ds_cached.dims['lon']/ds_cached.dims['lat'], cbar_kwargs={'label': 'SST Anomaly [K]'}, levels=levs)
    fg.set_titles(template="{value}")
    fg.map(lambda: plt.gca().coastlines())
    for pan_num, (i, ax) in enumerate(zip(fg.name_dicts.flatten(), fg.axs.flatten())):
        # Create a Rectangle patch
        geom = geometry.box(minx=anom_subset.lon.min(),maxx=anom_subset.lon.max(),miny=anom_subset.lat.min(), maxy=anom_subset.lat.max())
        ax.add_geometries([geom], crs=cartopy.crs.PlateCarree(), edgecolor='k', facecolor="None", linewidth=3)
        gl = ax.gridlines(draw_labels=False)
        if pan_num==0:
            gl.left_labels=True
        elif pan_num==5:
            gl.left_labels=True
            gl.bottom_labels=True
        elif pan_num>5:
            gl.bottom_labels=True
        #if pan_num==4 or pan_num==9:
        #    if pan_num==4:
        #        s='original'
        #    else:
        #        s='detrended'
        #    print(std_MHW_peak.sel({'method':s}).squeeze().shape)
        #    c = std_MHW_peak.sel({'method':s}).squeeze().plot.contour(ax=ax, colors='black', transform=ccrs.PlateCarree(), levels=5)
        #    c.clabel(fmt='%1.2f', inline=True, fontsize=8)
    plt.show()
    
    ens_MHW_cats = xr_cats_method.sel(model=model[1:]).mean('model')
    std_MHW_cats = xr_cats_method.sel(model=model[1:]).std('model')
    xr_cats_method = xr.concat([xr_cats_method, ens_MHW_cats.expand_dims({'model':['Ensemble Mean']})], dim='model')
    xr_cats_method = xr_cats_method.assign_coords({'model': titles})
    xr_cats_method = xr_cats_method.sel(model=model_order)
    
    to_plot = xr_cats_method.squeeze()
    fg = to_plot.plot.contourf(col='model', row='method', cmap='afmhot_r', levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cbar_kwargs={'ticks':[0, 1, 2, 3, 4]}, transform=ccrs.PlateCarree(), subplot_kws={"projection": ccrs.PlateCarree()}, aspect=(ds_sel.dims['lon']/ds_sel.dims['lat']))
    fg.set_titles(template="{value}")
    fg.cbar.set_ticklabels(['None', 'Moderate', 'Strong', 'Severe', 'Extreme'])
    fg.map(lambda: plt.gca().coastlines())
    for pan_num, (i, ax) in enumerate(zip(fg.name_dicts.flatten(), fg.axs.flatten())):
        # Create a Rectangle patch
        geom = geometry.box(minx=anom_subset.lon.min(),maxx=anom_subset.lon.max(),miny=anom_subset.lat.min(), maxy=anom_subset.lat.max())
        ax.add_geometries([geom], crs=cartopy.crs.PlateCarree(), edgecolor='k', facecolor="None", linewidth=3)
        gl = ax.gridlines(draw_labels=False)
        if pan_num==0:
            gl.left_labels=True
        elif pan_num==5:
            gl.left_labels=True
            gl.bottom_labels=True
        elif pan_num>5:
            gl.bottom_labels=True
        #if pan_num==4 or pan_num==9:
        #    if pan_num==4:
        #        s='original'
        #    else:
        #        s='detrended'
        #    print(std_MHW_cats.sel({'method':s}).squeeze().shape)
        #    c = std_MHW_cats.sel({'method':s}).squeeze().plot.contour(ax=ax, colors='black', transform=ccrs.PlateCarree(), levels=5)
        #    c.clabel(fmt='%1.3f', inline=True, fontsize=8)
    plt.show()
# %%
# TO DO: Show categories time vs depth plot?
