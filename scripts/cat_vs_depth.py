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
from tqdm import tqdm
dask.config.set(**{'array.slicing.split_large_chunks': False})
ProgressBar().register()

cwd = os.getcwd()
resdir=cwd+'/../results/'
figdir=cwd+'/../figures/'
model=['cglo', 'foam', 'glor', 'oras']
suffix=['original']
levidxs=["1", "9", "15", "19", "25", "28", "31"]
depths=["0.5m", "10m", "30m", "50m", "100m", "150m", "200m"]
datasets = []
levs = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
regions = {
    #'Western Australia': 
    #    {"lat": slice(-40,-20), "lon": slice(90,120), 'time': slice('2010-11-01', '2011-07-01'),
    #     "lat_sub": slice(-32.5, -27.5), "lon_sub": slice(110, 115), "time_sub": slice('2011-02-01', '2011-04-01')},
    #'Northeast Pacific Blob':
    #    {"lat": slice(15, 70), "lon": slice(-180, -100), 'time': slice('2014-05-01', '2016-08-01'),
    #     "lat_sub": slice(28,55), "lon_sub": slice(-140, -120), "time_sub": slice('2015-08-01', '2015-10-01')},
    #'Mediterranean Sea':
    #    {"lat": slice(30, 50), "lon": slice(0,30), "time": slice("2006-04-01", "2006-11-01"),
    #     "lat_sub": slice(40, 42.5), "lon_sub": slice(6,8.5), "time_sub": slice("2006-07-01", "2006-09-01")},
    #'Tasman Sea':
    #    {"lat": slice(-50,-35), "lon": slice(140, 160), "time": slice("2015-07-01", "2016-11-01"),
    #     "lat_sub": slice(-46, -39), "lon_sub": slice(147.5, 157.5), "time_sub": slice("2016-01-01", "2016-03-01")},
    #'Great Barrier Reef':
    #    {"lat": slice(-25,-5), "lon": slice(135, 155), "time": slice("2015-11-15", "2016-07-15"),
    #     "lat_sub": slice(-15, -5), "lon_sub": slice(140, 145), "time_sub": slice("2016-02-15", "2016-04-15")},
    'Santa Barbara':
        {"lat": slice(30, 37), "lon": slice(-125, -115), "time": slice("2015-06-15", "2016-01-15"),
         "lat_sub": slice(31,34), "lon_sub": slice(-121, -118), "time_sub": slice("2015-09-15", "2015-11-15")},
    'Northwest Atlantic':
        {"lat": slice(35, 50), "lon": slice(-80, -50), "time": slice("2011-11-01", "2012-08-01"),
         "lat_sub": slice(42, 44), "lon_sub": slice(-70, -65), "time_sub": slice("2012-02-01", "2012-04-01")},
    }
for regname, r in zip(regions.keys(), regions.values()):
    print(regname, r)
    cats_method = []
    for s in suffix:
        filenames = [resdir+'categories_'+s+'_lev_'+idx+'.nc' for idx in levidxs]
        ds = xr.open_mfdataset([f for f in filenames], combine='nested', concat_dim='depth').assign_coords({'depth':depths}).sel(model=model).assign_coords({'model':model})
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)
        ds_calset = ds.sel({"lat" : r["lat"], "lon":r["lon"], 'time': r['time']})
        cats_method.append(ds_calset)
        initfilepath = '/store/data_store/Ocean_Rean/GLOBAL_REANALYSIS_PHY_001_031/global-reanalysis-phy-001-031-grepv2-daily/global'
        print("initfilepath", initfilepath)
        ds_mld = [xr.open_mfdataset(initfilepath+'/grepv2_daily_*.nc', parallel=True)['mlotst_'+m] for m in model]
        ds_mld = xr.concat(ds_mld, dim='model').assign_coords({'model':model}).rename({'latitude':'lat', 'longitude':'lon'})
        ds_mld.coords['lon'] = (ds_mld.coords['lon'] + 180) % 360 - 180
        ds_mld = ds_mld.sortby(ds_mld.lon)
        ds_mld = ds_mld.sel({"lat" : r["lat_sub"], "lon":r["lon_sub"], 'time': r['time']})
        print("ds_mld", ds_mld)
        del ds_calset
    xr_cats_method = xr.concat(cats_method, dim='method').assign_coords({'method': suffix})
    cats_subset = xr_cats_method.sel({"lat": r["lat_sub"], "lon": r["lon_sub"]})
    print("cats subset", cats_subset)
    cats = cats_subset['mhw_cats'].max(dim=['lat', 'lon']).compute()
    mlds = ds_mld.weighted(np.cos(ds_mld.lat*np.pi/180.)).mean(dim=['lat', 'lon']).compute()
    #idxs = cats_subset['mhw_cats'].argmax(dim=['lat', 'lon'])
    #lat_idx = idxs['lat'].compute()
    #lon_idx = idxs['lon'].compute()
    #max_lat = np.zeros(cats.shape)
    #max_lon = np.zeros(cats.shape)
    #for m in tqdm(range(cats.shape[0]), desc='method'):
    #    for z in tqdm(range(cats.shape[1]), desc='depth'):
    #        for mod in tqdm(range(cats.shape[2]), desc='model'):
    #            for t in tqdm(range(cats.shape[3]), desc='time'):
    #                print("in the for loop", m,z,mod,t)
    #                max_lat[m,z,mod,t] = cats_subset.lat[lat_idx[m,z,mod,t].values]
    #                max_lon[m,z,mod,t] = cats_subset.lon[lon_idx[m,z,mod,t].values]
    #print(f"Maximum occurs at latitude: {max_lat.values}, longitude: {max_lon.values}")
    print("doing plot of category vs depth")
    p = cats.isel(depth=slice(None, None, -1), method=0).plot.contourf(col='model', col_wrap=2, cmap='afmhot_r', levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cbar_kwargs={'ticks':[0, 1, 2, 3, 4]}, add_colorbar=False)
    for i, ax in enumerate(p.axes.flat):
        print(i)
        Ax = ax.twinx()
        mld = mlds.isel(model=i)
        #mld.plot(ax=Ax, color='black', linewidth=2)
        Ax.set_title('')
        Ax.set_xlabel('')
        Ax.invert_yaxis()
        Ax.set_ylim(200, 0)
        mld_labels = [0.5, 10, 30, 50, 100, 150, 200]
        n_ticks = len(mld_labels)
        mld_positions = np.linspace(0, 200, n_ticks)
        Ax.set_yticks(mld_positions)            # Evenly spaced positions
        Ax.set_yticklabels(mld_labels)          # Custom labels (nonlinear)
        if i==1 or i==3:
            Ax.set_ylabel('MLD [m]', color='black')
        else:
            Ax.set_yticklabels([])
        import numpy as np
        from scipy.interpolate import interp1d
        interp = interp1d(mld_labels, mld_positions, bounds_error=False, fill_value="extrapolate")
        mld_mapped = interp(mld.values)
        Ax.plot(mld.time, mld_mapped, color='blue')
        
        # Grab the first QuadContourSet from the facets to use for colorbar
    cf = p._mappables[0]  # the filled contour set
    
    # Add a new colorbar axis manually (adjust [left, bottom, width, height] as needed)
    cbar_ax = p.fig.add_axes([1.1, 0.125, 0.020, 0.85])  # right-side vertical bar
    cbar = p.fig.colorbar(cf, cax=cbar_ax, ticks=[0, 1, 2, 3, 4])
    cbar.set_label("MHW Category")
    cbar.set_ticklabels(["Weak", "Moderate", "Strong", "Severe", "Extreme"])
    p.fig.suptitle(f'MHW Maximum Category in {regname}', y=1.02)
    plt.show(block=False)
    plt.savefig(figdir+'cats_depth'+regname+'.png', dpi=400)
    #xr_max_lat = xr.DataArray(max_lat, dims=['method', 'depth', 'model', 'time'], coords={'method':suffix, 'depth':depths, 'model':model, 'time':cats.time})
    #xr_max_lon = xr.DataArray(max_lon, dims=['method', 'depth', 'model', 'time'], coords={'method':suffix, 'depth':depths, 'model':model, 'time':cats.time})
    #print("doing plot of max lat vs depth")
    #xr_max_lat.isel(method=0).plot.contourf(col='model', col_wrap=2, cmap='jet', cbar_kwargs={'label':'Latitude [degrees]'})
    ##plt.show(block=False)
    #plt.savefig(figdir+'max_lat_depth'+regname+'.png', dpi=400)
    #print("doing plot of max lon vs depth")
    #xr_max_lon.isel(method=0).plot.contourf(col='model', col_wrap=2, cmap='jet', cbar_kwargs={'label':'Longitude [degrees]'})
    ##plt.show(block=False)
    #plt.savefig(figdir+'max_lon_depth'+regname+'.png', dpi=400)
# %%
