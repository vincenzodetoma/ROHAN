#%%
import xarray as xr
import plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import diagnostics
import numpy as np
from dask.diagnostics import ProgressBar
ProgressBar().register()
import os

cwd = os.getcwd()
AOI='Med'
if AOI=='Med':
    event_years = [2003, 2006, 2017]
elif AOI=='TPac':
    event_years = [1997, 1998, 2014, 2015, 2016]
else:
    event_years = [1997, 1998]
ds_whole = xr.open_dataset(cwd+'/../data/Global/ESACCI_daily_1x1_Global_1982-2021.nc', 
        chunks='auto')
ds_whole = ds_whole.rename({'latitude':'lat', 'longitude':'lon'})
ds_whole = ds_whole.sortby(ds_whole.lat)
if AOI=='TPac':
    lonmin, lonmax, latmin, latmax = 120, 230, -30, 30
    DetPeriod = ["1982-01-01", "2021-12-31"]
elif AOI=='Med':
    lonmin, lonmax, latmin, latmax = 0, 15, 40, 45
    DetPeriod = ["1982-01-01", "2021-12-31"]
else:
    AOI='Global'
    lonmin, lonmax, latmin, latmax = 0, 360, -90, 90
    DetPeriod = ["1982-01-01", "2021-12-31"]
if AOI=='Global' or AOI=='TPac':
    ds_whole.coords['lon'] = (ds_whole.coords['lon']) % 360
    ds = ds_whole.sortby(ds_whole.lon)
else:
    ds = ds_whole
ds_cut = ds.sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))
print("loading data...")
startdate, enddate=DetPeriod[0], DetPeriod[1]
sst_orig = ds_cut['analysed_sst'].chunk({'time':-1}).sel(time=slice(startdate, enddate)).compute()
suffixes = ['original', 'detrended', '2_7yrs_butt1', 'gt_9m_butt2']
mhw = xr.open_mfdataset([AOI+'_'+suff+'_'+startdate+'_'+enddate+'.nc' for suff in suffixes])
mhw_cut = mhw
origSST, detrSST, filtSST, filtdetSST = [mhw_cut['mhw_intensity_'+suff].sel(time=slice(startdate, enddate)).compute() for suff in suffixes]
clim_orig, clim_detr, clim_filt, clim_filtdet = [mhw_cut['clim_'+suff].compute() for suff in suffixes]
thresh_orig, thresh_detr, thresh_filt, thresh_filtdet = [mhw_cut['thresh_'+suff].compute() for suff in suffixes]
list_SSTs = [origSST.groupby('time.year').max('time').sel(year=event_years), 
            detrSST.groupby('time.year').max('time').sel(year=event_years),
            filtSST.groupby('time.year').max('time').sel(year=event_years), 
            filtdetSST.groupby('time.year').max('time').sel(year=event_years)]
events = xr.concat(list_SSTs,
                   dim='method').assign_coords({'method': suffixes}).chunk({'method':len(list_SSTs)})
events = events.compute()
#%%
from importlib import reload
reload(plotting)
Proj = ccrs.Robinson(central_longitude=180) if AOI=='TPac' else ccrs.Robinson()
plotting.plot_faceted(var=events, 
                      row_dim='year', 
                      col_dim='method',
                      proj=Proj, 
                      colors='Spectral_r', 
                      levels=25,
                      draw_labels=True, savefig=True, 
                      figname=AOI+'_map_comparison_filtering.png')

# %%
from importlib import reload
reload(diagnostics)
if AOI=='Med':
    lonmin, lonmax, latmin, latmax =  0, 15, 40, 45 # NWMed
    startfocus, endfocus="2003-03-01", "2003-09-30"
elif AOI=='TPac':
    lonmin, lonmax, latmin, latmax = 190, 210, -5, 5
    startfocus, endfocus="2015-01-01", "2016-12-31"
origNWMed = diagnostics.spatial_mean(origSST, latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax).sel(time=slice(startfocus, endfocus))
detrNWMed = diagnostics.spatial_mean(detrSST, latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax).sel(time=slice(startfocus, endfocus))
filtNWMed = diagnostics.spatial_mean(filtSST, latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax).sel(time=slice(startfocus, endfocus))
filtdetNWMed = diagnostics.spatial_mean(filtdetSST, latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax).sel(time=slice(startfocus, endfocus))

dotsize=5
fig = plt.figure(345, figsize=(10,5), facecolor='white')
ax = fig.add_subplot(111)
origNWMed.plot(ax=ax, label=suffixes[0], marker='o', markersize=dotsize, markeredgecolor='k', alpha=0.5)
detrNWMed.plot(ax=ax, label=suffixes[1], marker='o', markersize=dotsize, markeredgecolor='k', alpha=0.5)
filtNWMed.plot(ax=ax, label=suffixes[2], marker='o', markersize=dotsize, markeredgecolor='k', alpha=0.5)
filtdetNWMed.plot(ax=ax, label=suffixes[3], marker='o', markersize=dotsize, markeredgecolor='k', alpha=0.5)
ax.set_xlabel('time [days]')
ax.set_ylabel('SST anomalies [K]')
ax.set_title(AOI+' MHW event, '+startfocus+' '+endfocus)
ax.legend()
plt.show()
# %%
reload(plotting)

ClimOrigNWMed = diagnostics.spatial_mean(clim_orig, latmin=latmin,latmax=latmax, lonmin=lonmin, lonmax=lonmax).compute()
ThreshOrigNWMed = diagnostics.spatial_mean(thresh_orig, latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax).compute()
ClimDetrNWMed = diagnostics.spatial_mean(clim_detr, latmin=latmin,latmax=latmax, lonmin=lonmin, lonmax=lonmax).compute()
ThreshDetrNWMed = diagnostics.spatial_mean(thresh_detr, latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax).compute()
ClimFiltNWMed = diagnostics.spatial_mean(clim_filt, latmin=latmin,latmax=latmax, lonmin=lonmin, lonmax=lonmax).compute()
ThreshFiltNWMed = diagnostics.spatial_mean(thresh_filt, latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax).compute()
ClimFiltdetNWMed = diagnostics.spatial_mean(clim_filtdet, latmin=latmin,latmax=latmax, lonmin=lonmin, lonmax=lonmax).compute()
ThreshFiltdetNWMed = diagnostics.spatial_mean(thresh_filtdet, latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax).compute()
sstNWMed = diagnostics.spatial_mean(sst_orig, latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax).sel(time=slice(startfocus, endfocus)).compute()


plotting.plot_marineheatwaves_timeseries(sst_orig=sstNWMed, 
                                         origMHW=origNWMed, 
                                         ThreshOrigMHW=ThreshOrigNWMed, 
                                         ClimOrigMHW=ClimOrigNWMed, 
                                         title='no filtering, no detrending', savefig=True, figname=AOI+'_'+suffixes[0]+'_MHW.png')
plotting.plot_marineheatwaves_timeseries(sst_orig=sstNWMed, 
                                         origMHW=detrNWMed, 
                                         ThreshOrigMHW=ThreshDetrNWMed, 
                                         ClimOrigMHW=ClimDetrNWMed, 
                                         title='linearly detrended', savefig=True, figname=AOI+'_'+suffixes[1]+'_MHW.png')
plotting.plot_marineheatwaves_timeseries(sst_orig=sstNWMed, 
                                         origMHW=filtNWMed, 
                                         ThreshOrigMHW=ThreshFiltNWMed, 
                                         ClimOrigMHW=ClimFiltNWMed, 
                                         title='butterworth filtered between 2-7 years', savefig=True, figname=AOI+'_'+suffixes[2]+'_MHW.png')
plotting.plot_marineheatwaves_timeseries(sst_orig=sstNWMed, 
                                         origMHW=filtdetNWMed, 
                                         ThreshOrigMHW=ThreshFiltdetNWMed, 
                                         ClimOrigMHW=ClimFiltdetNWMed, 
                                         title='butterworth filtered > 9m', savefig=True, figname=AOI+'_'+suffixes[3]+'_MHW.png')
# %%
