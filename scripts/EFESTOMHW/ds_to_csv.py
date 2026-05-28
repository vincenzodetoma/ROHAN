#%%
import xarray as xr
import os
def xarray_to_csv(array, filename):
    df = array.to_dataframe()
    df.to_csv(filename)
    
def write_series(array, namefile):
    with open(namefile, "w") as txt_file:
        for line in array:
            print(line)
            txt_file.write(str(line) + "\n")        

AOIs = ['Med', 'TPac']
for AOI in AOIs:
    if AOI=='TPac':
        lonmin, lonmax, latmin, latmax = 120, 230, -30, 30
        DetPeriod = ["1998-01-01", "2018-12-31"]
        latpoint, lonpoint = 0, 220
    elif AOI=='Med':
        lonmin, lonmax, latmin, latmax = -6, 36, 30, 46
        DetPeriod = ["2019-01-01", "2021-12-31"]
        lonpoint, latpoint = 7.5, 42.5
    sst_orig = xr.open_dataset(AOI+'_singlePoint_orig_latlon_'+str(latpoint)+str(lonpoint)+'.nc')['analysed_sst']
    FiltSST = xr.open_dataset(AOI+'_singlePoint_butter_latlon_'+str(latpoint)+str(lonpoint)+'.nc')['butterworth_gt_1y']
    fastSST = xr.open_dataset(AOI+'_singlePoint_fastSSA_latlon_'+str(latpoint)+str(lonpoint)+'.nc')['FastSST_after_SSA_gt_1y']
    slowSST = xr.open_dataset(AOI+'_singlePoint_slowSSA_latlon_'+str(latpoint)+str(lonpoint)+'.nc')['SlowSST_after_SSA_gt_1y']
    xarray_to_csv(sst_orig, AOI+'_singlePoint_orig_latlon_'+str(latpoint)+str(lonpoint)+'.csv')
    xarray_to_csv(FiltSST, AOI+'_singlePoint_butter_latlon_'+str(latpoint)+str(lonpoint)+'.csv')
    xarray_to_csv(fastSST, AOI+'_singlePoint_fastSSA_latlon_'+str(latpoint)+str(lonpoint)+'.csv')
    xarray_to_csv(slowSST, AOI+'_singlePoint_slowSSA_latlon_'+str(latpoint)+str(lonpoint)+'.csv')
    write_series(sst_orig.values, 'sst_orig'+AOI+'.txt')
    write_series(FiltSST.values, 'sst_butter'+AOI+'.txt')
    write_series(fastSST.values, 'sst_fast'+AOI+'.txt')
    write_series(slowSST.values, 'sst_slow'+AOI+'.txt')

    
# %%
