
import xarray as xr
import numpy as np

def _sellonlatbox(x, lon_bnds, lat_bnds):
    return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))


def select_slice_along_dim(var, dim, start, end, method='values'):
    """
    return an Xarray sliced over dimension dim, from start to end, 
    depending on method: 
        - if method='value', then you have to provide 
        start and end values of the associate coordinate. 
        - if method='index', then you can provide
        start and end indices for the slicing.
    """
    if method=='values':
        return var.sel({dim:slice(start, end)})
    elif method=='indices':
        return var.isel({dim:slice(start, end)})
    else:
        raise Exception("Invalid method selected."+'\n'+
        "Valid methods are 'values' or 'indices'."+'\n'+
        "Use the Python built-in help() on this function to discover more")

def build_window_on_dim(var, dim, window):
    """
    This function builds a running window of size window 
    along a dimension dim, for the variable var, returning 
    an xarray object
    """
    """left_val = var[::-1, :,:].fillna(0.).rolling({'time':2*window+1}).construct('window').isel(window=slice(window+1, -window)).mean('window').groupby('time.dayofyear').mean('time').isel(dayofyear=slice(None, window+1)).mean().mean()
    right_val = var.fillna(0.).rolling({'time':2*window+1}).construct('window').isel(window=slice(window+1, -window)).mean('window').groupby('time.dayofyear').mean('time').isel(dayofyear=slice(-window, None)).mean().mean()
    print(left_val, right_val)"""
    return var.pad(pad_width={dim:window}, mode='wrap') \
        .rolling({dim:2*window+1}, center=True) \
            .construct('window') \
                .isel({dim:slice(window, -window)})

def get_percentile(var, dims, q, rechunk_dim, groupby_dim, rechunk=False):
    """
    This function calculates the q-th percentile along a
    groupby dimension and returns the corresponding xarray object.
    q : percentile threshold between 0 and 1;
    """
    if rechunk==True:
        return var.chunk({rechunk_dim:-1}) \
            .groupby(groupby_dim) \
            .quantile(q, dim=dims)
    else:
        return var.groupby(groupby_dim) \
                .quantile(q, dim=dims)

def smoothing(var, dim, smooth_window):
    """
    Here we do a smoothing on a dimension, 
    with a given smoothing window,
    rolling window and number of coefficients
    """
    #window = smooth_window
    #left_val = var.rolling({dim:2*window+1}).construct('window').isel(window=slice(window+1, -window)).mean('window').isel({dim:slice(None, window+1)}).mean().values
    #right_val = var.rolling({dim:2*window+1}).construct('window').isel(window=slice(window+1, -window)).mean('window').isel({dim:slice(-window, None)}).mean().values
    #return var.pad(pad_width={dim:window}, mode='constant', constant_values=(left_val, right_val)) \
    return (var.pad(pad_width={dim:smooth_window},mode="wrap") \
        .rolling({dim:2*smooth_window+1},center=True) \
            .construct("window") \
                .isel({dim:slice(smooth_window,-smooth_window)}).mean('window'))    
    
