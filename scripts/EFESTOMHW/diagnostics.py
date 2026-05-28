import xarray as xr
import numpy as np


def spatial_mean(var, latmin, latmax, lonmin, lonmax):
    """
    This function performs the weighted spatial mean to data.
    Accepts and returns xarray DataArray. Assumes to have spatial dimensions called 
    'lat' and 'lon'.
    """
    selection = var.sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax))
    spavg = selection.weighted(np.cos(selection.lat*np.pi/180)).mean(dim=['lat', 'lon'])
    return spavg

""" We need to implement Gruber Metrics for MHWs. Respectively, after labeling the events, 
    we need to calculate the following metrics, for each event:
    1. Duration
    2. Mean Intensity
    3. Magnitude, i.e. like mean intensity but wrt climatology
    4. Abruptness, i.e. the speed of the onset
    5. Heterogeneity, i.e. the variance of the intensity
    7. Cumulative Intensity. 
    Then, we need also to get global metrics, such as mean values of the above metrics, 
    and also:
    1. Recurrence, i.e. the average time between events, for each grid point.
"""
