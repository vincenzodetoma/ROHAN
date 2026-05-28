import numpy as np
import scipy.ndimage
from tqdm import tqdm
import xarray as xr

def label_events(binary_array, min_duration=5):
    labeled, _ = scipy.ndimage.label(binary_array)
    
    def filter_short_events(labeled_events, min_duration):
        filtered = np.zeros_like(labeled_events)
        labels = np.unique(labeled_events)
        labels = labels[labels != 0]  # Exclude background
        for label in labels:
            event_mask = labeled_events == label
            if np.sum(event_mask) >= min_duration:
                filtered[event_mask] = label
        return filtered
    
    return filter_short_events(labeled, min_duration=5)

def filter_by_duration_and_gap(labeled_events, max_gap=2):
    """
    Joins events in labeled_events that are separated by a gap of less than or equal to max_gap.
    Assumes labeled_events already contains only events longer than the minimum duration.
    """
    arr = labeled_events.flatten()
    result = np.zeros_like(arr)
    current_label = 1
    i = 0
    n = len(arr)
    while i < n:
        if arr[i] != 0:
            # Start of an event
            start = i
            label = arr[i]
            while i < n and arr[i] == label:
                i += 1
            end = i
            # Check for short gap after this event
            gap_start = end
            gap_end = gap_start
            while gap_end < n and arr[gap_end] == 0 and (gap_end - gap_start) < max_gap:
                gap_end += 1
            # If next event is within max_gap, join them
            if gap_end < n and arr[gap_end] != 0:
                # Merge the gap and next event
                arr[start:gap_end] = label
                i = gap_end
            else:
                # Keep the event as is
                result[start:end] = current_label
                current_label += 1
                i = end
        else:
            i += 1
    return result.reshape(labeled_events.shape)

def select_slice_along_dim(var, dim, start, end, method='values'):
    if method == 'values':
        return var.sel({dim: slice(start, end)})
    elif method == 'indices':
        return var.isel({dim: slice(start, end)})
    else:
        raise Exception("Invalid method selected.\n Valid methods are 'values' or 'indices'.")

def build_window_on_dim(var, dim, window):
    return var.pad(pad_width={dim: window}, mode='wrap') \
              .rolling({dim: 2*window + 1}, center=True) \
              .construct('window') \
              .isel({dim: slice(window, -window)})

def get_percentile(var, dims, q, rechunk_dim, groupby_dim, rechunk=False):
    if rechunk:
        return var.chunk({rechunk_dim: -1}) \
                  .groupby(groupby_dim) \
                  .quantile(q, dim=dims)
    else:
        return var.groupby(groupby_dim) \
                  .quantile(q, dim=dims)

def smoothing(var, dim, smooth_window):
    return (var.pad(pad_width={dim: smooth_window}, mode="wrap")
              .rolling({dim: 2*smooth_window + 1}, center=True)
              .construct("window")
              .isel({dim: slice(smooth_window, -smooth_window)})
              .mean('window'))

def calc_climatology(sst, start, end, Window=5, Pctile=0.9, Smoothing=True, SmoothWind=15):
    print('Selecting period for climatology:', start, end)
    climatology_selection = select_slice_along_dim(sst, 'time', start, end, method='values')
    print(f'Building climatology rolling window on {Window} days')
    climatology_rolling = build_window_on_dim(climatology_selection, 'time', Window)
    ret_clim = climatology_rolling.groupby('time.dayofyear').mean(["time", "window"])
    climatology_percentile = get_percentile(climatology_rolling, dims=["time", "window"], q=Pctile,
                                            rechunk_dim='time', rechunk=True,
                                            groupby_dim='time.dayofyear')
    if Smoothing:
        print(f'Smoothing with window size {SmoothWind} days...')
        ret_clim = smoothing(ret_clim, dim='dayofyear', smooth_window=SmoothWind)
        climatology_percentile = smoothing(climatology_percentile, dim='dayofyear', smooth_window=SmoothWind)
        # Repeat the climatology and threshold for each year in the input time series
        time = sst['time']
        dayofyear = time.dt.dayofyear
        ret_clim = ret_clim.sel(dayofyear=dayofyear)
        ret_clim = ret_clim.assign_coords(time=time).drop_vars('dayofyear')
        climatology_percentile = climatology_percentile.sel(dayofyear=dayofyear)
        climatology_percentile = climatology_percentile.assign_coords(time=time).drop_vars('dayofyear')
    return ret_clim.rename(sst.name+'_climatology'), climatology_percentile.rename(sst.name+'_threshold')


