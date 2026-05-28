import xarray as xr
import numpy as np
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from contextlib import nullcontext
import warnings
from .utils import label_events, filter_by_duration_and_gap, calc_climatology

@xr.register_dataarray_accessor("mhw")
class MHWAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def detect(
        self,
        threshold_percentile=90,
        min_duration=5,
        max_gap=2,
        climatology=None,
        clim_start=None,
        clim_end=None,
        verbose=True,
        window=5,
        smoothing=True,
        smoothwind=15,
        return_metrics=False,
        return_climatology=False
    ):
        da = self._obj
    
        def vprint(msg):
            if verbose:
                tqdm.write(msg)
    
        vprint("\U0001F4CA Starting MHW detection")
    
        if clim_start and clim_end:
            vprint(f"\U0001F552 Selecting climatology period: {clim_start} to {clim_end}")
            # Use the new calc_climatology function
            ret_clim, clim_thresh = calc_climatology(
                da,
                start=clim_start,
                end=clim_end,
                Window=window,
                Pctile=threshold_percentile / 100.0,
                Smoothing=smoothing,
                SmoothWind=smoothwind
            )
        elif climatology is None:
            vprint("\U0001F4C8 Calculating climatology and threshold")
            ret_clim, clim_thresh = calc_climatology(
                da,
                start=str(da.time.min().values),
                end=str(da.time.max().values),
                Window=window,
                Pctile=threshold_percentile / 100.0,
                Smoothing=smoothing,
                SmoothWind=smoothwind
            )
        else:
            vprint("\U0001F4C1 Using provided climatology")
            ret_clim = climatology["mean"]
            clim_thresh = climatology["threshold"]
    
        vprint("\U0001F321\ufe0f Identifying threshold exceedances")
        above_thresh = da > clim_thresh

        vprint("\U0001F50E Labeling events with Dask")
        events = above_thresh.astype(int)
        print(events.dims)  # Should show ("time", "latitude", "longitude")
        with ProgressBar() if verbose else nullcontext():
            event_labels = xr.apply_ufunc(
                label_events,
                events,
                kwargs={'min_duration':min_duration},
                input_core_dims=[["time",]],
                output_core_dims=[["time",]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[int],
                dask_gufunc_kwargs={"allow_rechunk": True},
            ).compute()
        
        print("Writing labels:")
        event_labels.rename('mhw_labels').to_netcdf('output/labels.nc')
        vprint("\U0001F9F9 Filtering by duration and filling gaps")
        event_labels_filt = xr.apply_ufunc(filter_by_duration_and_gap, event_labels, kwargs={'max_gap': max_gap},
                                           input_core_dims=[["time"]],
                                           output_core_dims=[["time"]],
                                           vectorize=True,
                                           dask="parallelized",
                                           output_dtypes=[int],
                                           dask_gufunc_kwargs={"allow_rechunk": True}).compute()
        print("Writing filtered labels:")
        event_labels_filt.rename('mhw_labels_join').to_netcdf('output/joined_labels.nc')
        vprint("\U0001F4C8 Creating MHW mask")
        mhw_mask = xr.where(event_labels_filt > 0, True, False).rename("mhw_mask")
        masked_intensities = xr.where(mhw_mask, da - clim_thresh, np.nan)
        masked_intensities = masked_intensities.where(~np.isnan(above_thresh)).rename('mhw_intensity')
        print(f"Counting events {event_labels_filt.max(dim='time')}")
        if return_metrics:
            vprint("\U0001F4CA Calculating Gruber metrics")
            metrics = self.gruber_metrics(mhw_mask=mhw_mask, ev_mask=event_labels_filt, clim_thresh=clim_thresh, ret_clim=ret_clim, verbose=verbose)
            vprint("\u2705 MHW detection and metrics complete")
            return (masked_intensities, metrics, clim_thresh, ret_clim) if return_climatology else (masked_intensities, metrics)
        else:
            vprint("\u2705 MHW detection complete")
            return (masked_intensities, clim_thresh, ret_clim) if return_climatology else (masked_intensities)

    def gruber_metrics(self, mhw_mask=None, ev_mask=None, clim_thresh=None, ret_clim=None, verbose=True):
        def vprint(msg):
            if verbose:
                tqdm.write(msg)

        da = self._obj

        if mhw_mask is None:
            mhw_mask = self.detect(verbose=verbose)

        if ev_mask is None:
            raise ValueError("ev_range must be provided to calculate metrics for each event.")

        metrics = []
        count = ev_mask.max().values
        for ev in range(1, count):
            event_mask = mhw_mask.where(ev_mask == ev, drop=True)
            event_da = da.where(ev_mask == ev, drop=True)
            start_date = event_mask.time[0].values
            end_date = event_mask.time[-1].values
            print(f"Processing event {ev} from {start_date} to {end_date}")
            # Intensity: mean anomaly during the event
            intensity = event_da.sel(time=slice(start_date, end_date)).mean(dim="time")
            
            intensity_anom = (event_da - ret_clim).sel(time=slice(start_date, end_date))
            intensity_anom_mean = intensity_anom.mean(dim='time')
            
            catdiff = (clim_thresh - ret_clim).sel(time=slice(start_date, end_date))
            twice_catdiff = 2 * catdiff
            triple_catdiff = 3 * catdiff
            fortuple_catdiff = 4 * catdiff
            
            categories = xr.where(intensity_anom > catdiff, 1, 0)
            categories = categories.where(intensity_anom < twice_catdiff, 0)
            categories = categories.where(intensity_anom < triple_catdiff, 0)
            categories = categories.where(intensity_anom < fortuple_catdiff, 0)
            categories = categories.max(dim='time')

            # Duration: number of time steps in the event
            duration = event_mask.sum(dim="time")

            metrics.append(
                xr.Dataset(
                    {
                        "intensity": intensity,
                        "mean_intensity_anomaly": intensity_anom_mean,
                        "category": categories,
                        "duration": duration,
                        "event_id": ev,
                        "start_date": start_date,
                        "end_date": end_date
                    }, 
                    attrs={
                        "category": "integer variable indicating the maximum category of the event based on intensity anomaly",
                        "category_description": "0: No MHW, 1: Weak, 2: Moderate, 3: Strong, 4: Extreme"
                    }
                )
            )

        vprint("\u2705 Gruber metrics complete")
        if metrics:
            return xr.concat(metrics, dim="event")
        else:
            return xr.Dataset()

