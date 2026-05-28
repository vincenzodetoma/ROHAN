
# import some modules
import xarray as xr
import numpy as np
import utils
import dask.array as da
from scipy.ndimage import label


def calc_climatology(sst, 
                     start, 
                     end, 
                     Window=5, 
                     Pctile=0.9, 
                     Smoothing=True, 
                     SmoothWind=15):
    """
    This function computes the climatology and threshold,
    given a window of days to enrich statistics, the percentile 
    for the threshold computation, and the additional possibility
    to carry out a over a given period.
    """
    print('selecting period for the climatology...', start, end)
    climatology_selection = utils.select_slice_along_dim(var=sst, 
                                                         dim='time', 
                                                         start=start, 
                                                         end=end, 
                                                         method='values')
    print(climatology_selection.shape)
    print('building climatology rolling window...')
    climatology_rolling = utils.build_window_on_dim(var=climatology_selection, 
                                                    dim='time', 
                                                    window=Window)
    print('calculating percentile...')
    ret_clim=climatology_rolling.groupby('time.dayofyear').mean(["time", "window"])
    climatology_percentile = utils.get_percentile(var=climatology_rolling, 
                                                  dims=["time", "window"],
                                                  q=Pctile,
                                                  rechunk_dim='time', 
                                                  rechunk=True,
                                                  groupby_dim='time.dayofyear')
    print('smoothing with 31 days...')
    ret_perc=climatology_percentile
    if Smoothing==True:
        climatology_smoothed = utils \
            .smoothing(var=ret_clim,
            dim="dayofyear", 
            smooth_window=SmoothWind)
        climatology_percentile_smoothed = utils \
            .smoothing(var=ret_perc,
            dim="dayofyear", 
            smooth_window=SmoothWind)
        ret_perc = climatology_percentile_smoothed
        ret_clim = climatology_smoothed
    return ret_clim.rename('sst climatology'), ret_perc.rename('sst threshold')

def join_mhws(anomalies, mindur, gap):
    bool_anom = xr.where(anomalies>0, 1, 0)
    mylist = []
    for i in range(mindur, len(bool_anom)-mindur):
        if np.all(bool_anom[i-int(mindur*0.5):i+int(mindur*0.5)+1]==1) and anomalies[i]>0.:
            mylist.append(1)
        else:
            mylist.append(0)
    mylist = [0,]*mindur+mylist+[0,]*mindur
    ones = np.array(np.argwhere(np.array(mylist)>0)).reshape(-1)
    #print(len(ones))
    for on in ones:
        if on>=len(bool_anom)-2:
            print(on)
            mylist[on-1], mylist[on-2] = 1, 1
        elif int(on)==0:
            mylist[on+1], mylist[on+2] = 1, 1
        else:
            mylist[on-1], mylist[on+1], mylist[on-2], mylist[on+2]= 1, 1, 1, 1
    gaps = [0,]*(mindur)+[1 if mylist[i]==0 and (
        sum(mylist[i-mindur:i+mindur])>=mindur+gap
        ) else 0 for i in range(mindur, len(bool_anom)-mindur,1)]+[0,]*(mindur)
    bool_anom = np.array(mylist)+np.array(gaps)==1
    mhw_filled = xr.where(bool_anom, anomalies, -999)
    #for i in range(12095,12105):#len(bool_anom)-mindur):
    #    print(i,'\t', anomalies[i],'\t', bool_anom[i],'\t', mylist[i],'\t', gaps[i], '\t', mhw_filled[i], '\t', sum(bool_anom[i-mindur+k:i+k+1]>=mindur for k in range(1,mindur)))
    return bool_anom, mhw_filled
#
#def join_mhws(anomalies, mindur, gap):
#    """
#    This function extract and join marine heatwaves given duration and gaps.
#    """
#    bool_anom = xr.where(anomalies>0, 1., 0.)
#    print(bool_anom.shape)
#    mylist = []
#    for i in range(mindur, len(bool_anom)-mindur):
#        filcond=[]
#        for k in range(1,mindur):
#            filcond.append((bool_anom[i-mindur+k:i+k+1].sum())>=mindur)
#        if np.any(filcond) and bool_anom[i]==1:
#            mylist.append(1)
#        else:
#            mylist.append(0)
#    mylist = [0,]*mindur+mylist+[0,]*mindur
#    gaps = [0,]*(mindur//2)+[1 if mylist[i]==0 and (
#        sum(mylist[i-mindur//2:i+mindur//2+1])>=mindur-gap
#        ) else 0 for i in range(mindur//2, len(bool_anom)-mindur//2,1)]+[0,]*(mindur//2)
#    mhw_filled = xr.where(np.array(mylist)+np.array(gaps)==1, anomalies, -999)
#    print(mhw_filled.shape)
#    return mhw_filled

def xr_join_mhws(anomalies, mindur, gap):
    bool_anom, mhw_filled = xr.apply_ufunc(join_mhws, anomalies, mindur, gap, input_core_dims=[['time'],[],[],], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True}, output_dtypes=float, output_core_dims=[['time',],['time',]])
    return bool_anom, mhw_filled

def label_chunk(chunk, structure):
    return label(chunk, structure=structure)[0]

def get_event_index_dask(mhw_bool: xr.DataArray) -> xr.DataArray:
    """
    Efficient labeling of MHW events using Dask-parallelized chunk-wise labeling.
    Note: Events that span across chunk boundaries will be labeled separately.
    """
    structure = np.zeros((3, 1, 1), dtype=np.int8)
    structure[0, 0, 0] = 1  # Previous time step
    structure[1, 0, 0] = 1  # Current time step
    structure[2, 0, 0] = 1  # Next time step
    # Apply label in parallel across chunks
    labeled = da.map_blocks(
        label_chunk,
        mhw_bool.data,
        dtype=int,
        structure=structure
    )

    labeled_da = xr.DataArray(
        labeled,
        coords=mhw_bool.coords,
        dims=mhw_bool.dims,
        name="MHW_event_id"
    )

    return labeled_da

def detection_period_giventhresh(sst, 
                                 thresh,
                                 mindur=5, 
                                 gap=2,
                                 DetectionStart="2019-01-01",
                                 DetectionEnd="2019-12-31"):
    """
    This function carry out the detection over a specified period
    with a given climatology field
    """
    print("Selecting detection time...")
    selection = utils.select_slice_along_dim(var=sst, 
                                             dim='time', 
                                             start=DetectionStart, 
                                             end=DetectionEnd)
    print("Calculating anomalies...")
    #print(selection.groupby('time.dayofyear').mean('time').dayofyear.values, thresh.dayofyear.values)
    sel_group = selection.groupby('time.dayofyear').map(lambda x: x.compute())
    anomalies = sel_group - thresh
    print("getting MHW locations in time and space...")
    bool_anom, mhw_intensity = xr_join_mhws(anomalies, mindur, gap)
    #positive_bool = xr.where(anomalies>0., 1, 0)
    #runn_bool = positive_bool.rolling({'time':mindur}, center=True).mean()
    #mhw_center = xr.where(runn_bool==1, 1, 0)
    #mhw_bool = mhw_center
    #for m in range(int(mindur*0.5)):
    #    mhw_bool = mhw_bool + mhw_center.shift({'time':+(m+1)}) + \
    #        mhw_center.shift({'time':-(m+1)})
    mhw_labels=get_event_index_dask(mhw_bool=bool_anom)
    print("Building the mhw_intensity field...")
    #mhw_intensity = anomalies.where(mhw_bool>=1-1/mindur, 0.)
    mhw_intensity = mhw_intensity.where(~np.isnan(selection), np.nan)
    #return mhw_labels, mhw_intensity.rename('mhw_intensity')
    return mhw_labels.rename('mhw_labels'), mhw_intensity.rename('mhw_intensity')

def detection_complete(var, 
                       period_clim,
                       period_detect, 
                       pctile, 
                       window,
                       smoothing, 
                       smooth_window,
                       minduration,
                       gap=2):
    """
    This function wraps up the calculation of the climatology
    over its reference period, and then carry out the detection on
    input data. Default values are those in Hobday et al. (2016).
    """
    ClimStart = period_clim[0]
    ClimEnd = period_clim[1]
    DetStart = period_detect[0]
    DetEnd = period_detect[1]
    clim, thresh = calc_climatology(var,
                                    ClimStart,
                                    ClimEnd,
                                    window,
                                    pctile,
                                    smoothing,
                                    smooth_window)
    mhw_labels, mhw_intensity = detection_period_giventhresh(var,
                                                             thresh,
                                                             minduration,
                                                             gap,
                                                             DetStart,
                                                             DetEnd)
    return clim, thresh, mhw_intensity, mhw_labels
