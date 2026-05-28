#     Evaluation Framework for Enhancement of Spatio-Temporal Organization of Marine Heat Waves (EFESTOMHW)
##    Disclaimer
This documentation is still under construction, and has still to be changed a lot!!! New features will be described as the code evolve. For any doubt, advice, suggestion or improvement please contact us at 

    Vincenzo.DeToma@artov.ismar.cnr.it, Andrea.Pisano@artov.ismar.cnr.it

<img src="https://www.analisidellopera.it/wp-content/uploads/2019/10/Hokusai_La_grande_onda_di_Kanagawa.jpg" width=100%>

##    General overview

This repository contains the set of routines used to study the sensitivity of Hobday et al. (2016) algorithm for the detection and classification of Marine Heat Waves. Specifically, this code contains the new detection algorithm
developed within the European Space Agency (ESA) project for deteCtion and threAts of maRinE Heat waves (CAREHeat),
which was developed basing upon the results of the following sensitivity tests:\

-  Sensitivity to percentile chosen in threshold calculation ($90^{th}$, $95^{th}$, $99^{th}$ percentiles were considered);
-  Sensitivity to minimum duration (3 to 21 days for minimum duration necessary to identify MHW events);
-  Sensitivity to climatology reference (trend/detrend effect in the calculation of the climatology);
-  Sensitivity to climate modes removal (effect of removing periodic modes inside a given frequency band).\

Tests up to now indicated that the new method should be based on the $90^{th}$ percentile threshold and 5 days minimum duration as in Hobday's standard method. The novelty of the proposed method stands in the possibility to filter out a frequency band from the original time series prior to the detection. Tests indicate the optimal compromise between effectiveness of the approach and computational load to be a Butterworth filter of order 2. This gives the possibility to remove climate modes in order to distinguish MHW events due to known modes of climate variability (such as El-Nino Southern Oscillation) from the ones which are due to climate change-driven global warming. 

## Algorithm details

This detection algorithm is built around the method of Hobday et al. (2016) for the classification of Marine Heat Waves. First, in a module called, we find some preprocessing routines which can be applied on raw data before doing any detection.
-   a linear detrending function:
```python
def detrend_linear(var, dim):
    """
    This function remove the linear trend on variable var along 
    dimension dim, and returns the detrended field.
    """
    ds_trend = var.polyfit(dim=dim, deg=1, skipna=True)
    linear_fit_var = xr.polyval(var[dim], ds_trend.polyfit_coefficients)
    detr_var = var - linear_fit_var + ds_trend['polyfit_coefficients'].sel(degree=0)
    return detr_var.rename('detrended_'+var.name)

```
- a butterworth filter to filter out frequencies in a specified frequency band:
```python
from scipy.signal import butter, filtfilt

def butter_bandstop(lowcut, highcut, fs, order=5):
    """
    This filter cuts out freqencies inside lowcut, highcut. 
    See the docs in scipy to learn how it's implemented.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandstop_filter(data, lowcut, highcut, fs, order=1):
    """
    This function actually applies the filter to the data
    """
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def xr_butter_bandstop_filter(var, 
                              lowcut=(2*np.pi)/(7*365*24*60*60), 
                              highcut=(2*np.pi)/(2*365*24*60*60), 
                              fs=(2*np.pi)/(24*60*60),
                              order=1,
                              lin_detrend=False):
    """
    Here we wrap the application of the butterworth filter.
    When lin_detrend is set to True also perform a linear detrending
    before applying the banstop butterworth filter
    """
    # first detrend the data
    if lin_detrend==True:
        var_touse = detrend_linear(var=var, dim='time')
        namevar = var.name+'_ensofilt_det'
    else:
        var_touse = var
        namevar = var.name+'_ensofilt'
    # Apply the filter on linearly detrended data
    print("Mapping blocks for "+namevar)
    var_filt = xr.apply_ufunc(butter_bandstop_filter,
                              var_touse,
                              lowcut,
                              highcut,
                              fs,
                              order,
                              input_core_dims=[['time'], [''], [''], [''], ['']],
                              vectorize=True,
                              dask='allowed',
                              output_core_dims=[['time']],
                              dask_gufunc_kwargs={'allow_rechunk':True, 'traverse':True})
    return var_filt.transpose('time', 'lat', 'lon').rename(namevar)
```
- a function to implement SSA pixelwise, choosing a number of components M, reconstructing the signal only on frequencies below the one corresponding to M:
```python
def single_pixel_ssa(data, n_components):
    M = n_components
    sst=data
    sst_fast=sst.copy()
    sst_slow=sst.copy()
    if np.isnan(sst).any():
        periods = np.nan
        sst_fast = np.nan*np.ones(len(sst))
        sst_slow = np.nan*np.ones(len(sst))
        return sst, sst_fast, sst_slow, periods
    else:
        N=len(sst)
        N=2*np.fix(N/2).astype(int)
        K=int(N-M)
        U=sst
        Umean = np.mean(U)
        print(Umean)
        T = U - Umean
        print(M, N-M)
        X = np.zeros((M, N-M))
        print("T:", T.shape)
        # The maximum lag will be M days. Construct trajectory matrix
        for l in range(M):
            X[l,:] = T[l:N-M+l]
        # Calculate the covariance matrix
        C = (1/(K-1))*np.dot(X, X.T)
        print(C.shape)
        # Perform Eigenanalysis
        VAL,VEC = np.linalg.eig(C)
        VEC = VEC
        print(VEC.shape, VAL.shape)
        PC = np.dot(VEC.T, X)
        # Check variances and calculate the trace
        Fs=1
        ff=np.zeros(M)
        eigenv = np.zeros(M)
        print(PC.shape)
        for i in range(M):
            Y = np.fft.fft(PC[i,:])
            P2 = np.abs(Y/N)
            P1 = P2[:int(N/2)+1]
            P1[1:-1] = 2*P1[1:-1]
            f = np.abs(np.fft.fftfreq(len(PC[0,:])))
            m,idx = P1.max(0), P1.argmax(0)
            ff[i]=f[idx]
            eigenv[i]=VAL[i]
        frq,ind = np.sort(ff,axis=0), np.argsort(ff,axis=0)
        eige=eigenv[ind]
        print(frq[0])
        perds = 365
        fcat = 1/perds
        ks=np.argwhere(frq<=fcat) # Select eigenvectors with period longer than 365.
        RR_slow = np.zeros(N)
        for k in ks:
            RR = np.zeros(N)
            for i in range(1,N):
                xi = 0.
                if i<M:
                    j_range = range(1, i+1)
                    norm = i
                elif i>M and i<K:
                    j_range = range(1, M+1)
                    norm = M
                elif i>K and i<N:
                    j_range = range(i-N+M+1, M+1)
                    norm = N-i+1
                for j in j_range:
                    xi = xi + (PC[k,i-j]*VEC[j-1, k])
                RR[i-1] = xi/norm
            RR_slow = RR_slow + RR
        sst_slow[:] = RR_slow.T + Umean
        sst_fast[:] = T - RR_slow.T
        periods = max(1/frq[ks])
        return sst, sst_fast, sst_slow, periods

def xr_SSA_decompose(var, modes):
    inSST, fastSST, slowSST, periods = xr.apply_ufunc(single_pixel_ssa, var, kwargs={'n_components':modes},
                                                      input_core_dims=[['time']], 
                                                      vectorize=True, dask='parallelized', 
                                                      output_core_dims=[['time'], ['time'], ['time'], []], 
                                                      output_dtypes=[('float32', 'float32'), 
                                                                     ('float32', 'float32'), 
                                                                     ('float32', 'float32'), 
                                                                     ('float32', 'float32')])
    results = (inSST.transpose('time', 'lat', 'lon'), 
               fastSST.transpose('time', 'lat', 'lon').rename('fast_component'),
               slowSST.transpose('time', 'lat', 'lon').rename('slow_component'),
               periods.transpose('lat', 'lon').rename('periods'))
    return results
```
The actual detection is carried out by routines in `detection.py`:
```python
import xarray as xr
import numpy as np
import utils

def calc_climatology(sst, 
                     start, 
                     end, 
                     Window=5, 
                     Pctile=0.9, 
                     BinomialSmoothing=True, 
                     SmoothWind=15, 
                     RollWindow=31):
    """
    This function computes the climatology and threshold,
    given a window of days to enrich statistics, the percentile 
    for the threshold computation, and the additional possibility
    to carry out a binomial over a given period.
    """
    print('selecting period for the climatology...')
    climatology_selection = utils.select_slice_along_dim(var=sst, 
                                                         dim='time', 
                                                         start=start, 
                                                         end=end, 
                                                         method='values')
    print('building climatology rolling window...')
    climatology_rolling = utils.build_window_on_dim(var=climatology_selection, 
                                                    dim='time', 
                                                    window=Window)
    print('calculating percentile...')    
    climatology_percentile = utils.get_percentile(var=climatology_rolling, 
                                                  dims=["time", "window"],
                                                  q=Pctile,
                                                  rechunk_dim='time', 
                                                  groupby_dim='time.dayofyear')
    print('smoothing with 31 days...')
    ret_clim=climatology_rolling.groupby('time.dayofyear').mean(['time', 'window'])
    ret_perc=climatology_percentile
    if BinomialSmoothing==True:
        climatology_percentile_smoothed = utils \
            .binomial_smoothing(var=climatology_percentile,
            dim="dayofyear", 
            smooth_window=SmoothWind,
            roll_window=RollWindow,
            num_coeff=SmoothWind*2)
        ret_perc = climatology_percentile_smoothed

    return ret_clim.rename('sst climatology'), ret_perc.rename('sst threshold')

def detection_period_giventhresh(sst, 
                                 thresh,
                                 mindur=5, 
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
    anomalies = selection.groupby('time.dayofyear') - thresh
    print("getting MHW locations in time and space...")
    positive_bool = xr.where(anomalies>0., 1, 0)
    runn_bool = positive_bool.rolling({'time':mindur}, center=True).mean()
    mhw_center = xr.where(runn_bool==1, 1, 0)
    mhw_bool = mhw_center
    for m in range(int(mindur*0.5)):
        mhw_bool = mhw_bool + mhw_center.shift({'time':+(m+1)}) + \
            mhw_center.shift({'time':-(m+1)})
    print("Building the mhw_intensity field...")
    mhw_intensity = anomalies.where(mhw_bool>=1-1/mindur, 0.)
    mhw_intensity = mhw_intensity.where(~np.isnan(selection), np.nan)
    return mhw_intensity.rename('mhw_intensity')

def detection_complete(var, 
                       period_clim,
                       period_detect, 
                       pctile, 
                       window,
                       smoothing, 
                       smooth_window,
                       roll_window,
                       minduration):
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
                                    smooth_window,
                                    roll_window)
    mhw_intensity = detection_period_giventhresh(var,
                                                 thresh,
                                                 minduration,
                                                 DetStart,
                                                 DetEnd)
    return clim, thresh, mhw_intensity
```
Notice that here it is possible to specify a wide range of parameters, such as the period over which to compute the climatology, the window for it, the possibility to perform a binomial smoothing with a certain window, the percentile to choose for the threshold. Moreover, in the detection part it is possible to use a given climatology, or to calculate it alongside with the detection. This flexibility allows to vary independently every parameter, including the minimum duration for the identification of a MHW event. There are a some packages which are not directly useful in a single implementation of the algorithm. Ignore them, or feel free to take what you find useful. 

Please refer to `main.py` to have an idea on how to set up your own detection with our framework!!

### *Warning: Global detection may not work, so for now it's better to slice or subset your data. Updates of the code to handle larger data are planned for future versions. Have fun!*

## EXAMPLE OF REMOVING CLIMATE MODES IN MHW DETECTION: ENSO
![image](TPac_map_comparison_filtering.png)
![image](TPac_original_MHW.png)
![image](TPac_gt_9m_butt2_MHW.png)



## References
    [1] Hobday, A. J., Alexander, L. V., Perkins, S. E., Smale, D. A., Straub, S. C., Oliver, E. C., ... & Wernberg, T. (2016). A hierarchical approach to defining marine heatwaves. Progress in Oceanography, 141, 227-238.

<img src="Ismar_completo.jpg" width=45%> <img src="careheat_logo.jpg" width=30%>
