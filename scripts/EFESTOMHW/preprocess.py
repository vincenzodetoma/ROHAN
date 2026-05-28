import numpy as np
import xarray as xr

def detrend_linear(var, dim):
    """
    This function remove the linear trend on variable var along 
    dimension dim, and returns the detrended field.
    """
    ds_trend = var.polyfit(dim=dim, deg=1, skipna=True)
    linear_fit_var = xr.polyval(var[dim], ds_trend.polyfit_coefficients)
    detr_var = var - linear_fit_var + ds_trend['polyfit_coefficients'].sel(degree=0)
    return detr_var.rename('detrended_'+var.name)

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
    b = b.astype(data.dtype)
    a = a.astype(data.dtype)
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

def single_pixel_ssa(data, n_components, only_gravest=False):
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
        if only_gravest==True:
            ks=[0]
        else:
            ks=np.argwhere(frq<=fcat) # Select eigenvectors with period longer than 365.
        RR_slow = np.zeros(N)
        print(ks)
        for k in ks:
            print(k, frq[k])
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

def opt_single_pixel_ssa(data, n_components, only_gravest=False):
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
        #Umean = np.mean(U)
        #print(Umean)
        T = U #- Umean
        #print(M, N-M)
        X = np.zeros((M, N-M))
        #print("T:", T.shape)
        # The maximum lag will be M days. Construct trajectory matrix
        for l in range(M):
            X[l,:] = T[l:N-M+l]
        # Calculate the covariance matrix
        C = (1/(K-1))*np.dot(X, X.T)
        #print(C.shape)
        # Perform Eigenanalysis
        VAL,VEC = np.linalg.eig(C)
        VEC = VEC
        #print("VEC, VAL shapes:", VEC.shape, VAL.shape)
        PC = np.dot(VEC.T, X)
        # Check variances and calculate the trace
        Fs=1
        ff=np.zeros(M)
        eigenv = np.zeros(M)
        #print(PC.shape)
        Y = np.fft.fft(PC[:M, :], axis=1)
        P2 = np.abs(Y / N)
        P1 = P2[:, :int(N/2)+1]
        P1[:, 1:-1] = 2 * P1[:, 1:-1]
        f = np.abs(np.fft.fftfreq(len(PC[0, :])))
        max_indices = np.argmax(P1, axis=1)
        ff = f[max_indices]
        eigenv = VAL[:M]/VAL.sum()*100
        frq,ind = np.sort(ff,axis=0), np.argsort(ff,axis=0)
        frq, ind = frq[1:], ind[1:]
        eige=eigenv[ind]
        #print(frq[0])
        perds = 365
        fcat = 1/perds
        if only_gravest==True:
            ks=[0]
        else:
            ks=np.argwhere(frq<=fcat) # Select eigenvectors with period longer than 365.
        RR_slow = np.zeros(N)
        #print(ks)
        for k in ks:
            #print(k, frq[k])
            RR = np.zeros(N)
            for i in range(N):
                xi = 0.
                if i < M:
                    j_range = range(i + 1)
                    norm = i + 1
                elif M <= i < K:
                    j_range = range(M)
                    norm = M
                elif i >= K and i < N:
                    j_range = range(i - N + M + 1, M+1)
                    norm = N - i
                PC_slice = PC[k, i - np.array(list(j_range))]
                VEC_slice = VEC[np.array(list(j_range))-1, k]
                xi = np.dot(PC_slice, VEC_slice)
                RR[i] = xi / norm
            RR_slow += RR
        sst_slow[:] = RR_slow.T #+ Umean
        sst_fast[:] = T - RR_slow.T
        periods = max(1/frq[ks])
        return sst, sst_fast, sst_slow, periods


def xr_SSA_decompose(var, modes, only_gravest=False):
    inSST, fastSST, slowSST, periods = xr.apply_ufunc(opt_single_pixel_ssa, var, kwargs={'n_components':modes, 'only_gravest':only_gravest},
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
