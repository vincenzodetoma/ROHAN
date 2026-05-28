from datetime import date
import xarray as xr
import numpy as np
import pandas as pd
try:
    from .utils import overlap
except ImportError:
    from utils import overlap

def cluster_events_spacetime(ds_mhw, 
                            date_start_name='date_start', 
                            date_end_name='date_end', 
                            date_peak_name='date_peak',
                            eps_space=1.0, 
                            eps_time=5, 
                            min_samples=3,
                            clusterator="DBSCAN"):
    """
    Raggruppa eventi locali in eventi globali spazio-temporali coerenti usando DBSCAN.
    Restituisce un nuovo dataset con dimensione 'global_event' = cluster_id.
    """

    lat = ds_mhw.latitude.values
    lon = ds_mhw.longitude.values
    n_events, nlat, nlon = ds_mhw[date_start_name].shape

    # Estrai tutti gli eventi locali validi come lista di feature
    records = []
    idx_map = []  # (e, i, j) per ricostruire
    for e in range(n_events):
        for i in range(nlat):
            for j in range(nlon):
                ds_val = ds_mhw[date_start_name].values[e, i, j]
                de_val = ds_mhw[date_end_name].values[e, i, j]
                dp_val = ds_mhw[date_peak_name].values[e, i, j]
                if pd.isnull(ds_val) or pd.isnull(de_val) or pd.isnull(dp_val):
                    continue
                # Converti date in ordinal (giorni)
                ds_ord = pd.to_datetime(ds_val).toordinal()
                de_ord = pd.to_datetime(de_val).toordinal()
                dp_ord = pd.to_datetime(dp_val).toordinal()
                records.append([lat[i], lon[j], ds_ord, de_ord, dp_ord])
                idx_map.append((e, i, j))

    if not records:
        raise ValueError("Nessun evento valido trovato per il clustering.")

    X = np.array(records)
    # Normalizza le feature: spazio in gradi, tempo in giorni
    X_scaled = np.copy(X)

    if clusterator=="HDBSCAN":
        from hdbscan import HDBSCAN
        clustering = HDBSCAN(min_cluster_size=min_samples).fit(X_scaled)
    else:
        from sklearn.cluster import DBSCAN
        X_scaled[:,0] /= eps_space   # lat
        X_scaled[:,1] /= eps_space   # lon
        X_scaled[:,2] /= eps_time    # date_start
        X_scaled[:,3] /= eps_time    # date_end
        X_scaled[:,4] /= eps_time    # date_peak
        # Clustering DBSCAN
        clustering = DBSCAN(eps=1.0, min_samples=min_samples).fit(X_scaled)
    labels = clustering.labels_  # -1 = outlier

    # Filtra outlier
    valid = labels >= 0
    X = X[valid]
    idx_map = np.array(idx_map)[valid]
    labels = labels[valid]
    n_clusters = labels.max() + 1

    # Prepara array per ogni variabile
    data_vars = {}
    for var in ds_mhw.data_vars:
        arr = np.full((n_clusters, nlat, nlon), np.nan)
        for label, (e, i, j) in zip(labels, idx_map):
            arr[label, i, j] = ds_mhw[var].values[e, i, j]
        data_vars[var] = (['global_event', 'latitude', 'longitude'], arr)

    # Date globali (media delle date dei membri del cluster)
    global_date_start = np.full(n_clusters, np.nan)
    global_date_end = np.full(n_clusters, np.nan)
    global_date_peak = np.full(n_clusters, np.nan)
    for k in range(n_clusters):
        ds_vals = X[labels == k, 2]
        de_vals = X[labels == k, 3]
        dp_vals = X[labels == k, 4]
        if len(ds_vals) > 0:
            global_date_start[k] = np.median(ds_vals)
            global_date_end[k] = np.median(de_vals)
            global_date_peak[k] = np.median(dp_vals)
    # Converti in datetime64
    from datetime import date
    global_date_start = np.array([np.datetime64(date.fromordinal(int(x))) if not np.isnan(x) else np.datetime64('NaT') for x in global_date_start])
    global_date_end = np.array([np.datetime64(date.fromordinal(int(x))) if not np.isnan(x) else np.datetime64('NaT') for x in global_date_end])
    global_date_peak = np.array([np.datetime64(date.fromordinal(int(x))) if not np.isnan(x) else np.datetime64('NaT') for x in global_date_peak])
    
    ds_grouped = xr.Dataset(
        {**data_vars,
         'global_date_start': (['global_event'], global_date_start),
         'global_date_end': (['global_event'], global_date_end),
         'global_date_peak': (['global_event'], global_date_peak)
        },
        coords={
            'global_event': np.arange(n_clusters),
            'latitude': lat,
            'longitude': lon,
        }
    )
    return ds_grouped

def group_events_spacetime(
    ds_mhw,
    date_start_name='date_start',
    date_end_name='date_end',
    date_peak_name='date_peak',
    tolerance_days=1,
    min_contiguous_pixels=3
):
    import numpy as np
    import pandas as pd
    from scipy.ndimage import label

    date_start = ds_mhw[date_start_name].values
    date_end = ds_mhw[date_end_name].values
    date_peak = ds_mhw[date_peak_name].values
    lat = ds_mhw.latitude.values
    lon = ds_mhw.longitude.values

    n_events, nlat, nlon = date_start.shape
    global_events = []
    event_map = {}

    # Per ogni evento locale
    for e in range(n_events):
        for i in range(nlat):
            for j in range(nlon):
                ds_val = date_start[e, i, j]
                de_val = date_end[e, i, j]
                if pd.isnull(ds_val) or pd.isnull(de_val):
                    continue
                ds_val = pd.to_datetime(ds_val)
                de_val = pd.to_datetime(de_val)
                found = False
                for k, (gds, gde) in enumerate(global_events):
                    if overlap(ds_val, de_val, gds, gde, tolerance_days):
                        event_map.setdefault(k, []).append((e, i, j))
                        found = True
                        break
                if not found:
                    global_events.append((ds_val, de_val))
                    event_map[len(global_events)-1] = [(e, i, j)]

    # Per ogni gruppo temporale, trova i cluster spaziali
    final_events = []
    final_event_map = {}
    final_event_peaks = []
    for k, indices in event_map.items():
        mask = np.zeros((nlat, nlon), dtype=bool)
        for _, i, j in indices:
            mask[i, j] = True
        labeled, num = label(mask)
        for group in range(1, num+1):
            group_indices = [(e, i, j) for (e, i, j) in indices if labeled[i, j] == group]
            if len(group_indices) >= min_contiguous_pixels:
                final_events.append(global_events[k])
                final_event_map[len(final_events)-1] = group_indices
                # Calcola la media delle date di picco per il cluster
                peaks = []
                for e, i, j in group_indices:
                    peak_val = date_peak[e, i, j]
                    if not pd.isnull(peak_val):
                        peaks.append(pd.to_datetime(peak_val).toordinal())
                if peaks:
                    mean_peak = np.mean(peaks)
                    final_event_peaks.append(mean_peak)
                else:
                    final_event_peaks.append(np.nan)

    n_final = len(final_events)
    data_vars = {}
    for var in ds_mhw.data_vars:
        arr = np.full((n_final, nlat, nlon), np.nan)
        for k, indices in final_event_map.items():
            for e, i, j in indices:
                arr[k, i, j] = ds_mhw[var].values[e, i, j]
        data_vars[var] = (['global_event', 'latitude', 'longitude'], arr)

    global_date_start = np.array([
        np.datetime64('NaT') if pd.isnull(x) else np.datetime64(date.fromordinal(int(x)))
        for x in [ev[0].toordinal() if hasattr(ev[0], 'toordinal') else np.nan for ev in final_events]
    ], dtype='datetime64[ns]')

    global_date_end = np.array([
        np.datetime64('NaT') if pd.isnull(x) else np.datetime64(date.fromordinal(int(x)))
        for x in [ev[1].toordinal() if hasattr(ev[1], 'toordinal') else np.nan for ev in final_events]
    ], dtype='datetime64[ns]')

    global_date_peak = np.array([
        np.datetime64('NaT') if np.isnan(x) else np.datetime64(date.fromordinal(int(x)))
        for x in final_event_peaks
    ], dtype='datetime64[ns]')

    ds_grouped = xr.Dataset(
        {**data_vars,
         'global_date_start': (['global_event'], global_date_start),
         'global_date_end': (['global_event'], global_date_end),
         'global_date_peak': (['global_event'], global_date_peak)
        },
        coords={
            'global_event': np.arange(n_final),
            'latitude': lat,
            'longitude': lon,
        }
    )
    return ds_grouped

def clustering(sst, ds_mhw, intensity, tol_days, cont_pixels, output_path="mhw_Atlas.nc", method='both', eps_space=30, eps_time=20, min_samples=2, clusterator="DBSCAN"):
    results = {}
    if method=='by-hand':
        ds_mhw_grouped = group_events_spacetime(ds_mhw, tolerance_days=tol_days, min_contiguous_pixels=cont_pixels).assign_coords({'latitude': sst.latitude, 'longitude': sst.longitude})
        ds_mhw_grouped.to_netcdf(output_path.replace('Atlas', 'GroupedEvents'))
        results[method] = ds_mhw_grouped
    elif method=='clustering': 
        ds_mhw_grouped_clusters = cluster_events_spacetime(ds_mhw, eps_space=eps_space, eps_time=eps_time, min_samples=min_samples, clusterator=clusterator)
        ds_mhw_grouped_clusters.to_netcdf(output_path.replace('Atlas', f'{clusterator}_cluster'))
        results[method] = ds_mhw_grouped_clusters
    elif method=='clustering_wtime':
        ds_mhw_grouped_clusters_wtime = cluster_events_with_timeseries(ds_mhw=ds_mhw, eps_space=eps_space, eps_time=eps_time, min_samples=min_samples, intensity=intensity, time=sst.time, clusterator=clusterator)
        ds_mhw_grouped_clusters_wtime.to_netcdf(output_path.replace('Atlas', f'{clusterator}_wtime_cluster'))
        results[method] = ds_mhw_grouped_clusters_wtime
    else:
        ds_mhw_grouped = group_events_spacetime(ds_mhw, tolerance_days=tol_days, min_contiguous_pixels=cont_pixels).assign_coords({'latitude': sst.latitude, 'longitude': sst.longitude})
        ds_mhw_grouped_clusters = cluster_events_spacetime(ds_mhw, eps_space=eps_space, eps_time=eps_time, min_samples=min_samples)
        ds_mhw_grouped_clusters.to_netcdf(output_path.replace('Atlas', f'{clusterator}_cluster'))
        ds_mhw_grouped_clusters_wtime = cluster_events_with_timeseries(ds_mhw=ds_mhw, eps_space=eps_space, eps_time=eps_time, min_samples=min_samples, intensity=intensity, time=sst.time)
        ds_mhw_grouped_clusters_wtime.to_netcdf(output_path.replace('Atlas', f'{clusterator}_wtime_cluster'))
        results['by-hand'] = ds_mhw_grouped
        results[f'{clusterator}_clustering'] = ds_mhw_grouped_clusters
        results[f'{clusterator}_clustering_wtime'] = ds_mhw_grouped_clusters_wtime
        return results
    
def cluster_events_with_timeseries(
    ds_mhw,
    intensity,
    time,
    date_start_name='date_start',
    date_end_name='date_end',
    date_peak_name='date_peak',
    eps_space=1.0,
    eps_time=5,
    min_samples=2,
    clusterator="DBSCAN"
):
    # 1. Usa la funzione già presente per il clustering
    ds_clusters = cluster_events_spacetime(
        ds_mhw,
        date_start_name=date_start_name,
        date_end_name=date_end_name,
        date_peak_name=date_peak_name,
        eps_space=eps_space,
        eps_time=eps_time,
        min_samples=min_samples, 
        clusterator=clusterator
    )

    # 2. Ricava le info base
    lat = ds_clusters.latitude.values
    lon = ds_clusters.longitude.values
    n_clusters = ds_clusters.dims['global_event']
    n_time = len(time)
    nlat = len(lat)
    nlon = len(lon)

    # 3. Costruisci la maschera (se non già presente)
    if 'mask' in ds_clusters:
        mask = ds_clusters['mask'].values
    else:
        # Costruisci la maschera: True dove almeno una variabile non è nan
        mask = ~np.isnan(ds_clusters[date_start_name].values)

    # 4. Serie temporale di intensità per ogni evento/pixel
    arr = np.full((n_clusters, n_time, nlat, nlon), np.nan)
    for k in range(n_clusters):
        t_start = ds_clusters['global_date_start'].values[k]
        t_end = ds_clusters['global_date_end'].values[k]
        mask_k = mask[k]
        t_mask = (time >= t_start) & (time <= t_end)
        for i in range(nlat):
            for j in range(nlon):
                if mask_k[i, j]:
                    arr[k, t_mask, i, j] = intensity[t_mask, i, j]

    # 5. Aggiungi la serie temporale di intensità al dataset dei cluster
    ds_clusters['event_intensity'] = (['global_event', 'time', 'latitude', 'longitude'], arr)
    ds_clusters.assign_coords({'time':time})
    return ds_clusters