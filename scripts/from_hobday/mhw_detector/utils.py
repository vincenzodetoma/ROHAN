import numpy as np
import xarray as xr
from datetime import date

def overlap(ds_val, de_val, gds, gde, tolerance_days, max_separation_days=25):
    # Caso 1: eventi che si sovrappongono o sono separati da un piccolo gap
    latest_start = max(ds_val, gds)
    earliest_end = min(de_val, gde)
    delta = (earliest_end - latest_start).days
    if delta >= -tolerance_days:
        return True
    # Caso 2: la fine di uno è vicina all'inizio dell'altro (eventi consecutivi)
    separation1 = (gds - earliest_end).days  # inizio del secondo - fine del primo
    separation2 = (latest_start - gde).days  # inizio del primo - fine del secondo
    if 0 < separation1 <= max_separation_days:
        return True
    if 0 < separation2 <= max_separation_days:
        return True
    return False # permette un piccolo gap negativo

# --- Safe conversion from ordinal to datetime64[ns] ---
def safe_fromordinal_np(x):
    if np.isnan(x):
        return np.datetime64('NaT')
    return np.datetime64(date.fromordinal(int(x)))

def take_event_year_all(ds_mhw_grouped, year):

    # Convert ordinal values to datetime64 for each time dimension
    start_dates = xr.apply_ufunc(
        safe_fromordinal_np,
        ds_mhw_grouped['time_start'],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.datetime64]
    )
    peak_dates = xr.apply_ufunc(
        safe_fromordinal_np,
        ds_mhw_grouped['time_peak'],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.datetime64]
    )
    end_dates = xr.apply_ufunc(
        safe_fromordinal_np,
        ds_mhw_grouped['time_end'],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.datetime64]
    )
    
    # Filter events by the given year on start_dates (or could be peak or end)
    mask_year = start_dates.dt.year == year
    
    # Select only events that happen in that year
    start_sel = start_dates.where(mask_year, drop=True)
    peak_sel = peak_dates.where(mask_year, drop=True)
    end_sel = end_dates.where(mask_year, drop=True)
    
    # Get indices of the selected events (along the global_event dimension)
    # Assumes global_event is a dimension, otherwise adapt this part
    indices_start = start_sel['global_event'].values
    indices_end = end_sel['global_event'].values
    indices_peak = peak_sel['global_event'].values
    
    return start_sel, peak_sel, end_sel, indices_start, indices_peak, indices_end


def extract_field(da, field):
    arr = np.empty(da.shape, dtype=object)
    for idx, _ in np.ndenumerate(da.values):
        arr[idx] = da.values[idx][field]
    return xr.DataArray(arr, coords=da.coords, dims=da.dims)
    
    
def extract_field_with_time(da, field, time_coord=None):
    """
    Estrae un campo (array 1D, tipicamente temporale) da un DataArray di dizionari/lista di dizionari
    e restituisce un DataArray con dimensioni ('time', 'latitude', 'longitude').
    Se time_coord non è fornito, usa np.arange sulla lunghezza trovata.
    """

    # Trova la massima lunghezza della serie temporale tra tutti i punti
    max_time = 0
    for idx, _ in np.ndenumerate(da.values):
        v = da.values[idx]
        # Se è un dizionario e il campo è una serie
        if isinstance(v, dict) and field in v and hasattr(v[field], '__len__'):
            max_time = max(max_time, len(v[field]))
        # Se è una lista di dizionari, prendi il primo
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and field in v[0] and hasattr(v[0][field], '__len__'):
            max_time = max(max_time, len(v[0][field]))
    if field=='exceed_bool':
        arr = np.full((max_time,) + da.shape, np.nan, dtype=bool)
    else:
        arr = np.full((max_time,) + da.shape, np.nan, dtype=float)
    for idx, _ in np.ndenumerate(da.values):
        v = da.values[idx]
        if isinstance(v, dict) and field in v:
            serie = v[field]
            for t in range(len(serie)):
                arr[t][idx] = float(serie[t])
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and field in v[0]:
            serie = v[0][field]
            for t in range(len(serie)):
                arr[t][idx] = float(serie[t])
        # Se None o altro, lascia nan

    # Costruisci le coordinate
    coords = {}
    if time_coord is not None:
        coords['time'] = time_coord
    else:
        coords['time'] = np.arange(max_time)
    coords.update({k: v for k, v in da.coords.items()})
    dims = ('time',) + da.dims
    return xr.DataArray(arr, coords=coords, dims=dims)

def extract_event_field(da, field):

    # Trova la massima lunghezza delle liste di eventi
    max_events = 0
    for idx, _ in np.ndenumerate(da.values):
        lista = da.values[idx]
        # Se è una lista, prendi la lunghezza; se è un dict, considera 1; se None, 0
        if isinstance(lista, list):
            max_events = max(max_events, len(lista))
        elif isinstance(lista, dict):
            max_events = max(max_events, len(lista[field]))
        else:
            max_events = max(max_events, 0)

    arr = np.full((max_events,) + da.shape, np.nan, dtype=object)
    for idx, _ in np.ndenumerate(da.values):
        lista = da.values[idx]
        if isinstance(lista, list):
            for e, d in enumerate(lista):
                if isinstance(d, dict) and field in d:
                    arr[e][idx] = d
        elif isinstance(lista, dict):
            if field in lista:
                to_assign = lista[field]
                if to_assign:
                    for e, v in enumerate(to_assign):
                        if "date" in field:
                            v = np.datetime64(v)
                        arr[e][idx] = v
        # Se None, lascia nan

    coords = {'event': np.arange(max_events)}
    coords.update({k: v for k, v in da.coords.items()})
    dims = ('event',) + da.dims
    DA = xr.DataArray(np.array(arr), coords=coords, dims=dims)
    return DA

def sel_year(da, year):
    return da.where(da.dt.year == year)
