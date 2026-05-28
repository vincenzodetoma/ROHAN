import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
try:
    from .utils import sel_year
except ImportError:
    from utils import sel_year
from tqdm import tqdm

import cartopy.crs as ccrs
import inspect

def get_projection(name, *args, **kwargs):
    """
    Returns a Cartopy projection class instance by name, with optional arguments.

    Parameters:
    - name (str): Name of the projection class (case-insensitive).
    - *args, **kwargs: Arguments to pass to the projection class constructor.

    Returns:
    - An instance of the requested cartopy.crs projection.
    """
    # Get all projection classes from ccrs
    projection_classes = {
        cls.__name__.lower(): cls
        for _, cls in inspect.getmembers(ccrs, inspect.isclass)
        if issubclass(cls, ccrs.Projection) and cls is not ccrs.Projection
    }

    # Normalize and look up
    name = name.lower()
    if name in projection_classes:
        return projection_classes[name](*args, **kwargs)
    else:
        available = ', '.join(sorted(projection_classes.keys()))
        raise ValueError(f"Unsupported projection name: '{name}'. Available options: {available}")


def plot_map(to_plot, cmap='jet', title="MHW Intensity Max in 2011", cbar_label=None, cwrap=3, idx=0, suptitle=None, vmin=0, vmax=1, levs=None, projection='platecarree', tcks=None, tcklabs=None):
    proj=get_projection(projection) if projection!='orthographic' else get_projection(projection, central_longitude=12, central_latitude=41)
    if vmin is None:
        vmin = to_plot.min().values
    if vmax is None:
        vmax = to_plot.max().values
    if levs==None:
        levs=21
    if len(to_plot.dims)==2:
        fig = plt.figure(figsize=(9,5))
        ax = fig.add_subplot(111, projection=proj)
        p = to_plot.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levs, extend='both', add_colorbar=True, cbar_kwargs={'label': cbar_label}, vmin=vmin, vmax=vmax)
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.text(0.5, -0.15, 'Longitude', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.text(-0.15, 0.5, 'Latitude', transform=ax.transAxes,
                ha='center', va='center', rotation='vertical', fontsize=12)
        ax.set_title(title)
        fig.tight_layout()
    elif len(to_plot.dims) == 3:
        if "time" in to_plot.dims:
            dimension = "time"
        elif "global_event" in to_plot.dims:
            dimension = "global_event"
        elif "exp" in to_plot.dims:
            dimension = "exp"
        p = to_plot.plot.imshow(
            col=dimension, 
            col_wrap=cwrap, 
            cmap=cmap,
            vmin=vmin, 
            vmax=vmax,
            levels=levs,
            transform=ccrs.PlateCarree(), 
            cbar_kwargs={'label': cbar_label, 'ticks': tcks}, 
            subplot_kws={'projection': proj}
        )
        if tcklabs is not None:
            p.cbar.set_ticklabels(tcklabs)
        ncols = cwrap
        nrows = int(np.ceil(len(to_plot[dimension]) / ncols))

        lon_min = float(to_plot['longitude'].min())
        lon_max = float(to_plot['longitude'].max())
        lat_min = float(to_plot['latitude'].min())
        lat_max = float(to_plot['latitude'].max())
        region_extent = [lon_min, lon_max, lat_min, lat_max]
        p.fig.suptitle(suptitle, y=1.05)

        for i, ax in enumerate(p.axes.flat):
            ax.coastlines()
            gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            if dimension=="time":
                if i < len(to_plot[dimension]):
                    ax.set_title(f"Event {idx}: \n {pd.to_datetime(to_plot[dimension].values[i]).strftime('%Y-%m-%d')}", fontsize=10)
            elif dimension=='global_event':
                if i < len(to_plot[dimension]):
                    ax.set_title(to_plot[dimension].values[i].replace(',', '\n'), fontsize=10)
            
            ax.set_extent(region_extent, crs=ccrs.PlateCarree()) if lon_max - lon_min > 358 and lat_max - lat_min > 179 else None
            
            # Calcola posizione del subplot (riga e colonna)
            row = i // ncols
            col = i % ncols

            # Attiva i labels solo ai bordi del faceted plot
            gl.top_labels = False
            gl.right_labels = False

            # Attiva label longitudine solo se è l'ultimo subplot della riga (bordo inferiore)
            if row == nrows - 1:
                gl.bottom_labels = True
            else:
                gl.bottom_labels = False

            # Attiva label latitudine solo se è il primo subplot della colonna (bordo sinistro)
            if col == 0:
                gl.left_labels = True
            else:
                gl.left_labels = False
    else:
        print("Plotting not supported for this DataArray shape.")
        p = None
    return p

def make_plots(ds_mhw_grouped, year, cwrap=10, suptitle=None, vmin=0, vmax=1):
    # Estrai le date globali
    ds_mhw_year_start_dates = ds_mhw_grouped['global_date_start']
    ds_mhw_year_peak_dates = ds_mhw_grouped['global_date_peak']
    ds_mhw_year_end_dates = ds_mhw_grouped['global_date_end']

    # Filtra per anno
    ds_mhw_year_start_dates = sel_year(ds_mhw_year_start_dates, year)
    ds_mhw_year_peak_dates = sel_year(ds_mhw_year_peak_dates, year)
    ds_mhw_year_end_dates = sel_year(ds_mhw_year_end_dates, year)

    # Filtra solo eventi coerenti: start <= peak <= end
    mask_coherent = (ds_mhw_year_start_dates.values <= ds_mhw_year_peak_dates.values) & \
                    (ds_mhw_year_peak_dates.values <= ds_mhw_year_end_dates.values)
    ds_mhw_year_start_dates = ds_mhw_year_start_dates[mask_coherent]
    ds_mhw_year_peak_dates = ds_mhw_year_peak_dates[mask_coherent]
    ds_mhw_year_end_dates = ds_mhw_year_end_dates[mask_coherent]

    # Ordina per data di inizio
    sort_idx = np.argsort(ds_mhw_year_start_dates.values)
    ds_mhw_year_start_dates = ds_mhw_year_start_dates[sort_idx]
    ds_mhw_year_peak_dates = ds_mhw_year_peak_dates[sort_idx]
    ds_mhw_year_end_dates = ds_mhw_year_end_dates[sort_idx]

    # Stampa info
    times_start = pd.to_datetime(ds_mhw_year_start_dates.values).strftime('%Y-%m-%d')
    times_end = pd.to_datetime(ds_mhw_year_end_dates.values).strftime('%Y-%m-%d')
    times_peak = pd.to_datetime(ds_mhw_year_peak_dates.values).strftime('%Y-%m-%d')
    print(f"Found {len(times_peak)} Events in {year}")
    print(f"Times start: {times_start}")
    print(f"Times peak: {times_peak}")
    print(f"Times end: {times_end}")

    # Ricava gli indici degli eventi selezionati rispetto all'array originale
    # Ricava gli indici degli eventi selezionati rispetto all'array originale
    idx = ds_mhw_year_start_dates['global_event'].values

    # Seleziona solo gli eventi coerenti nei dati
    to_plot2 = ds_mhw_grouped['intensity_max'].isel(global_event=idx)

    # Assegna le nuove etichette
    to_plot3 = to_plot2.assign_coords({
        'global_event': [
            f"{i}, {str(s)}, {str(p)}, {str(e)}" for (i, s, p, e) in (zip(to_plot2.global_event.values, times_start, times_peak, times_end))
        ]
    })

    plot_map(to_plot=to_plot3, cmap='copper', cwrap=cwrap, suptitle=suptitle, vmin=vmin, vmax=vmax)

def plot_ev_intensity_evolution(ds, sst, idx, cwrap=5, cmap='jet', suptitle=None, vmin=0, vmax=1):
    ev = ds.isel(global_event=idx)
    ev_int = ev['event_intensity'].assign_coords({'time': sst.time}).sel(time=slice(ev['global_date_start'].values, ev['global_date_end'].values))
    plot_map(ev_int, cmap=cmap, cwrap=cwrap, idx=idx, suptitle=suptitle, vmin=vmin, vmax=vmax)
    
def plot_events_active_in_year(ds, sst, year, cwrap=4, cmap='jet', suptitle=None, vmin=0, vmax=1):
    starts = pd.to_datetime(ds['global_date_start'].values)
    ends = pd.to_datetime(ds['global_date_end'].values)
    peaks = pd.to_datetime(ds['global_date_peak'].values)
    idxs = [i for i, (s, e, p) in enumerate(zip(starts, ends, peaks))
            if (pd.notnull(s) and s.year == year) or
               (pd.notnull(e) and e.year == year) or
               (pd.notnull(p) and p.year == year)]
    print(f"Found {len(idxs)} events active in {year} (start, end, peak)")
    for idx in tqdm(idxs, desc="Event plot {idx},  (start: {starts[idx]}, end: {ends[idx]}, peak: {peaks[idx]})"):
        print(f"Plotting event {idx} (start: {starts[idx]}, end: {ends[idx]}, peak: {peaks[idx]})")
        plot_ev_intensity_evolution(ds, sst, idx, cwrap=cwrap, cmap=cmap, suptitle=suptitle, vmin=vmin, vmax=vmax)