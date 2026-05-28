#%%
import xarray as xr
import numpy as np
import dask
import fire
import warnings
warnings.filterwarnings("ignore")
import glob
import sys
import os
try:
    from .plotting import *
    from .percentage import Percentage, Mean, Max, Std
except ImportError:
    from plotting import *
    from percentage import Percentage, Mean, Max, Std
import seaborn as sns

def concatenate_datasets(datasets):
    combined = xr.concat(datasets, dim='event')
    combined = combined.assign_coords(
        depth=('depth', np.unique(combined.depth.values)),
        exp=('exp', np.unique(combined.exp.values))
    )
    return combined

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xarray as xr
import numpy as np
plt.rcParams['font.size'] = 20

def pad_dataset_to_length(ds, target_length, dim="event"):
    """
    Pad all DataArrays in a Dataset to a target length along `dim`.
    Only variables that include `dim` are padded.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    target_length : int
        Desired length along `dim`.
    dim : str
        Dimension to pad.

    Returns
    -------
    xr.Dataset
        Dataset with padded variables.
    """
    
    def pad_da(da):
        """Pad a single DataArray like your original function."""
        current_length = da.sizes[dim]
        if current_length == target_length:
            return da
        
        pad_shape = list(da.shape)
        axis = da.get_axis_num(dim)
        pad_shape[axis] = target_length - current_length

        # create pad block (NaNs)
        template = da.isel({dim: slice(0,1)})
        pad = xr.full_like(template, np.nan).isel({dim: 0}).expand_dims({dim: pad_shape[axis]})

        return xr.concat([da, pad], dim=dim, coords='minimal', compat='override')

    # Build new dataset with padded DataArrays
    new_vars = {}
    for var_name, da in ds.data_vars.items():
        if dim in da.dims:
            new_vars[var_name] = pad_da(da)
        else:
            new_vars[var_name] = da  # unchanged

    return xr.Dataset(new_vars, coords=ds.coords, attrs=ds.attrs)


def pad_to_length(da, target_length, dim='event'):
    """Pad a DataArray to target length along a dimension."""
    current_length = da.sizes[dim]
    if current_length == target_length:
        return da
    pad_shape = list(da.shape)
    pad_shape[da.get_axis_num(dim)] = target_length - current_length
    pad = xr.full_like(da.isel({dim: slice(0,1)}), np.nan).isel({dim: 0}).expand_dims({dim: pad_shape[da.get_axis_num(dim)]})
    da_padded = xr.concat([da, pad], dim=dim, coords='minimal', compat='override')
    return da_padded

def _parse_areas_from_script(script_path):
    """
    Parse the `areas` definition from the given bash script.
    Returns a list of dicts: {'name', 'lat_min','lat_max','lon_min','lon_max'}
    'None' strings are converted to None.
    """
    areas = []
    if not os.path.exists(script_path):
        return areas
    text = open(script_path, 'r').read().splitlines()
    in_areas = False
    for line in text:
        line = line.strip()
        if line.startswith('areas='):
            in_areas = True
            continue
        if in_areas:
            if line.startswith(')'):
                break
            # strip comments and leading/trailing quotes/spaces
            if line.startswith('#'):
                continue
            # remove surrounding quotes/backticks if present
            line_clean = line.strip().strip('"').strip("'")
            if not line_clean:
                continue
            # split by whitespace (first five items are name lat_min lat_max lon_min lon_max)
            parts = line_clean.split()
            if len(parts) < 5:
                continue
            name = parts[0]
            lat_min_s, lat_max_s, lon_min_s, lon_max_s = parts[1:5]
            def conv(v):
                if v in ('None', 'none', 'NULL', 'null', ''):
                    return None
                try:
                    return float(v)
                except Exception:
                    return None
            areas.append({
                'name': name,
                'lat_min': conv(lat_min_s),
                'lat_max': conv(lat_max_s),
                'lon_min': conv(lon_min_s),
                'lon_max': conv(lon_max_s)
            })
    return areas

def _make_tiles_for_area(area, tile_size_deg):
    """
    Given an area dict with lat/lon bounds (None means global extent), return list of tile bboxes:
    [(lat0, lat1, lon0, lon1), ...]
    tile_size_deg: scalar for both lat and lon (degrees)
    """
    lat_min = -90.0 if area.get('lat_min') is None else float(area['lat_min'])
    lat_max = 90.0 if area.get('lat_max') is None else float(area['lat_max'])
    lon_min = -180.0 if area.get('lon_min') is None else float(area['lon_min'])
    lon_max = 180.0 if area.get('lon_max') is None else float(area['lon_max'])
    tiles = []
    # handle wrap-around longitudes if lon_min > lon_max (not expected in script but be robust)
    if lon_min <= lon_max:
        lon_ranges = [(lon_min, lon_max)]
    else:
        # split into two ranges
        lon_ranges = [(lon_min, 180.0), (-180.0, lon_max)]
    lat_edges = np.arange(lat_min, lat_max, tile_size_deg)
    for rlon in lon_ranges:
        lon_edges = np.arange(rlon[0], rlon[1], tile_size_deg)
        for la in lat_edges:
            la1 = min(la + tile_size_deg, lat_max)
            for lo in lon_edges:
                lo1 = min(lo + tile_size_deg, rlon[1])
                tiles.append((la, la1, lo, lo1))
    return tiles

def main(area='Global', varname='intensity_var', filename_sst='mhw_climatology.nc', varname_sst='mhw_intensity', expnames=['cglo', 'glor', 'oras'], threshold=50, diagnostic='Mean', plot=False, use_tiles=False, distribution_cache=None, use_dask=True, chunks='auto'):
    """
    Load Global datasets (all depths) and subset by area bounds defined in test_mhw_global.sh.
    Separate into shallow (<= threshold) and deep (> threshold).
    Return tuple: (shallow_ds, deep_ds)
    """
    # Parse areas from script to get bounds for the target area
    script_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'test_mhw_global.sh'))
    areas = _parse_areas_from_script(script_path)
    
    # Find the target area bounds
    area_bounds = None
    for a in areas:
        if a['name'] == area:
            area_bounds = a
            break
    if area_bounds is None:
        # If area not found, assume Global (no subsetting)
        print(f"Area '{area}' not found in script. Using Global (no subsetting).")
        area_bounds = {'name': 'Global', 'lat_min': None, 'lat_max': None, 'lon_min': None, 'lon_max': None}
    
    lat_min = -90.0 if area_bounds['lat_min'] is None else float(area_bounds['lat_min'])
    lat_max = 90.0 if area_bounds['lat_max'] is None else float(area_bounds['lat_max'])
    lon_min = -180.0 if area_bounds['lon_min'] is None else float(area_bounds['lon_min'])
    lon_max = 180.0 if area_bounds['lon_max'] is None else float(area_bounds['lon_max'])
    
    print(f"Loading Global files and subsetting to area: {area}")
    print(f"  Latitude: {lat_min} to {lat_max}")
    print(f"  Longitude: {lon_min} to {lon_max}")
    print(f"  Depth threshold: {threshold}m (shallow <={threshold}, deep >{threshold})")
    
    # Collect datasets for shallow and deep separately
    shallow_list = []
    deep_list = []
    total_list = []
    max_events_shallow = 0
    max_events_deep = 0
    max_events = 0
    
    for e in tqdm(expnames, desc="Processing experiments"):
        # Always read Global files (all depth levels)
        flist = f"../output/Global_mhw_Atlas_{e}_levidx*.nc"
        filelist = sorted(glob.glob(flist))
        
        if len(filelist) == 0:
            print(f"Warning: no files found for pattern {flist}")
            continue
        
        # Load and subset all depth files for this experiment
        arrays_exp = []
        depths_exp = []
        
        for f in filelist:
            try:
                if use_dask:
                    ds = xr.open_dataset(f, chunks=chunks, cache=False)
                else:
                    ds = xr.open_dataset(f)
            except Exception as ex:
                print(f"Warning: could not open {f}: {ex}")
                continue
            
            # Subset by area bounds
            if 'latitude' in ds.coords and 'longitude' in ds.coords:
                ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
            
            # Extract the variable
            if varname in ds.data_vars:
                var = ds[varname]
            else:
                print(f"Warning: variable '{varname}' not found in {f}")
                continue
            
            # Get depth value
            if 'depth' in var.coords:
                depth_val = var.depth.values
            elif 'depth' in ds.coords:
                depth_val = ds.depth.values
            else:
                print(f"Warning: depth coordinate not found in {f}")
                continue
            
            if np.ndim(depth_val) == 0:
                depth_val_scalar = float(depth_val)
            else:
                depth_val_scalar = float(depth_val.flat[0])
            
            arrays_exp.append(ds)
            depths_exp.append(depth_val_scalar)
            max_events = max(max_events, var.sizes['event'])
        
        if len(arrays_exp) == 0:
            print(f"No data extracted for experiment '{e}'")
            continue
                
        # Concatenate along depth for this experiment
        var_all_exp = [pad_dataset_to_length(v, max_events, dim='event') for v in arrays_exp]#xr.concat(arrays_exp, dim='depth')
        var_all_exp = xr.concat(var_all_exp, dim='depth', coords='minimal', compat='override')
        var_all_exp = var_all_exp.assign_coords(depth=('depth', np.array(depths_exp)))
        var_all_exp = var_all_exp.sortby('depth')
        total_list.append(var_all_exp)
        # Separate shallow and deep
        shallow_exp = var_all_exp.sel(depth=var_all_exp.depth <= threshold)
        deep_exp = var_all_exp.sel(depth=var_all_exp.depth > threshold)
        
        # Sum across depth (to combine all depth levels within shallow/deep category)
        if shallow_exp.sizes['depth'] > 0:
            shallow_exp = shallow_exp#.sum('depth')
            shallow_list.append(shallow_exp)
            if 'event' in shallow_exp.dims:
                max_events_shallow = max(max_events_shallow, shallow_exp.sizes['event'])
        
        if deep_exp.sizes['depth'] > 0:
            deep_exp = deep_exp#.sum('depth')
            deep_list.append(deep_exp)
            if 'event' in deep_exp.dims:
                max_events_deep = max(max_events_deep, deep_exp.sizes['event'])
    
    if len(shallow_list) == 0 and len(deep_list) == 0:
        raise RuntimeError(f"No data found for area={area}, expnames={expnames}")
    
    # Pad and concatenate shallow
    if len(shallow_list) > 0:
        shallow_padded = [pad_dataset_to_length(s, max_events_shallow, dim='event') for s in shallow_list]
        shallow_exp_names = [expnames[i] for i in range(len(shallow_list))]
        shallow_ds = xr.concat(shallow_padded, dim='exp', coords='minimal', compat='override')
        shallow_ds = shallow_ds.assign_coords({'exp': ('exp', shallow_exp_names)})
    else:
        shallow_ds = None
    
    # Pad and concatenate deep
    if len(deep_list) > 0:
        deep_padded = [pad_dataset_to_length(d, max_events_deep, dim='event') for d in deep_list]
        deep_exp_names = [expnames[i] for i in range(len(deep_list))]
        deep_ds = xr.concat(deep_padded, dim='exp', coords='minimal', compat='override')
        deep_ds = deep_ds.assign_coords({'exp': ('exp', deep_exp_names)})
    else:
        deep_ds = None
    
    if len(total_list) > 0:
        total_padded = [pad_dataset_to_length(t, max_events, dim='event') for t in total_list]
        total_exp_names = [expnames[i] for i in range(len(total_list))]
        total_ds = xr.concat(total_padded, dim='exp', coords='minimal', compat='override')
        total_ds = total_ds.assign_coords({'exp': ('exp', total_exp_names)})
    else:
        total_ds = None
        
    if plot==True:
        print(f"Plotting violins in area {area}... ")
        if distribution_cache is None:
            distribution_cache = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'cache', f"violin_depth_exp_{area}.pkl"))
        df, fig = flatten_and_plot(
            shallow_ds,
            deep_ds,
            sample_frac=1,
            tile_size_deg=1.0,
            min_per_tile=1,
            use_tiles=use_tiles,
            cache_path=distribution_cache
        )
        fig.savefig(f'../violin_depth_exp_{area}.png')
    
    return total_ds, shallow_ds, deep_ds, area

def flatten_and_plot(shallow_ds=None,
                     deep_ds=None,
                     sample_frac=1, 
                     tile_size_deg=1, 
                     min_per_tile=1,
                     max_samples=20000,
                     use_tiles=True,
                     areas_script="../test_mhw_global.sh",
                     random_state=0,
                     cache_path=None):
    """
    Flatten data along latitude, longitude, and event dimensions with optional tile-based sampling,
    then plot violins with depth_category (shallow/deep) on x-axis and 'exp' as hue.

    Parameters
    ----------
    shallow_ds : xarray.Dataset or None
        Dataset for shallow events (depth <= threshold).
    deep_ds : xarray.Dataset or None
        Dataset for deep events (depth > threshold).
    sample_frac : float or None
        If given, sample this fraction per-tile (0 < sample_frac <= 1).
    max_samples : int or None
        If sample_frac is None and the total flattened rows exceed max_samples,
        a per-variable cap is applied and split among tiles.
    use_tiles : bool
        If True, perform sampling per geographic tile defined by `tile_size_deg` inside areas.
    tile_size_deg : float
        Tile size in degrees for both latitude and longitude.
    min_per_tile : int
        Minimum rows to keep per tile (if available).
    areas_script : str or None
        Path to the bash script defining `areas`.
    random_state : int
        Seed for reproducible sampling.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading cached distribution from {cache_path}...")
        df = pd.read_pickle(cache_path)
    else:
        print("Extracting and (optionally) tile-sampling...")
        data_list = []
        skip_vars = {'time', 'depth', 'exp', 'event', 'index_start', 'index_end', 'index_peak',
                     'time_start', 'time_end', 'time_peak', 'date_start', 'date_end', 'date_peak'}
    
        # Determine areas and tiles
        if areas_script is None:
            areas_script = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'test_mhw_global.sh'))
        areas = _parse_areas_from_script(areas_script) if use_tiles else []
        if use_tiles and len(areas) == 0:
            # fallback to a single global area
            areas = [{'name': 'Global', 'lat_min': None, 'lat_max': None, 'lon_min': None, 'lon_max': None}]
        tiles = []
        if use_tiles:
            for area in areas:
                tiles.extend(_make_tiles_for_area(area, tile_size_deg))
            if len(tiles) == 0:
                # safety fallback
                tiles = [(-90.0, 90.0, -180.0, 180.0)]
    
        # Process both shallow and deep datasets
        for depth_category, ds in tqdm([('Shallow', shallow_ds), ('Deep', deep_ds)], desc="Processing depth categories"):
            
            if ds is None:
                print(f"Skipping {depth_category} (no data).")
                continue
            
            # iterate variables and sample per-tile
            for var in ds.data_vars:
                if var in skip_vars:
                    continue
                da = ds[var]
                ensmean_da = da.mean(dim='exp')
                da = xr.concat([da, ensmean_da.expand_dims({'exp': ['EnsembleMean']})], dim='exp')
                try:
                    da_flat = da.stack(flat=('latitude', 'longitude', 'event', 'depth')).reset_index('flat')
                    df_temp = da_flat.to_dataframe(name='value').reset_index()
                except Exception:
                    # can't stack; skip variable
                    continue
                
                # Ensure lat/lon columns exist
                lat_col = None
                lon_col = None
                for c in ['latitude', 'lat', 'y']:
                    if c in df_temp.columns:
                        lat_col = c
                        break
                for c in ['longitude', 'lon', 'x']:
                    if c in df_temp.columns:
                        lon_col = c
                        break
                
                if lat_col is None or lon_col is None:
                    # Try to get coords from da coords and broadcast
                    if 'latitude' in ds.coords and 'longitude' in ds.coords:
                        lat_vals = ds.coords['latitude'].values
                        lon_vals = ds.coords['longitude'].values
                        # Not trivial to attach here; skip variable if no spatial coords in dataframe
                        continue
                
                # Perform tile-based sampling
                if use_tiles:
                    n_tiles = max(1, len(tiles))
                    per_tile_cap = None
                    if sample_frac is None and max_samples is not None:
                        per_tile_cap = max(min_per_tile, int(max_samples / max(1, n_tiles)))

                    # Vectorized tile assignment
                    lat_min_all = min(t[0] for t in tiles)
                    lon_min_all = min(t[2] for t in tiles)
                    lat_max_all = max(t[1] for t in tiles)
                    lon_max_all = max(t[3] for t in tiles)
                    n_lon_bins = int(np.ceil((lon_max_all - lon_min_all) / tile_size_deg))

                    lat_bin = np.floor((df_temp[lat_col] - lat_min_all) / tile_size_deg).astype(int)
                    lon_bin = np.floor((df_temp[lon_col] - lon_min_all) / tile_size_deg).astype(int)
                    tile_id = (lat_bin * n_lon_bins + lon_bin).astype(int)

                    valid_tile_ids = set()
                    for (lat0, _lat1, lon0, _lon1) in tiles:
                        lb = int(np.floor((lat0 - lat_min_all) / tile_size_deg))
                        lob = int(np.floor((lon0 - lon_min_all) / tile_size_deg))
                        valid_tile_ids.add(lb * n_lon_bins + lob)

                    df_temp = df_temp.assign(tile_id=tile_id)
                    df_temp = df_temp[df_temp['tile_id'].isin(valid_tile_ids)]
                    if df_temp.empty:
                        continue

                    if sample_frac is not None and 0 < sample_frac < 1:
                        def _sample_group(g):
                            want = int(np.ceil(len(g) * sample_frac))
                            want = max(want, min_per_tile) if len(g) >= min_per_tile else len(g)
                            want = min(want, len(g))
                            return g.sample(n=want, random_state=random_state) if len(g) > want else g
                        df_var_sampled = df_temp.groupby('tile_id', group_keys=False).apply(_sample_group)
                    elif per_tile_cap is not None:
                        def _sample_group(g):
                            want = min(per_tile_cap, len(g))
                            return g.sample(n=want, random_state=random_state) if len(g) > want else g
                        df_var_sampled = df_temp.groupby('tile_id', group_keys=False).apply(_sample_group)
                    else:
                        df_var_sampled = df_temp
                    df_var_sampled = df_var_sampled.drop(columns=['tile_id'])
                else:
                    # non-tile sampling: use sample_frac or max_samples for whole variable
                    if sample_frac is not None and 0 < sample_frac < 1:
                        df_var_sampled = df_temp.sample(frac=sample_frac, random_state=random_state)
                    elif max_samples is not None and len(df_temp) > max_samples:
                        df_var_sampled = df_temp.sample(n=max_samples, random_state=random_state)
                    else:
                        df_var_sampled = df_temp

                df_var_sampled['variable'] = var
                df_var_sampled['depth_category'] = depth_category
                data_list.append(df_var_sampled)
        
        if len(data_list) == 0:
            raise ValueError("No data variables found to flatten/plot after sampling.")
        
        df = pd.concat(data_list, ignore_index=True)
        print(f"Total rows after tile-sampling: {len(df)}")
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_pickle(cache_path)
    
    ## Ensure necessary cols exist
    #if 'exp' not in df.columns and 'exp' in (shallow_ds.coords if shallow_ds else []) or (deep_ds.coords if deep_ds else []):
    #    # Try to extract from one of the datasets
    #    ref_ds = shallow_ds if shallow_ds is not None else deep_ds
    #    if ref_ds is not None and 'exp' in ref_ds.coords:
    #        df['exp'] = np.repeat(ref_ds.exp.values.flatten()[0], len(df))
                
    # Convert depth_category to categorical for better ordering
    cat_order = ['Shallow', 'Deep']
    df['depth_category'] = pd.Categorical(df['depth_category'], categories=cat_order, ordered=True)
    
    # Convert any numeric columns to numeric
    try:
        df['value'] = pd.to_numeric(df['value'])
    except Exception:
        pass
    df = df.dropna(subset=['value'])
    
    variables = df['variable'].unique()
    n_vars = len(variables)
    n_cols = 5
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    for idx, var in enumerate(variables):
        ax = axes[idx]
        df_var = df[df['variable'] == var]
        if df_var.empty:
            ax.set_visible(False)
            continue
        sns.violinplot(data=df_var, x='depth_category', y='value', hue='exp', palette='Set2', ax=ax, cut=0, scale='width')
        ax.set_title(f'{var}')
        ax.set_ylabel('Value')
        ax.set_xlabel('Depth Category')
        ax.get_legend().remove()
    
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)
    
    # Single legend at the top of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        # fallback: collect from the first visible axis that has handles
        for ax in axes[:n_vars]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, title='exp', loc='upper center',
               ncol=len(labels), bbox_to_anchor=(0.5, 1.02),
               frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return df, fig

if __name__=='__main__':
    print(f"len of Argv: {len(sys.argv)}")
    if len(sys.argv) > 2:
        print("Running with Fire...")
        _ = fire.Fire(main)
    else:
        print("Running without Fire...")
        for area in ['WestAust', 'NortheastPacBlob', 'MediterraneanSea', 'TasmanSea', 'GreatBarrierReef', 'SantaBarbara', 'NorthWestAtlantic']:
            total_ds, shallow_ds, deep_ds, area = main(expnames=['cglo', 'glor', 'oras'], area=area, threshold=50, plot=True)
