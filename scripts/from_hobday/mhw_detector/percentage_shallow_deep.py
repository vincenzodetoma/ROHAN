#%%
import xarray as xr
import numpy as np
import fire
import glob
import sys
try:
    from .plotting import *
    from .percentage import Percentage, Mean, Max, Std
except ImportError:
    from plotting import *
    from percentage import Percentage, Mean, Max, Std

def pad_to_length(da, target_length, dim='event'):
    current_length = da.sizes[dim]
    if current_length == target_length:
        return da
    pad_shape = list(da.shape)
    pad_shape[da.get_axis_num(dim)] = target_length - current_length
    pad = xr.full_like(da.isel({dim: slice(0,1)}), np.nan).isel({dim: 0}).expand_dims({dim: pad_shape[da.get_axis_num(dim)]})
    da_padded = xr.concat([da, pad], dim=dim)
    return da_padded

def get_shallow_deep(filenames, varname, threshold=100):
    # Gestione lista file
    if isinstance(filenames, str) and '*' in filenames:
        filenames = glob.glob(filenames)
    elif isinstance(filenames, str):
        filenames = filenames.split(',')
    print(filenames)

    # Trova la dimensione massima tra tutti i file
    max_events = max([xr.open_dataset(f)[varname].sizes['event'] for f in filenames])
    print(f"Maximum number of events across files: {max_events}")

    # Carica SOLO la variabile di interesse, pad e concatena come DataArray
    arrays = []
    depths = []
    for f in filenames:
        ds = xr.open_dataset(f)
        var = ds[varname]
        var_padded = pad_to_length(var, max_events, dim='event')
        arrays.append(var_padded)
        # Ricava la profondità associata a questo file
        if 'depth' in var.coords:
            depth_val = var.depth.values
        elif 'depth' in ds.coords:
            depth_val = ds.depth.values
        else:
            raise ValueError("Depth coordinate not found in file: " + f)
        if np.ndim(depth_val) == 0:
            depths.append(depth_val)
        else:
            depths.append(depth_val[0])
    # Concatena lungo 'depth'
    var_all = xr.concat(arrays, dim='depth')
    var_all = var_all.assign_coords(depth=('depth', np.array(depths)))
    var_all = var_all.sortby('depth')
    print(var_all)

    # Separa shallow e deep
    shallow = var_all.sel(depth=var_all.depth <= threshold)
    deep = var_all.sel(depth=var_all.depth > threshold)
    return shallow, deep
    
def main(area='Global', varname='intensity_var', filename_sst='mhw_climatology.nc', varname_sst='mhw_intensity', expnames=['cglo', 'glor', 'oras'], threshold=50, diagnostic='Mean'):
    sst = xr.open_dataset(f"../output/{area}_"+filename_sst.replace('.nc', '_ESACCISST_levidx1.nc'))[varname_sst]
    projection='miller'
    shallowList = []
    deepList = []
    #expnames = [expnames[0]]
    print(expnames)
    max_event_shallow = 0
    max_event_deep = 0
    temp_shallow = []
    temp_deep = []
    for e in expnames:
        flist = f"../output/{area}_mhw_Atlas_{e}_levidx*.nc"
        filelist = glob.glob(flist)
        print(f"Pattern: {flist}")
        print(f"Files: {filelist}")
        shallow, deep = get_shallow_deep(filenames=filelist, varname=varname, threshold=threshold)
        temp_shallow.append(shallow)
        temp_deep.append(deep)
        if 'event' in shallow.dims:
            max_event_shallow = max(max_event_shallow, shallow.sizes['event'])
        if 'event' in deep.dims:
            max_event_deep = max(max_event_deep, deep.sizes['event'])
    # Pad shallow e deep per uniformare la dimensione 'event'
    shallowList = [pad_to_length(ds, max_event_shallow, dim='event') for ds in temp_shallow]
    deepList = [pad_to_length(ds, max_event_deep, dim='event') for ds in temp_deep]
    ShallowXR = xr.concat(shallowList, dim='exp').assign_coords({'exp': expnames})
    DeepXR = xr.concat(deepList, dim='exp').assign_coords({'exp': expnames})
    # Percentuale shallow
    obs=varname
    if diagnostic == 'Percentage':
        perc_shallow = Percentage(da=ShallowXR.sum('depth'), sst=sst)
    elif diagnostic == 'Mean':
        perc_shallow = Mean(da=ShallowXR.sum('depth'))
    elif diagnostic == 'Max':
        perc_shallow = Max(da=ShallowXR.max('depth'))
    elif diagnostic == 'Std':
        perc_shallow = Std(da=ShallowXR.mean('depth'))
    if varname=='duration':
        obs= 'MHW duration [days]'
    perc_shallow = perc_shallow.where(~np.isnan(sst[0].values))
    WRAP = int(len(expnames)/2) if len(expnames)>1 else 1
    WRAP=4
    cmap='afmhot_r' if varname=='category' else 'jet'
    levs=[-0.5,0.5,1.5,2.5,3.5,4.5] if varname=='category' else None
    lvtcks=[0,1,2,3,4] if varname=='category' else None
    vmin = 0 if varname!='category' else levs[0]
    vmax = 65 if varname!='category' else levs[-1]
    if varname=='duration':
        vmin=0
        vmax=40
        if diagnostic=='Percentage':
            vmin=0
            vmax=0.40
    if varname=='rate_decline':
        obs='MHW rate of decline [K/day]'
        vmin=0
        vmax=0.30
    if varname=='rate_onset':
        obs='MHW rate of onset [K/day]'
        vmin=0
        vmax=0.5
    if varname=='intensity_cumulative':
        obs='MHW cumulative intensity [K*days]'
        vmin=0
        vmax=100
    if varname=='intensity_var':
        obs='MHW intensity variance [K^2]' 
        vmin=0
        vmax=0.5
    lbls = ['None', 'Moderate', 'Strong', 'Severe', 'Extreme'] if varname=='category' else None
    ensmean_perc_shallow = perc_shallow.mean(dim='exp')
    ensstd_perc_shallow = perc_shallow.std(dim='exp')
    print("ensstd_perc_shallow:", ensstd_perc_shallow)
    perc_shallow = xr.concat([perc_shallow, ensmean_perc_shallow.expand_dims({'exp':['Ensemble Mean']})], dim='exp').assign_coords({'exp':list(perc_shallow.exp.values)+['Ensemble Mean']})
    
    p1 = plot_map(perc_shallow, cwrap=WRAP, cmap=cmap, levs=levs, suptitle=f"Shallow (<{threshold}m) {diagnostic} of {obs}", vmin=vmin, vmax=vmax, projection=projection, tcks=lvtcks, tcklabs=lbls)
    for i, ax in enumerate(p1.axs.flat):
        print("subplot number:", i)
        ax.set_title(f"exp = {perc_shallow.exp.values[i]}")
        if i==3:
            ax.set_title(f"exp = Ensemble Mean \n Std Dev range: {ensstd_perc_shallow.min().values:.2f} - {ensstd_perc_shallow.max().values:.2f}")
    plt.show()

    # Percentuale deep
    if diagnostic=='Percentage':
        perc_deep = Percentage(da=DeepXR.sum('depth'), sst=sst)
    elif diagnostic=='Mean':
        perc_deep = Mean(da=DeepXR.sum('depth'))
    elif diagnostic=='Max':
        perc_deep = Max(da=DeepXR.max('depth'))
    elif diagnostic=='Std':
        perc_deep = Std(da=DeepXR.mean('depth'))
    perc_deep = perc_deep.where(~np.isnan(sst[0].values))
    ensmean_perc_deep = perc_deep.mean(dim='exp')
    ensstd_perc_deep = perc_deep.std(dim='exp')
    print("ensstd_perc_deep:", ensstd_perc_deep)
    perc_deep = xr.concat([perc_deep, ensmean_perc_deep.expand_dims({'exp':['Ensemble Mean']})], dim='exp').assign_coords({'exp':list(perc_deep.exp.values)+['Ensemble Mean']})
    
    p2 = plot_map(perc_deep, cwrap=WRAP, cmap=cmap, levs=levs, suptitle=f"Deep (>{threshold}m) {diagnostic} of {obs}", vmin=vmin, vmax=vmax, projection=projection, tcks=lvtcks, tcklabs=lbls)
    for i, ax in enumerate(p2.axs.flat):
        print("subplot number:", i)
        ax.set_title(f"exp = {perc_deep.exp.values[i]}")
        if i==3:
            ax.set_title(f"exp = Ensemble Mean, \n Std Dev range: {ensstd_perc_deep.min().values:.2f} - {ensstd_perc_deep.max().values:.2f}")
    plt.show()

if __name__=='__main__':
    print(f"len of Argv: {len(sys.argv)}")
    if len(sys.argv) >2:
        print("Running with Fire...")
        fire.Fire(main)
    else:
        print("Running without Fire...")
        main()
# %%
