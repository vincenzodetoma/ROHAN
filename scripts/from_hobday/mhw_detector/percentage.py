import xarray as xr
import numpy as np
import fire
try:
    from .plotting import *
except ImportError:
    from plotting import *

def Percentage(da, sst):
    count = da.sum('event')
    N = len(sst.time)
    print(f"count shape {count.shape}, {sst[0].values.shape}")
    print(f"sst: {sst}")
    perc = (count / float(N)).where(~np.isnan(sst[0].values))
    return perc

def Mean(da):
    return da.mean('event')

def Max(da):
    return da.max('event')

def Std(da):
    return da.std('event')

def process_dataset(ds, filename, metric, varname, filename_sst, varname_sst, expname=None, levidx=None, vmin=None, vmax=None, levels=None, projection='platecarree', save=False, plot=False):
    print(f"opened filename {filename} into xarray dataset {ds}")
    nc_outname = filename.replace('Atlas', metric+'_'+varname) 
    print(f"Saving return to file: {nc_outname}")
    var = ds[varname]
    sst = xr.open_dataset(filename_sst)[varname_sst]
    if metric=='mean':
        ret = Mean(da=var)
    elif metric=='percentage':    
        ret = Percentage(da = var, sst=sst)
    print(ret[10,10])
    Title = f'{metric} of MHW {varname}' if expname==None and levidx==None else f"{metric} of MHW {varname} from {expname}, level {levidx}"
    if expname!='ESACCISST':
        Title += f' minus ESACCISST'
    print("vmin and vmax: ", vmin, vmax)
    if vmin==None and vmax==None:
        vmin=ret.min().values
        vmax=ret.max().values
    print("vmin and vmax after assignment: ", vmin, vmax)
    print("Saving...")
    if save:
        ret.rename(varname).to_netcdf(nc_outname)
    if plot:
        cmap='RdBu_r' if expname!='ESACCISST' else 'turbo'
        plot_map(to_plot=ret, cmap=cmap, vmin=vmin, vmax=vmax, levs=levels, title=Title, projection=projection)
    return ret

def main(filename, varname, filename_sst, varname_sst, expname=None, levidx=None, vmin=None, vmax=None, levels=None, metric='mean', projection='platecarree'):
    if expname=='ensemble':
        ref_file = filename.replace('ensemble', 'ESACCISST')
        ds_ref = xr.open_dataset(ref_file)
        print(f"Opened reference file {ref_file}")
        models = ['cglo', 'glor', 'oras']
        filenames = [filename.replace('ensemble', m) for m in models]
        for f in filenames:
            print(f"Opening file {f}")
        datasets = [xr.open_dataset(f) for f in filenames]
        rets = []
        filename_ssts = [filename_sst.replace('ensemble', m) for m in models]
        for i, ds in enumerate(datasets):
            filename_sst = filename_ssts[i]
            single_ret = process_dataset(ds, filename, metric, varname, filename_sst, varname_sst, expname, levidx, vmin, vmax, levels, projection, save=False, plot=False)
            rets.append(single_ret)
        ret = xr.concat(rets, dim='model').assign_coords({'model':models})
        ret_ref = process_dataset(ds_ref, ref_file, metric, varname, filename_sst.replace('ensemble', 'ESACCISST'), varname_sst, expname, levidx, vmin, vmax, levels, projection, save=False, plot=False)
        ret_diff = ret - ret_ref
        print("Calculation done: ", ret)
        print("Difference: ", ret_diff)
        ensmean = ret_diff.mean(dim='model')
        ensstd = ret.std(dim='model')
        Title = f'{metric} of MHW {varname} for ensemble mean'
        Title += f' minus ESACCISST'
        print("vmin and vmax: ", vmin, vmax)
        if vmin==None and vmax==None:
            vmin=ensmean.min().values
            vmax=ensmean.max().values
        p = plot_map(to_plot=ensmean, cmap='RdBu_r', vmin=vmin, vmax=vmax, levs=levels, title=Title, projection=projection)
        # get matplotlib axis from returned object (Figure or Axes-like)
        ax = p.axes if hasattr(p, 'axes') else [p.ax]
        # boolean agreement mask where residuals fall within ensemble spread
        agreement = (np.abs(ensmean.values) < ensstd.values).astype(int)
        # plot shaded overlay where agreement==1
        ax.contourf(ensmean['longitude'], ensmean['latitude'], agreement, levels=[0.5, 1.5],
                colors=['lightgrey'], transform=ccrs.PlateCarree())
        # add black borders
        ax.contour(ensmean['longitude'], ensmean['latitude'], agreement, levels=[0.5, 1.5],
               colors='k', linewidths=0.5, transform=ccrs.PlateCarree())
        plt.show()
        Title = f'{metric} of MHW {varname} for ensemble stddev'
        vmin = 0
        vmax = ensstd.max().values
        plot_map(to_plot=ensstd, cmap='Greens', vmin=vmin, vmax=vmax, levs=levels, title=Title, projection=projection)
        plt.show()
    else:
        if expname!='ESACCISST':
            filename_ref = filename.replace(expname, 'ESACCISST')
            ds_ref = xr.open_dataset(filename_ref).mean(dim='event')
            print(f"Opened reference file {filename_ref}")
        ds = xr.open_dataset(filename) - ds_ref if expname!='ESACCISST' else xr.open_dataset(filename)
        print(f"Opened file {filename}")
        print(ds[varname])
        process_dataset(ds, filename, metric, varname, filename_sst, varname_sst, expname, levidx, vmin, vmax, levels, projection, save=True, plot=True)
        plt.show()

if __name__=='__main__':
    fire.Fire(main)
