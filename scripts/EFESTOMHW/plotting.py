import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def single_map_plot(var, 
                    levels=None, 
                    cmap='jet', 
                    centlon=0, 
                    cbar_title=None, 
                    figtitle=None, 
                    savefig=False, 
                    figname=None, 
                    draw_labels=False,
                    x_label=None, 
                    y_label=None):
    """
    This snippet plots a single map
    """
    proj = ccrs.Robinson(central_longitude=centlon)
    if levels==None:
        levels = np.linspace(var.min(), var.max(), 50)
    fig = plt.figure(figsize=(9,4), facecolor='white')
    ax = fig.add_subplot(111, projection=proj)
    p = var.plot.contourf(ax=ax, 
                          levels=levels,
                          transform=ccrs.PlateCarree(),
                          cmap=cmap,
                          cbar_kwargs={'label':cbar_title,
                                       'orientation':'vertical'})
    ax.coastlines('50m')
    gl = ax.gridlines(draw_labels=draw_labels)
    gl.top_labels=False
    gl.right_labels=False
    ax.set_title(figtitle)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if savefig==True:
        fig.savefig(figname, dpi=300)
        
def plot_faceted(var, row_dim, col_dim, colors, levels, proj=ccrs.Robinson(central_longitude=0),
                 draw_labels=True, cbar_label='MHW Intensity [K]', savefig=False,
                 figname=None, showfig=False):
    aspratio = len(var.lon)/len(var.lat)
    # col is method, row is year
    width, height = 2*len(var[col_dim])*aspratio, 2*len(var[row_dim])
    p = var.plot.contourf(figsize=(width, height),
                          col=col_dim, row=row_dim, 
                          cmap=colors, levels=levels,
                          transform=ccrs.PlateCarree(),
                          add_colorbar=False,
                          subplot_kws={'projection':proj})
    for ax in p.axs.flatten():
        ax.coastlines('50m')
        ax.gridlines()
    
    if draw_labels==True:
        gl_last = p.axs[-1,-1].gridlines(draw_labels=False)
        gl_last.bottom_labels, gl_last.xlines, gl_last.ylines=True, False, False
        jj_rowcol=0
        for row, col in zip(p.axs[-1,:], p.axs[:,0]):
            gl_row, gl_col = row.gridlines(draw_labels=True), col.gridlines(draw_labels=True)
            gl_row.top_labels, gl_row.right_labels, gl_row.xlines, gl_row.ylines = False, False, False, False
            gl_col.top_labels, gl_col.right_labels, gl_col.xlines, gl_col.ylines=False, False, False, False
            if jj_rowcol!=0:
                gl_col.bottom_labels, gl_row.left_labels=False, False
            else:
                gl_col.bottom_labels=False
            jj_rowcol+=1

    fig = p.fig
    fig.tight_layout(rect=[0., 0., 1.5, 0.99])
    fig.facecolor = 'white'
    cbar_ax = fig.add_axes([0.87, 0.01, 0.015, 0.98]) #[0.85, 0.25, 0.015, 0.5]
    cbar = p.add_colorbar(cax=cbar_ax, orientation='vertical', label=cbar_label)
    fig.subplots_adjust(bottom=0.01, top=0.97, left=0.01, right=0.83, wspace=0., hspace=0.1)
    if savefig==True:
        fig.savefig(figname, dpi=300)
    if showfig==True:
        plt.show()
        
def plot_marineheatwaves_timeseries(sst_orig, origMHW, ThreshOrigMHW, ClimOrigMHW, title=None, 
                                    savefig=False, 
                                    figname=None):
    InterestTime = origMHW.time
    selector = xr.DataArray(InterestTime.dt.dayofyear, dims=["time"], coords=[InterestTime])
    sstMHW = origMHW.groupby('time.dayofyear') + ThreshOrigMHW
    climatology = ClimOrigMHW.sel(dayofyear=selector)
    threshold = ThreshOrigMHW.sel(dayofyear=selector)
    diff = threshold - climatology
    firstdiff = threshold + diff
    secondiff = threshold + 2*diff
    thirdiff = threshold + 3*diff
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    sstMHW.plot(ax=ax, color='k', label='mhw intensity', lw=2)
    sst_orig.plot(ax=ax, color='k', ls=':', label=' original SST')
    climatology.plot(ax=ax, color='k', ls='solid', label='climatology')
    threshold.plot(ax=ax, color='green', ls='solid', label='threshold')
    firstdiff.plot(ax=ax, color='green', ls='--', label='2x')
    secondiff.plot(ax=ax, color='green', ls='-.', label='3x')
    thirdiff.plot(ax=ax, color='green', ls=':', label='4x')
    ax.fill_between(sstMHW.time, threshold, sstMHW, where=sstMHW>threshold, color='khaki')
    ax.fill_between(sstMHW.time, firstdiff, sstMHW, where=sstMHW>firstdiff, color='orange')
    ax.fill_between(sstMHW.time, secondiff, sstMHW, where=sstMHW>secondiff, color='red')
    ax.fill_between(sstMHW.time, thirdiff, sstMHW, where=sstMHW>thirdiff, color='darkred')
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    if savefig==True:
        fig.savefig(figname, dpi=300)
    return 0