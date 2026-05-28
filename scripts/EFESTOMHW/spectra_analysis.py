#%%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('spectra/originalspec.out', header=0, names=['frequency', 'power'], delim_whitespace=True)
df2 = pd.read_csv('spectra/fastSSA.out', header=0, names=['frequency2', 'power2'], delim_whitespace=True)
df3 = pd.read_csv('spectra/butterspec.out', header=0, names=['frequency3', 'power3'], delim_whitespace=True)
df4 = pd.read_csv('spectra/butterspec_o2.out', header=0, names=['frequency4', 'power4'], delim_whitespace=True)
df_sig = pd.read_csv('spectra/confidence.out', header=0, names=['frequency', '90perc', '95perc', '99perc'], delim_whitespace=True)
freq, power = df['frequency'], df['power']
freq2, power2 = df2['frequency2'], df2['power2']
freq3, power3 = df3['frequency3'], df3['power3']
freq4, power4 = df4['frequency4'], df4['power4']
freq5, perc99 = df_sig['frequency'], df_sig['90perc']
fig = plt.figure(1, figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(freq, power, label='original', marker='o', markersize=5, markeredgecolor='k', color='k')
ax.plot(freq2, power2, label='fast_1y_SSA', marker='o', markersize=5, markeredgecolor='k', color='blue')
ax.plot(freq3, power3, label='butterworth_1y_o1', marker='o', markersize=5, markeredgecolor='k', color='red')
ax.plot(freq4, power4, label='butterworth_1y_o2', marker='o', markersize=5, markeredgecolor='k', color='green')
ax.plot(freq4, perc99, label='99_confidence AR(1)', linestyle='dashdot', color='magenta', lw=4)
ax.loglog()
ax.grid(which='both')
ax.axvline(1/365, color='k', label='1 year')
ax.axvline(1/(365*0.5), linestyle='dashed', color='k', label='6 months')
ax.set_xlim(10**-4, 10**-2)
ax.legend(ncol=2, loc='lower left')
left, bottom, width, height = [0.65, 0.12, 0.3, 0.25]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(freq, power, label='original', marker='o', markersize=5, markeredgecolor='k', color='k')
ax2.plot(freq2, power2, label='fast_1y_SSA', marker='o', markersize=5, markeredgecolor='k', color='blue')
ax2.plot(freq3, power3, label='butterworth_1y_o1', marker='o', markersize=5, markeredgecolor='k', color='red')
ax2.plot(freq4, power4, label='butterworth_1y_o2', marker='o', markersize=5, markeredgecolor='k', color='green')
ax2.plot(freq4, perc99, label='99_confidence AR(1)', linestyle='dashdot', color='magenta', lw=4)
ax2.loglog()
ax2.grid(which='both')
ax2.axvline(1/365, color='k', label='1 year')
ax2.axvline(1/(365*0.5), linestyle='dashed', color='k', label='6 months')
ax.set_xlim(10**-4, 10**-2)
ax.set_xlabel('Frequency [cycles/day]')
ax.set_ylabel('PSD [K/(cycles*day)]')
ax.set_title('SST PSD in TPac, (lat=0°N, lon=140°W)')
fig.tight_layout()
fig.savefig('spectral.png', dpi=300)
# %%
