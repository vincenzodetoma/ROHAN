#%%
import xarray as xr
from tqdm import tqdm
import numpy as np
import sys
from scipy.ndimage import label, generate_binary_structure
from dask.diagnostics import ProgressBar

ProgressBar().register()

try:
    varname = sys.argv[1]
    year_end = sys.argv[2]
    suffix = sys.argv[3]
except:
    varname = 'cglo'
    year_end = '2019'
    suffix = 'original'

depths = [0.5, 10, 30, 50, 100, 150, 200]
ds = xr.open_mfdataset('../../results/' + varname + '_Global_' + suffix + '_1993-01-01_' + year_end + '-12-31_lev*.nc', concat_dim='depth', combine='nested', chunks='auto').assign_coords({'depth': depths})
mhw_bool = ds['mhw_bool_' + suffix]
mhw_anom = ds['mhw_intensity_' + suffix]
print(ds)

# Create a binary mask for pixels that belong to the marine heatwave event
mhw_mask = (mhw_bool == 1)

# Label connected components in the 4D space-time-depth array
structure = generate_binary_structure(4, 1)  # 4D structure for space-time-depth connectivity
labeled_array, num_labels = label(mhw_mask, structure=structure)
print(f"Number of labeled events: {num_labels}")

# Initialize an empty list to store the events
arrayofEvents = []

# Loop through each labeled event
for n_event in tqdm(range(1, num_labels)):
    # Create a mask for the selected event based on its label
    selected_event_mask = labeled_array == n_event
    
    # Apply the mask to the original data using xr.where
    selected_event_data = xr.where(selected_event_mask, mhw_anom, np.nan)
    
    # Drop NaN values to get the selected data
    selected_event_data = selected_event_data.dropna(dim='time', how='all')
    
    # Calculate the spatial extent of the event
    spatial_extent = np.sum(np.mean(selected_event_mask, axis=(1, 2, 3)))
    
    # Set a threshold for the minimum spatial extent (5x5 gridpoints)
    min_spatial_extent = 5 * 5
    
    # Filter out events with spatial extent smaller than the threshold
    if spatial_extent >= min_spatial_extent:
        # Continue processing the selected event
        arrayofEvents.append(selected_event_data)

# Concatenate all events into a single xarray DataArray
XR_Events = xr.concat(arrayofEvents, dim='events')

print(XR_Events)
# Save the concatenated events to a new NetCDF file
XR_Events.to_netcdf('../../results/' + varname + '_catalogue_' + suffix + '_' + year_end + '.nc')
#%%

# Assuming your xarray DataArray is named 'mhw_intensity'
# Replace 'your_variable_name' with the actual name of your variable

# Set a threshold for marine heatwave events
#threshold = 0.0  # Adjust this threshold based on your specific criteria

# Create a binary mask for pixels that belong to the marine heatwave event
#mhw_mask = (mhw_intensity > threshold) & (mhw_intensity != -999) & (mhw_intensity != -1836)

## Label connected components in the 3D space-time array
#structure = generate_binary_structure(3, 1)  # 3D structure for space-time connectivity
#labeled_array, num_labels = label(mhw_mask, structure=structure)
#print(num_labels)
## Choose a specific event index (n_event)
#i=0
#arrayofEvents = []
#for n_event in tqdm(range(1,num_labels)): 
#    # Adjust this index based on your specific needs
#    # Create a mask for the selected event based on its label
#    selected_event_mask = labeled_array == n_event
#    # Apply the mask to the original data using xr.where
#    selected_event_data = xr.where(selected_event_mask, mhw_intensity, np.nan)
#    # Drop NaN values to get the selected data
#    selected_event_data = selected_event_data.dropna(dim='time', how='all')
#    spatial_extent = np.sum(np.mean(selected_event_mask, axis=1))
#    # Set a threshold for the minimum spatial extent (5x5 gridpoints)
#    min_spatial_extent = 1 * 1
#    # Filter out events with spatial extent smaller than the threshold
#    if spatial_extent >= min_spatial_extent:
#        # Continue processing the selected event
#        # If you also want to get the coordinates of the selected pixels
#        selected_latitudes = selected_event_data.lat.values
#        selected_longitudes = selected_event_data.lon.values
#        selected_times = selected_event_data.time.values
#        # If you need to save the selected data to a new file, you can use:
#        arrayofEvents.append(selected_event_data)
#        #selected_event_data.to_netcdf('selected_mhw_event'+str(i)+'.nc')
#        i=i+1
#
#XR_Events = xr.concat(arrayofEvents, dim='events')
#XR_Events.to_netcdf('../../results/'+varname+'_catalogue_'+suffix+'_'+year_end+'.nc')
#