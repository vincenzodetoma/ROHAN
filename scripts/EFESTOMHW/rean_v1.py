#%%
################################## MODULES IMPORT ###########################
import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import detection
from dask.diagnostics import ProgressBar
import dask
import preprocess
import plotting
import gc
import pandas as pd
import sys
dask.config.set(**{'array.slicing.split_large_chunks': False})
from dask.distributed import Client, LocalCluster
from main import main
import fire
from dask.distributed import Client
from dask.config import set


if __name__=='__main__':
    ProgressBar().register() 
    # Start distributed client with 16 workers, each using up to 6GB RAM
    #client = Client(n_workers=16, threads_per_worker=1, memory_limit="6GB")
    #set(array_chunk_size='512MB')
    print("Firing main!")
    fire.Fire(main)
