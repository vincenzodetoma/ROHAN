# main.py
from mhw_detector.cli import run_mhw_detection
from mhw_detector.plotting import *
from mhw_detector.utils import *
from mhw_detector.clustering import *
import fire
import sys
import xarray as xr
import fire
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import sys
sys.path.append('../')  # Adjust the path to your project structure
from marineHeatWaves.marineHeatWaves import detect
import matplotlib.pyplot as plt
import gc
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
import warnings
warnings.filterwarnings("ignore")
sys.stdout.flush()
from datetime import datetime

if __name__ == "__main__":
    #cluster = LocalCluster(processes=True, n_workers=16, threads_per_worker=1, memory_limit='7GB')
    #client = Client(cluster, scheduler='synchronous')
    #print(client.dashboard_link, flush=True)
    print(len(sys.argv))
    now = datetime.now()  # current date and time
    print("Current time before fire:", now.strftime("%d-%m-%Y %H:%M:%S"))
    fire.Fire(run_mhw_detection)
    now = datetime.now()  # current date and time
    print("Current time after fire:", now.strftime("%d-%m-%Y %H:%M:%S"))

