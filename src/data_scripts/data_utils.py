# data_utils.py
import sys
sys.dont_write_bytecode = True
import os
import numpy as np
from torch.utils.data import default_collate
import xarray as xr
import cftime
from pathlib import Path
import yaml
import logging
import gc
from datetime import timedelta
#from flufl.lock import Lock
import time
import torch.distributed as dist
from datetime import datetime
import re
from typing import Optional, Union
import pandas as pd
import dask.array as dask_array
from dask.diagnostics import ProgressBar
from tqdm import tqdm

logger = logging.getLogger()
#====================================================================
# ''' Handles printing and logging from multiple GPUs (from ChatGPT lol)'''

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
#====================================================================
def dataset_path(dataset: str, base_dir: str = None) -> Path:
    if base_dir is None:
        base_dir = os.getenv("DERIVED_DATA")

    print(f" >> >> INSIDE dataset_path | {base_dir}, {dataset}")
    logger.info(f" >> >> INSIDE dataset_path | {base_dir}, {dataset}")

    return Path(base_dir, dataset)
#====================================================================
def datafile_path(dataset: str, filename: str, base_dir: str = None) -> Path:
    p = dataset_path(dataset, base_dir=base_dir) / filename
    if is_main_process():
        print(f" >> >> INSIDE datafile_path | {p}")
    logger.info(f" >> >> INSIDE datafile_path | {p}")
    return p
#====================================================================
def open_zarr(dataset_name, filename):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    path = datafile_path(dataset_name, filename)
    if is_main_process():
        print(" >> >> INSIDE open_zarr | path:", path)
    logger.info(" >> >> INSIDE open_zarr | [Rank %d], path: %s", rank, str(path))
    
    try:
        return xr.open_zarr(str(path), consolidated=True)
    except KeyError:
        return xr.open_zarr(str(path))
#====================================================================
def get_variables(dataset_name):
    print(" >> >> INSIDE get_variables dataset_name", dataset_name)
    logger.info(" >> >> INSIDE get_variables dataset_name %s", dataset_name)
    ds_config = dataset_config(dataset_name)

    variables = ds_config["predictors"]["variables"]
    target_variables = list(
        #map(lambda v: f"target_{v}", ds_config["predictands"]["variables"])
        ds_config["predictands"]["variables"]
    )

    #print(" >> >> target_variables", target_variables)
    return variables, target_variables
#====================================================================
def get_variables_per_var(config):
    print(" >> >> INSIDE get_variables_per_var")
    logger.info(" >> >> INSIDE get_variables_per_var")
    
    variables = config.data.predictors.variables #predictors.get("variables", [])
    target_variables = config.data.predictands.variables

    return variables, target_variables
#====================================================================
def generate_output_filepath(output_dirpath):
    output_dir = Path(output_dirpath)

    # Check if the directory exists
    if not output_dir.exists():
        raise FileNotFoundError(f"The directory {output_dirpath} does not exist.")

    # Count the number of .nc files in the directory
    nc_files = list(output_dir.glob("*.nc"))
    count = len(nc_files)

    # Generate the output filepath with an incremented integer
    output_filepath = os.path.join(output_dir, "predictions-"+str(count)+".nc")

    return output_filepath
#====================================================================
TIME_RANGE = (
    datetime(2000, 6, 1),
    datetime(2024, 11, 30),
)
#====================================================================
def custom_collate(batch):
        return *default_collate([(e[0], e[1]) for e in batch]), np.concatenate(
            [e[2] for e in batch]
        )
#====================================================================
def _get_zarr_length(zarr_path):
    if is_main_process():
        print(f" >> >> INSIDE data_scripts.data_utils _get_zarr_length | zarr_path {zarr_path}")
    logger.info(f" >> >> INSIDE data_scripts.data_utils _get_zarr_length | zarr_path {zarr_path}")
    
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except KeyError:
        ds = xr.open_zarr(zarr_path)
    n = len(ds.time)
    try:
        ds.close()
    except Exception:
        # ds.close() exists and will release any file handles. (docs). 
        pass
    return n
#====================================================================
def _parse_cf_time_units(units: str):
    """
    Parse CF time units like "hours since 2000-06-01 00:00:00".
    Returns (unit, origin_str).
    Raises ValueError if unparsable.
    """
    if not isinstance(units, str):
        raise ValueError("units must be a string (CF 'units' attribute).")
    m = re.match(r'\s*(\w+)\s+since\s+(.+)', units, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized CF units string: {units!r}")
    unit = m.group(1).lower()
    origin_str = m.group(2).strip()
    # normalize unit naming
    unit_map = {
        'sec': 'seconds', 'second': 'seconds', 'seconds': 'seconds',
        'min': 'minutes', 'minute': 'minutes', 'minutes': 'minutes',
        'hour': 'hours', 'hours': 'hours',
        'day': 'days', 'days': 'days'
    }
    if unit not in unit_map:
        raise ValueError(f"Unsupported time unit '{unit}' in units '{units}'")
    return unit_map[unit], origin_str
#====================================================================
def decode_zarr_time_array(
    z_or_array,
    time_key: str = "time",
    prefer_numpy_datetime: bool = True,
) -> Union[np.ndarray, pd.DatetimeIndex]:
    """
    Decode a Zarr time array (or zarr group + key) using CF 'units' and optional 'calendar'.

    Parameters
    ----------
    z_or_array : zarr.hierarchy.Group or zarr.core.Array
        Either the opened zarr group (so we will access z_or_array[time_key]) or
        a zarr Array object that already represents the time variable.
    time_key : str
        Name of the time variable in the zarr group (default "time").
    prefer_numpy_datetime : bool
        If True and calendar is standard/gregorian, return numpy datetime64[ns] array.
        If calendar is non-standard, returns an object array of cftime datetimes.

    Returns
    -------
    np.ndarray (dtype='datetime64[ns]') or pandas.DatetimeIndex or object-array of cftime datetimes

    Notes
    -----
    - Requires pandas. If non-standard calendars are present the cftime package is used.
    - If you prefer the easiest route, use: `xr.open_zarr(path)[ "time" ].values`
    """
    # accept either a zarr.Group (access by key) or a zarr.Array
    try:
        import zarr
        is_group = hasattr(z_or_array, "array_keys") and callable(z_or_array.array_keys)
    except Exception:
        is_group = False

    if is_group:
        if time_key not in z_or_array.array_keys():
            raise KeyError(f"time key '{time_key}' not found in Zarr group keys: {list(z_or_array.array_keys())}")
        arr = z_or_array[time_key]
    else:
        arr = z_or_array

    # raw values
    vals = arr[:]  # numpy array (ints/floats or possibly already datetime64)
    # early exit if already datetime dtype
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[ns]")

    attrs = getattr(arr, "attrs", {}) or {}

    # prefer xarray-style automatic decoding if no units present
    if "units" not in attrs:
        raise ValueError("Zarr time array missing 'units' attribute. "
                         "Either open with xarray (xr.open_zarr) or ensure 'units' present in Zarr attrs.")

    units = attrs["units"]
    calendar = attrs.get("calendar", "standard").lower()

    # parse units
    unit, origin_str = _parse_cf_time_units(units)

    # if calendar is standard/gregorian -> use pandas vectorized path
    if calendar in ("standard", "gregorian", "proleptic_gregorian"):
        # parse origin to pandas.Timestamp (handles many string formats)
        origin_ts = pd.to_datetime(origin_str)
        # pandas to_timedelta: accept unit 'days','hours','minutes','seconds'
        # to_timedelta accepts fractional values.
        pandas_unit_map = {"seconds": "s", "minutes": "m", "hours": "h", "days": "D"}
        if unit not in pandas_unit_map:
            raise ValueError(f"Unit '{unit}' not supported for pandas path.")
        td = pd.to_timedelta(vals, unit=pandas_unit_map[unit])
        dtindex = origin_ts + td
        # return numpy datetime64 if requested
        if prefer_numpy_datetime:
            return dtindex.values.astype("datetime64[ns]")
        else:
            return dtindex

    # non-standard calendar: use cftime
    try:
        import cftime
    except Exception as exc:
        raise ImportError("cftime is required to decode non-standard calendars. Install `cftime`.") from exc

    # ensure 1D list
    flat_vals = np.array(vals).ravel().tolist()
    dt_objs = cftime.num2date(flat_vals, units, calendar=calendar)
    # return as object array shaped like original
    dt_arr = np.asarray(dt_objs, dtype=object).reshape(vals.shape)
    return dt_arr
#====================================================================
def count_precip_above_threshold(config, zarr_path):
    #derived_data = os.getenv('DERIVED_DATA', '.')
    #zarr_path    = os.path.join(derived_data, config.data.dataset_name)
    ds           = xr.open_zarr(zarr_path, consolidated=True)
    precip_var   = config.data.predictands.variables[0]
    precip_da    = ds[precip_var]
    
    threshold    = config.training.precip_threshold
    logger.info(f" >> >> INSIDE data_utils.count_precip_above_threshold | Calculating weight... threshold = {threshold}")

    if isinstance(precip_da.data, dask_array.Array):
        # ---- Dask path: lazy graph → single compute with progress bar ----
        count = (precip_da > threshold).sum()
        with ProgressBar():
            result = int(count.compute())
    else:
        # ---- NumPy path: chunk along time axis with tqdm ----
        result = 0
        time_chunks = np.array_split(precip_da.values, min(100, len(precip_da.time)), axis=0)
        for chunk in tqdm(time_chunks, desc="Counting", unit="chunk"):
            result += int((chunk > threshold).sum())

    weight = (precip_da.size-result) / result
    weight_path = os.path.join(os.getcwd(), 'precip_weight', config.experiment_name, config.data.dataset_name, 'precip_weight_'+str(config.training.precip_threshold)+'.npy')
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    
    logger.info(f" >> >> INSIDE data_utils.count_precip_above_threshold | Weight = {weight}\n >> >> Saving to {weight_path}")
    print(f" >> >> INSIDE data_utils.count_precip_above_threshold | Weight = {weight}\n >> >> Saving to {weight_path}")
    np.save(weight_path, weight)
     
    return weight