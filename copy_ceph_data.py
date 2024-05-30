import os
from glob import glob
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd

import concurrent.futures
VARNAME_ERA2CMA = {"u10": "u10", "v10": "v10", "t2m": "t2m", "tp6h": "pre6h"}

def time2cma_path(timestamp, var_name, scale=3):
    cma_path = os.path.join(
        "np2000x2334" if scale == 3 else "",
        str(timestamp.year),
        f"{str(timestamp.month).zfill(2)}{str(timestamp.day).zfill(2)}",
        f"{timestamp.year}{str(timestamp.month).zfill(2)}{str(timestamp.day).zfill(2)}{str(timestamp.hour).zfill(2)}_{VARNAME_ERA2CMA[var_name]}.npy",
    )
    return cma_path

def time2ec_path(timestamp, var_name):
    ec_path = os.path.join(
        "single",
        str(timestamp)[:4],
        str(timestamp).replace(" ", "/") + "-" + f"{var_name}.npy"
    )
    return ec_path

def download_ceph_file(ceph_path, dst_dir, bucket):
    subdir = os.path.dirname(ceph_path)
    dst_dir = os.path.join(dst_dir, bucket, subdir)
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    os.system(f"~/bin/rclone-v1.64.2-linux-amd64/rclone copy ceph_new:{bucket}/{ceph_path} {dst_dir}")

def download_files(root_dir, timestamp, var_name):
    cma_path = time2cma_path(timestamp, var_name)
    ec_path = time2ec_path(timestamp, var_name)
    download_ceph_file(cma_path, root_dir, "cma_land")
    download_ceph_file(ec_path, root_dir, "era5_np_float32")
    print(f"Downloaded {cma_path} and {ec_path}")

if __name__ == "__main__":
    root_dir = "/mnt/hwfile/ai4earth/wangjiong/weather_data"
    time_start, time_end, time_freq, var_name = "2012-01-01", "2021-12-31-23", "6H", "u10"
    all_timestamps = pd.date_range(start=time_start, end=time_end, freq=time_freq)
    
    max_workers = 8  # 指定线程数
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for timestamp in all_timestamps:
            futures.append(executor.submit(download_files, root_dir, timestamp, var_name))
        
        # Wait for all threads to complete
        concurrent.futures.wait(futures)
