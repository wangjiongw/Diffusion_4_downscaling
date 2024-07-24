import glob
import json
import os
import random
from bisect import bisect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
        str(timestamp).replace(" ", "/") + "-" + f"{var_name}.npy",
    )
    return ec_path


class SR3_CNDataset_patch(torch.utils.data.Dataset):
    def __init__(
        self,
        lr_root,
        hr_root,
        land_paths,
        mask_paths,
        var,
        patch_size,
        year_start="2012-01-01",
        year_end="2020-12-31-23",
        year_freq="6H",
        scale=1,
    ):
        assert var in ["u10", "v10", "t2m", "tp", "tp6h"]
        self.lr_root = lr_root
        self.hr_root = hr_root
        self.variable_name = var
        self.scale = scale
        self.patch_size = patch_size

        self.land_01 = np.expand_dims(
            np.load(land_paths, mmap_mode="r+")[:: self.scale, :: self.scale], axis=0
        )  # 1, H(6000/2000), W(7000/2334)
        self.mask_data = np.expand_dims(
            np.load(mask_paths, mmap_mode="r+")[:: self.scale, :: self.scale], axis=0
        )  # 1, H(6000/2000), W(7000/2334)
        if scale == 3:
            hr_height, hr_width = 2000, 2334
        elif scale == 1:
            hr_height, hr_width = 6001, 7001

        self.land_01 = interpolate(
            torch.from_numpy(self.land_01).float().unsqueeze(0),  # 1, 1, H, W
            size=(hr_height, hr_width),
            mode="bilinear",
        ).squeeze(0)
        self.mask_data = interpolate(
            torch.from_numpy(self.mask_data).float().unsqueeze(0),
            size=(hr_height, hr_width),
            mode="bilinear",
        ).squeeze(
            0
        )  # 1, H, W; torch.tensor
        self.land_01 = (self.land_01 - np.min(self.land_01)) / (
            np.max(self.land_01) - np.min(self.land_01)
        )
        self.mask_data = (self.mask_data - np.min(self.mask_data)) / (
            np.max(self.mask_data) - np.min(self.mask_data)
        )

        # build file paths
        self.timestamps = pd.date_range(start=year_start, end=year_end, freq=year_freq)
        self.lr_paths, self.hr_paths = list(), list()
        for time_stamp in tqdm(self.timestamps):
            lr_bucket_path = time2ec_path(time_stamp, self.variable_name)
            hr_bucket_path = time2cma_path(
                time_stamp, self.variable_name, scale=self.scale
            )
            lr_path = os.path.join(self.lr_root, lr_bucket_path)
            hr_path = os.path.join(self.hr_root, hr_bucket_path)
            if not os.path.exists(lr_path) or not os.path.exists(hr_path):
                print(f"{lr_path} | {hr_path} not okay for training!")
                continue
            self.lr_paths.append(lr_path)
            self.hr_paths.append(hr_path)
        assert len(self.lr_paths) == len(self.hr_paths)
        print(
            f"File counts: {len(self.lr_paths)} LR files & {len(self.hr_paths)} HR files found!"
        )
        self.data_count = len(self.lr_paths)

        # load stats to normalize
        self.stats_cma = self.get_stats_cma()
        self.stats_ec = self.get_stats_ec()
        # print(self.max.shape)

    def get_stats_cma(self):
        """load mean & std of data of CMA variables in self.vnames; SINGLE"""

        with open(
            "/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/nwp/datasets/mean_std_cma.json",
            mode="r",
        ) as f:
            mean_std_cma = json.load(f)
            f.close()
        mean_list, std_list = [], []
        # stat for CMA
        mean_list.append(mean_std_cma["mean"][self.variable_name])
        std_list.append(mean_std_cma["std"][self.variable_name])
        return dict(
            mean=np.array(mean_list, dtype=np.float32),  # C(1)
            std=np.array(std_list, dtype=np.float32),  # C(1)
        )

    def get_stats_ec(self):
        """load mean & std of data of ERA5 variables in self.vnames; PRESSURE-SINGLE

        Raises:
            ValueError: _description_

        Returns:
            _type_: dictionary of mean & std
        """
        with open(
            "/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/nwp/datasets/mean_std_single.json",
            mode="r",
        ) as f:
            mean_std_single = json.load(f)
            f.close()

        mean_list, std_list = [], []
        # stat of FengWu multi-level variables; Order: surface -> low pressure -> high pressure
        # stat for ERA5 surface
        mean_list.append(mean_std_single["mean"][self.variable_name])
        std_list.append(mean_std_single["std"][self.variable_name])

        return dict(
            mean=np.array(mean_list, dtype=np.float32),
            std=np.array(std_list, dtype=np.float32),
        )

    def normalize_max_min(self, data, iy, ix, ip):
        var_list = {
            "u": 0,
            "v": 1,
            "t2m": 2,
            "sp": 3,
            "tp": 4,
        }
        index_var = var_list[self.variable_name]

        max_ = self.max[index_var : index_var + 1, iy : iy + ip, ix : ix + ip]
        min_ = self.min[index_var : index_var + 1, iy : iy + ip, ix : ix + ip]

        # print(max_.max(),max_.min())
        # print((max_-min_).max(),(max_-min_).min())
        return (data - min_) / (max_ - min_ + 1e-6)

    def normalize_data(self, data, data_mean, data_std):
        """expect torch.tensor

        Args:
            data (_type_): _description_
            data_mean (_type_): _description_
            data_std (_type_): _description_

        Returns:
            _type_: _description_
        """
        data -= data_mean[:, None, None]
        data /= data_std[:, None, None]
        return data

    def clean_cma_data(self, cma_data, return_gt=False):
        """process invalid data points in CMA data

        Args:
            frame_data (np.array): CMA data; C, H, W
            return_gt (bool): whether the data is GT or not

        Returns:
            np.array: cleaned data of current frame
        """
        # cma_data = frame_data[c_idx]
        if self.variable_name == "t2m":
            cma_data[cma_data == 9999.0] = 0.0 if return_gt else 0.0
            cma_data = np.clip(cma_data, a_min=213.15, a_max=333.15)
        elif self.variable_name == "r2":
            cma_data = np.clip(cma_data, a_min=0.0, a_max=100.0)
        elif self.variable_name == "u10" or self.variable_name == "v10":
            cma_data[cma_data < -1000.0] = 0.0 if return_gt else 0.0
            cma_data[cma_data > 1000.0] = 0.0 if return_gt else 0.0
            cma_data = np.clip(cma_data, a_min=-1000.0, a_max=1000.0)
        elif self.variable_name == "pre":
            cma_data[cma_data < 0] = 0.0 if return_gt else 0.0  # pre must larger than 0
            cma_data = np.clip(cma_data, a_min=0.0, a_max=300.0)
        elif self.variable_name == "pre6h":
            cma_data[cma_data < 0] = 0.0 if return_gt else 0.0  # pre must larger than 0
            cma_data = np.clip(cma_data, a_min=0.0, a_max=1500.0)

        return cma_data

    def get_patch(self, hr, mask, hr_land, lr_inter):

        ih_hr, iw_hr = hr.shape[1:]
        ip = self.patch_size
        ix = random.randrange(0, iw_hr - ip + 1)
        iy = random.randrange(0, ih_hr - ip + 1)

        # print(ip, ix, iy, hr.shape, mask.shape, hr_land.shape, lr_inter.shape, flush=True)
        mask_data = (mask[:, iy : iy + ip, ix : ix + ip]).float()
        land_data = (hr_land[:, iy : iy + ip, ix : ix + ip]).float()

        # if self.variable_name in ["u10", "v10", "t2m", "sp"]:

        lr_data = self.normalize_data(
            lr_inter[:, iy : iy + ip, ix : ix + ip],
            torch.from_numpy(self.stats_ec["mean"]),
            torch.from_numpy(self.stats_ec["std"]),
        )
        ret = {
            "HR": self.normalize_data(
                torch.from_numpy(hr[:, iy : iy + ip, ix : ix + ip]).float(),
                torch.from_numpy(self.stats_cma["mean"]),
                torch.from_numpy(self.stats_cma["std"]),
            ),
            "mask": mask_data,
            "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
            "LAND": land_data,
        }
        # else:
        #     lr_data = lr_inter[:, iy : iy + ip, ix : ix + ip].float()
        #     ret = {
        #         "HR": torch.from_numpy(hr[:, iy : iy + ip, ix : ix + ip]).float(),
        #         "mask": mask_data,
        #         "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
        #         "LAND": land_data,
        #     }

        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        land_01_data = self.land_01  # topological data
        mask_data = self.mask_data  # land_sea mask

        hr_target = np.expand_dims(
            np.load(self.hr_paths[index], mmap_mode="r+")[::-1, :], axis=0
        )  # 1, H, W; torch
        hr_target = self.clean_cma_data(hr_target)
        hr_target = np.ascontiguousarray(hr_target)

        if "tp" in self.variable_name:
            lr_inter = interpolate(
                torch.from_numpy(
                    np.expand_dims(
                        np.load(self.lr_paths[index], mmap_mode="r+")[120:361, 280:561],
                        axis=0,
                    )
                )
                .float()
                .unsqueeze(0),
                size=(hr_target.shape[1], hr_target.shape[2]),
                mode="bilinear",
            ).squeeze(
                0
            )  # 1, H, W; torch
        else:
            lr_inter = interpolate(
                torch.from_numpy(
                    np.expand_dims(
                        np.load(self.lr_paths[index], mmap_mode="r+")[120:361, 280:561],
                        axis=0,
                    )
                )
                .float()
                .unsqueeze(0),
                size=(hr_target.shape[1], hr_target.shape[2]),
                mode="bicubic",
            ).squeeze(0)
        lr_inter = lr_inter.contiguous()
        return self.get_patch(hr_target, mask_data, land_01_data, lr_inter)


class SR3_CNDataset_patch_preload(torch.utils.data.Dataset):
    def __init__(
        self,
        lr_root,
        hr_root,
        land_paths,
        mask_paths,
        var,
        patch_size,
        area="china",
        year_start="2012-01-01",
        year_end="2020-12-31-23",
        year_freq="6H",
        scale=1,
    ):
        assert var in ["u10", "v10", "t2m", "tp", "tp6h"]
        self.lr_root = lr_root
        self.hr_root = hr_root
        self.variable_name = var
        self.scale = scale
        self.patch_size = patch_size
        self.area = area

        self.land_01 = np.expand_dims(
            np.load(land_paths, mmap_mode="r+")[:: self.scale, :: self.scale], axis=0
        )  # 1, H(6000/2000), W(7000/2334)
        self.mask_data = np.expand_dims(
            np.load(mask_paths, mmap_mode="r+")[:: self.scale, :: self.scale], axis=0
        )  # 1, H(6000/2000), W(7000/2334)

        if scale == 3:
            hr_height, hr_width = 2000, 2334
        elif scale == 1:
            hr_height, hr_width = 6001, 7001

        self.land_01 = interpolate(
            torch.from_numpy(self.land_01).float().unsqueeze(0),  # 1, 1, H, W
            size=(hr_height, hr_width),
            mode="bilinear",
        ).squeeze(0)
        self.mask_data = interpolate(
            torch.from_numpy(self.mask_data).float().unsqueeze(0),
            size=(hr_height, hr_width),
            mode="nearest",
        ).squeeze(
            0
        )  # 1, H, W; torch.tensor
        self.land_01 = (self.land_01 - torch.min(self.land_01)) / (
            torch.max(self.land_01) - torch.min(self.land_01)
        )
        self.mask_data = (self.mask_data - torch.min(self.mask_data)) / (
            torch.max(self.mask_data) - torch.min(self.mask_data)
        )  # 1==land; 0==sea
        # if self.area == "gansu":
        #     self.land_01 = self.land_01[:]
        #     self.mask_data = self.mask_data[]
        # print(self.land_01.shape, self.mask_data.shape)

        # build file paths
        self.timestamps = pd.date_range(start=year_start, end=year_end, freq=year_freq)
        self.lr_items, self.hr_items = list(), list()
        for time_stamp in self.timestamps:
            lr_bucket_path = time2ec_path(time_stamp, self.variable_name)
            hr_bucket_path = time2cma_path(
                time_stamp, self.variable_name, scale=self.scale
            )
            lr_path = os.path.join(self.lr_root, lr_bucket_path)
            hr_path = os.path.join(self.hr_root, hr_bucket_path)
            if not os.path.exists(lr_path) or not os.path.exists(hr_path):
                print(f"{lr_path} | {hr_path} NOT Available for training!")
                continue
            self.lr_items.append(np.load(lr_path, mmap_mode="r+"))
            self.hr_items.append(np.load(hr_path, mmap_mode="r+"))

        assert len(self.lr_items) == len(self.hr_items)
        print(
            f"File counts: {len(self.lr_items)} LR files & {len(self.hr_items)} HR files found!"
        )
        self.data_count = len(self.lr_items)

        # load stats to normalize
        self.stats_cma = self.get_stats_cma()
        self.stats_ec = self.get_stats_ec()
        # print(self.max.shape)

    def get_stats_cma(self):
        """load mean & std of data of CMA variables in self.vnames; SINGLE"""

        with open(
            "/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/nwp/datasets/mean_std_cma.json",
            mode="r",
        ) as f:
            mean_std_cma = json.load(f)
            f.close()
        mean_list, std_list = [], []
        # stat for CMA
        mean_list.append(mean_std_cma["mean"][self.variable_name])
        std_list.append(mean_std_cma["std"][self.variable_name])
        return dict(
            mean=np.array(mean_list, dtype=np.float32),  # C(1)
            std=np.array(std_list, dtype=np.float32),  # C(1)
        )

    def get_stats_ec(self):
        """load mean & std of data of ERA5 variables in self.vnames; PRESSURE-SINGLE

        Raises:
            ValueError: _description_

        Returns:
            _type_: dictionary of mean & std
        """
        with open(
            "/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/nwp/datasets/mean_std_single.json",
            mode="r",
        ) as f:
            mean_std_single = json.load(f)
            f.close()

        mean_list, std_list = [], []
        # stat of FengWu multi-level variables; Order: surface -> low pressure -> high pressure
        # stat for ERA5 surface
        mean_list.append(mean_std_single["mean"][self.variable_name])
        std_list.append(mean_std_single["std"][self.variable_name])

        return dict(
            mean=np.array(mean_list, dtype=np.float32),
            std=np.array(std_list, dtype=np.float32),
        )

    def normalize_max_min(self, data, iy, ix, ip):
        var_list = {
            "u": 0,
            "v": 1,
            "t2m": 2,
            "sp": 3,
            "tp": 4,
        }
        index_var = var_list[self.variable_name]

        max_ = self.max[index_var : index_var + 1, iy : iy + ip, ix : ix + ip]
        min_ = self.min[index_var : index_var + 1, iy : iy + ip, ix : ix + ip]

        return (data - min_) / (max_ - min_ + 1e-6)

    def normalize_data(self, data, data_mean, data_std):
        """expect torch.tensor

        Args:
            data (_type_): _description_
            data_mean (_type_): _description_
            data_std (_type_): _description_

        Returns:
            _type_: _description_
        """
        data -= data_mean[:, None, None]
        data /= data_std[:, None, None]
        return data

    def clean_cma_data(self, cma_data, return_gt=False):
        """process invalid data points in CMA data

        Args:
            frame_data (np.array): CMA data; C, H, W
            return_gt (bool): whether the data is GT or not

        Returns:
            np.array: cleaned data of current frame
        """
        # cma_data = frame_data[c_idx]
        if self.variable_name == "t2m":
            cma_data[cma_data == 9999.0] = 0.0 if return_gt else 0.0
            cma_data = np.clip(cma_data, a_min=213.15, a_max=333.15)
        elif self.variable_name == "r2":
            cma_data = np.clip(cma_data, a_min=0.0, a_max=100.0)
        elif self.variable_name == "u10" or self.variable_name == "v10":
            cma_data[cma_data < -1000.0] = 0.0 if return_gt else 0.0
            cma_data[cma_data > 1000.0] = 0.0 if return_gt else 0.0
            cma_data = np.clip(cma_data, a_min=-1000.0, a_max=1000.0)
        elif self.variable_name == "pre":
            cma_data[cma_data < 0] = 0.0 if return_gt else 0.0  # pre must larger than 0
            cma_data = np.clip(cma_data, a_min=0.0, a_max=300.0)
        elif self.variable_name == "pre6h":
            cma_data[cma_data < 0] = 0.0 if return_gt else 0.0  # pre must larger than 0
            cma_data = np.clip(cma_data, a_min=0.0, a_max=1500.0)
        return cma_data

    def get_patch(self, hr, mask, hr_land, lr_inter):

        ih_hr, iw_hr = hr.shape[1:]
        ip = self.patch_size
        ix = random.randrange(0, iw_hr - ip + 1)
        iy = random.randrange(0, ih_hr - ip + 1)

        mask_data = (mask[:, iy : iy + ip, ix : ix + ip]).float()
        land_data = (hr_land[:, iy : iy + ip, ix : ix + ip]).float()

        lr_data = self.normalize_data(
            lr_inter[:, iy : iy + ip, ix : ix + ip],
            torch.from_numpy(self.stats_ec["mean"]),
            torch.from_numpy(self.stats_ec["std"]),
        )
        ret = {
            "HR": self.normalize_data(
                torch.from_numpy(hr[:, iy : iy + ip, ix : ix + ip]).float(),
                torch.from_numpy(self.stats_cma["mean"]),
                torch.from_numpy(self.stats_cma["std"]),
            ),
            "mask": mask_data,
            "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
            "LAND": land_data,
        }
        # else:
        #     lr_data = lr_inter[:, iy : iy + ip, ix : ix + ip].float()
        #     ret = {
        #         "HR": torch.from_numpy(hr[:, iy : iy + ip, ix : ix + ip]).float(),
        #         "mask": mask_data,
        #         "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
        #         "LAND": land_data,
        #     }

        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        if self.area == "china":
            cma_start_idx_lat, cma_end_idx_lat = 0, self.hr_items[0].shape[0]
            cma_start_idx_lon, cma_end_idx_lon = 0, self.hr_items[0].shape[1]
            ec_start_idx_lat, ec_end_idx_lat = int((30 + cma_start_idx_lat * 0.03) // 0.25), int((30 + cma_end_idx_lat * 0.03) // 0.25 + 1)
            ec_start_idx_lon, ec_end_idx_lon = int((70 + cma_start_idx_lon * 0.03) // 0.25), int((70 + cma_end_idx_lon * 0.03) // 0.25 + 1)
        elif self.area == "gansu":
            # center crop ganse area with patch size; center at 37.5N, 100E
            cma_start_idx_lat, cma_end_idx_lat = int(750 - self.patch_size // 2), int(750 + self.patch_size // 2)
            cma_start_idx_lon, cma_end_idx_lon = int(1000 - self.patch_size // 2), int(1000 + self.patch_size // 2)
            ec_start_idx_lat, ec_end_idx_lat = int((30 + cma_start_idx_lat * 0.03) // 0.25), int((30 + cma_end_idx_lat * 0.03) // 0.25 + 1)
            ec_start_idx_lon, ec_end_idx_lon = int((70 + cma_start_idx_lon * 0.03) // 0.25), int((70 + cma_end_idx_lon * 0.03) // 0.25 + 1)
        
        land_01_data = self.land_01[:, cma_start_idx_lat:cma_end_idx_lat, cma_start_idx_lon: cma_end_idx_lon]  # topological data
        mask_data = self.mask_data[:, cma_start_idx_lat:cma_end_idx_lat, cma_start_idx_lon: cma_end_idx_lon]   # land_sea mask

        hr_target = np.expand_dims(
            self.hr_items[index][::-1, :][cma_start_idx_lat:cma_end_idx_lat, cma_start_idx_lon: cma_end_idx_lon], axis=0
        )  # 1, H, W; torch
        hr_target = self.clean_cma_data(hr_target)
        hr_target = np.ascontiguousarray(hr_target)
        
        if "tp" in self.variable_name:
            lr_inter = interpolate(
                torch.from_numpy(
                    np.expand_dims(
                        self.lr_items[index][ec_start_idx_lat:ec_end_idx_lat, ec_start_idx_lon:ec_end_idx_lon],
                        axis=0,
                    )
                )
                .float()
                .unsqueeze(0),
                size=(hr_target.shape[1], hr_target.shape[2]),
                mode="bilinear",
            ).squeeze(
                0
            )  # 1, H, W; torch
        else:
            lr_inter = interpolate(
                torch.from_numpy(
                    np.expand_dims(
                        self.lr_items[index][ec_start_idx_lat:ec_end_idx_lat, ec_start_idx_lon:ec_end_idx_lon],
                        axis=0,
                    )
                )
                .float()
                .unsqueeze(0),
                size=(hr_target.shape[1], hr_target.shape[2]),
                mode="bicubic",
            ).squeeze(0)
        lr_inter = lr_inter.contiguous()
        return self.get_patch(hr_target, mask_data, land_01_data, lr_inter)


class SR3_CNDataset_all_preload(torch.utils.data.Dataset):
    def __init__(
        self,
        var,
        land_paths,
        mask_paths,
        lr_paths,
        hr_paths=None,
        # patch_size,
        year_start="2012-01-01",
        year_end="2020-12-31-23",
        year_freq="6H",
        scale=1,
        fengwu_steps=-1
    ):
        assert var in ["u10", "v10", "t2m", "tp", "tp6h"]
        # self.lr_root = lr_root
        # self.hr_root = hr_root
        self.variable_name = var
        self.scale = scale
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths

        if scale == 3:
            hr_height, hr_width = 2000, 2334
        elif scale == 1:
            hr_height, hr_width = 6001, 7001

        # load land mask & DEM data
        self.land_01 = np.expand_dims(
            np.load(land_paths, mmap_mode="r+")[:: self.scale, :: self.scale], axis=0
        )  # 1, H(6000/2000), W(7000/2334)
        self.mask_data = np.expand_dims(
            np.load(mask_paths, mmap_mode="r+")[:: self.scale, :: self.scale], axis=0
        )  # 1, H(6000/2000), W(7000/2334)

        self.land_01 = interpolate(
            torch.from_numpy(self.land_01).float().unsqueeze(0),  # 1, 1, H, W
            size=(hr_height, hr_width),
            mode="bilinear",
        ).squeeze(0)
        self.land_01 = (self.land_01 - torch.min(self.land_01)) / (
            torch.max(self.land_01) - torch.min(self.land_01)
        )
        self.mask_data = interpolate(
            torch.from_numpy(self.mask_data).float().unsqueeze(0),
            size=(hr_height, hr_width),
            mode="nearest",
        ).squeeze(0)  # 1, H, W; torch.tensor
        self.mask_data = (self.mask_data - torch.min(self.mask_data)) / (
            torch.max(self.mask_data) - torch.min(self.mask_data)
        )  # 0==land; 1==sea
        self.mask_data = (1 - self.mask_data)  # 1==land; 0==sea

        # build file paths
        self.lr_items, self.hr_items = list(), list()
        if isinstance(self.lr_paths, list):
            for lr_path in self.lr_paths:
                if os.path.isfile(lr_path):
                    self.lr_items.append(np.load(lr_path, mmap_mode="r+"))
        else:
            raise ValueError(f"lr_paths is expected to be a list, but {type(lr_paths)}")
        
        if hr_paths is not None and isinstance(self.hr_paths, list):
            for hr_path in self.hr_paths:
                if os.path.isfile(hr_path):
                    self.hr_items.append(np.load(hr_path, mmap_mode="r+"))
        else:
            raise ValueError(f"hr_paths is expected to be a list, but {type(lr_paths)}")
        
        assert len(self.lr_items) == len(self.hr_items)
        print(
            f"File counts: {len(self.lr_items)} LR files & {len(self.hr_items)} HR files found!"
        )
        self.data_count = len(self.lr_items)

        # load stats to normalize
        self.stats_cma = self.get_stats_cma()
        self.stats_ec = self.get_stats_ec()

    def return_stats_cma(self):
        return self.stats_cma
    
    def return_stats_ec(self):
        return self.stats_ec
        
    def get_stats_cma(self):
        """load mean & std of data of CMA variables in self.vnames; SINGLE"""

        with open(
            "/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/nwp/datasets/mean_std_cma.json",
            mode="r",
        ) as f:
            mean_std_cma = json.load(f)
            f.close()
        mean_list, std_list = [], []
        # stat for CMA
        mean_list.append(mean_std_cma["mean"][self.variable_name])
        std_list.append(mean_std_cma["std"][self.variable_name])
        return dict(
            mean=np.array(mean_list, dtype=np.float32),  # C(1)
            std=np.array(std_list, dtype=np.float32),  # C(1)
        )

    def get_stats_ec(self):
        """load mean & std of data of ERA5 variables in self.vnames; PRESSURE-SINGLE

        Raises:
            ValueError: _description_

        Returns:
            _type_: dictionary of mean & std
        """
        with open(
            "/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/nwp/datasets/mean_std_single.json",
            mode="r",
        ) as f:
            mean_std_single = json.load(f)
            f.close()

        mean_list, std_list = [], []
        # stat of FengWu multi-level variables; Order: surface -> low pressure -> high pressure
        # stat for ERA5 surface
        mean_list.append(mean_std_single["mean"][self.variable_name])
        std_list.append(mean_std_single["std"][self.variable_name])

        return dict(
            mean=np.array(mean_list, dtype=np.float32),
            std=np.array(std_list, dtype=np.float32),
        )

    def normalize_max_min(self, data, iy, ix, ip):
        var_list = {
            "u": 0,
            "v": 1,
            "t2m": 2,
            "sp": 3,
            "tp": 4,
        }
        index_var = var_list[self.variable_name]

        max_ = self.max[index_var : index_var + 1, iy : iy + ip, ix : ix + ip]
        min_ = self.min[index_var : index_var + 1, iy : iy + ip, ix : ix + ip]

        return (data - min_) / (max_ - min_ + 1e-6)

    def normalize_data(self, data, data_mean, data_std):
        """expect torch.tensor

        Args:
            data (_type_): _description_
            data_mean (_type_): _description_
            data_std (_type_): _description_

        Returns:
            _type_: _description_
        """
        data -= data_mean[:, None, None]
        data /= data_std[:, None, None]
        return data

    def clean_cma_data(self, cma_data, return_gt=False):
        """process invalid data points in CMA data

        Args:
            frame_data (np.array): CMA data; C, H, W
            return_gt (bool): whether the data is GT or not

        Returns:
            np.array: cleaned data of current frame
        """
        # cma_data = frame_data[c_idx]
        if self.variable_name == "t2m":
            cma_data[cma_data == 9999.0] = 0.0 if return_gt else 0.0
            cma_data = np.clip(cma_data, a_min=213.15, a_max=333.15)
        elif self.variable_name == "r2":
            cma_data = np.clip(cma_data, a_min=0.0, a_max=100.0)
        elif self.variable_name == "u10" or self.variable_name == "v10":
            cma_data[cma_data < -1000.0] = 0.0 if return_gt else 0.0
            cma_data[cma_data > 1000.0] = 0.0 if return_gt else 0.0
            cma_data = np.clip(cma_data, a_min=-1000.0, a_max=1000.0)
        elif self.variable_name == "pre":
            cma_data[cma_data < 0] = 0.0 if return_gt else 0.0  # pre must larger than 0
            cma_data = np.clip(cma_data, a_min=0.0, a_max=300.0)
        elif self.variable_name == "pre6h":
            cma_data[cma_data < 0] = 0.0 if return_gt else 0.0  # pre must larger than 0
            cma_data = np.clip(cma_data, a_min=0.0, a_max=1500.0)

        return cma_data

    def get_patch(self, hr, mask, hr_land, lr_inter):

        # ih_hr, iw_hr = hr.shape[1:]
        # ip = self.patch_size
        # ix = random.randrange(0, iw_hr - ip + 1)
        # iy = random.randrange(0, ih_hr - ip + 1)

        mask_data = mask.float()
        land_data = hr_land.float()

        lr_data = self.normalize_data(
            lr_inter,
            torch.from_numpy(self.stats_ec["mean"]),
            torch.from_numpy(self.stats_ec["std"]),
        )
        ret = {
            "HR": self.normalize_data(
                torch.from_numpy(hr).float(),
                torch.from_numpy(self.stats_cma["mean"]),
                torch.from_numpy(self.stats_cma["std"]),
            ),
            "mask": mask_data,
            "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
            "LAND": land_data,
        }
        
        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        land_01_data = self.land_01  # topological data
        mask_data = self.mask_data  # land_sea mask

        hr_target = np.expand_dims(
            self.hr_items[index][::-1, :], axis=0
        )  # 1, H, W; torch
        hr_target = self.clean_cma_data(hr_target)
        hr_target = np.ascontiguousarray(hr_target)

        if "tp" in self.variable_name:
            lr_inter = interpolate(
                torch.from_numpy(
                    np.expand_dims(
                        self.lr_items[index][120:361, 280:561],
                        axis=0,
                    )
                )
                .float()
                .unsqueeze(0),
                size=(hr_target.shape[1], hr_target.shape[2]),
                mode="bilinear",
            ).squeeze(
                0
            )  # 1, H, W; torch
        else:
            lr_inter = interpolate(
                torch.from_numpy(
                    np.expand_dims(
                        self.lr_items[index][120:361, 280:561],
                        axis=0,
                    )
                )
                .float()
                .unsqueeze(0),
                size=(hr_target.shape[1], hr_target.shape[2]),
                mode="bicubic",
            ).squeeze(0)
        lr_inter = lr_inter.contiguous()
        return self.get_patch(hr_target, mask_data, land_01_data, lr_inter)



class SR3_CNDataset_finetune_patch(torch.utils.data.Dataset):
    def __init__(self, hr_paths, land_paths, mask_paths, lr_paths, var, patch_size):
        index_list = []
        for i, i_start in enumerate(np.arange(0, 400, patch_size)):
            for j, j_start in enumerate(np.arange(0, 700, patch_size)):
                i_end = i_start + patch_size
                j_end = j_start + patch_size
                if i_end > 400:
                    i_end = 400
                    i_start = 400 - 128
                if j_end > 700:
                    j_end = 700
                    j_start = 700 - 128
                index_list.append((i_start, i_end, j_start, j_end))
        self.loc_dict = {}
        for i, index in enumerate(index_list):
            self.loc_dict[str(i)] = index
        var_list = {
            "u": 0,
            "v": 1,
            "t2m": 2,
            "sp": 3,
            "tp": 4,
        }
        index_var = var_list[var]
        self.variable_name = var
        # for path1,path2 in zip(hr_paths,physical_paths):
        #     print(path1,path2)
        self.target_hr = [
            np.load(path, mmap_mode="r+").transpose(0, 3, 1, 2)[
                :, index_var : index_var + 1
            ]
            for path in hr_paths
        ]
        self.target_lr = [
            np.load(path, mmap_mode="r+").transpose(0, 3, 1, 2)[
                :, index_var : index_var + 1
            ]
            for path in lr_paths
        ]

        # [0,2,4,6,8]# 500 zrtuv #[6,8,4,0,2]u v t z r
        self.land_01 = np.expand_dims(np.load(land_paths, mmap_mode="r+"), axis=0)
        self.land_01 = (self.land_01 - np.min(self.land_01)) / (
            np.max(self.land_01) - np.min(self.land_01)
        )
        self.mask_data = np.expand_dims(np.load(mask_paths, mmap_mode="r+"), axis=0)
        self.mask_data = (self.mask_data - np.min(self.mask_data)) / (
            np.max(self.mask_data) - np.min(self.mask_data)
        )
        self.start_indices = [0] * len(self.target_hr)
        self.data_count = 0
        # self.scale=scale
        self.patch_size = patch_size
        for index, memmap in enumerate(self.target_hr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
        self.max = torch.from_numpy(
            np.load(
                "/home/data/downscaling/downscaling_1023/data/train_dataset/max_new_10.npy",
                mmap_mode="r+",
            )
        ).float()
        self.min = torch.from_numpy(
            np.load(
                "/home/data/downscaling/downscaling_1023/data/train_dataset/min_new_10.npy",
                mmap_mode="r+",
            )
        ).float()
        # print(self.max.shape)

    def normalize_max_min(self, data, i_start, i_end, j_start, j_end):
        var_list = {
            "u": 0,
            "v": 1,
            "t2m": 2,
            "sp": 3,
            "tp": 4,
        }
        index_var = var_list[self.variable_name]

        max_ = self.max[index_var : index_var + 1, i_start:i_end, j_start:j_end]
        min_ = self.min[index_var : index_var + 1, i_start:i_end, j_start:j_end]

        # print(max_.max(),max_.min())
        # print((max_-min_).max(),(max_-min_).min())
        return (data - min_) / (max_ - min_ + 1e-6)

    def get_patch(self, hr, mask, hr_land, lr_inter):
        loc = random.randrange(0, len(self.loc_dict))
        i_start, i_end, j_start, j_end = self.loc_dict[str(loc)]
        # ih_hr, iw_hr = hr.shape[1:]
        # ip=self.patch_size
        # ix = random.randrange(0, iw_hr - ip + 1)
        # iy = random.randrange(0, ih_hr - ip + 1)
        mask_data = torch.from_numpy(mask[:, i_start:i_end, j_start:j_end]).float()
        land_data = torch.from_numpy(hr_land[:, i_start:i_end, j_start:j_end]).float()

        if self.variable_name in ["u10", "v10", "t2m", "sp"]:
            lr_data = self.normalize_max_min(
                lr_inter[:, i_start:i_end, j_start:j_end].float(),
                i_start,
                i_end,
                j_start,
                j_end,
            )
            ret = {
                "HR": self.normalize_max_min(
                    torch.from_numpy(hr[:, i_start:i_end, j_start:j_end]).float(),
                    i_start,
                    i_end,
                    j_start,
                    j_end,
                ),
                "mask": mask_data,
                "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
                "LAND": land_data,
            }
        else:
            lr_data = lr_inter[:, i_start:i_end, j_start:j_end].float()
            ret = {
                "HR": torch.from_numpy(hr[:, i_start:i_end, j_start:j_end]).float(),
                "mask": mask_data,
                "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
                "LAND": land_data,
            }

        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]

        land_01_data = self.land_01
        mask_data = self.mask_data
        hr_target = self.target_hr[memmap_index][index_in_memmap] * mask_data
        if self.variable_name == "tp":
            lr_inter = (
                interpolate(
                    torch.from_numpy(
                        np.expand_dims(
                            self.target_lr[memmap_index][index_in_memmap], axis=0
                        )
                    ).float(),
                    scale_factor=10,
                    mode="bilinear",
                ).squeeze(0)
                * mask_data
            )
        else:
            lr_inter = (
                interpolate(
                    torch.from_numpy(
                        np.expand_dims(
                            self.target_lr[memmap_index][index_in_memmap], axis=0
                        )
                    ).float(),
                    scale_factor=10,
                    mode="bicubic",
                ).squeeze(0)
                * mask_data
            )

        return self.get_patch(hr_target, mask_data, land_01_data, lr_inter)


# class SR3_CNDataset_val_new(torch.utils.data.Dataset):
#     def __init__(self,hr_paths,land_paths,mask_paths,lr_paths,var,patch_size,loc):
#         index_list = []
#         for i, i_start in enumerate(np.arange(0, 400, patch_size)):
#             for j, j_start in enumerate(np.arange(0, 700, patch_size)):
#                 i_end = i_start + patch_size
#                 j_end = j_start + patch_size
#                 if i_end > 400:
#                     i_end = 400
#                     i_start=400-128
#                 if j_end > 700:
#                     j_end = 700
#                     j_start=700-128
#                 index_list.append((i_start, i_end, j_start, j_end))
#         loc_dict={}
#         for i,index in enumerate(index_list):
#             loc_dict[str(i)]=index
#         var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
#         index_var=var_list[var]
#         self.loc_index=loc_dict[str(loc)]
#         self.target_hr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in hr_paths]
#         self.target_lr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in lr_paths]

#         #[0,2,4,6,8]# 500 zrtuv #[6,8,4,0,2]u v t z r
#         self.land_01=np.expand_dims(np.load(land_paths, mmap_mode='r+'),axis=0)
#         self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
#         self.start_indices = [0] * len(self.target_hr)
#         self.data_count = 0
#         self.patch_size=patch_size
#         for index, memmap in enumerate(self.target_hr):
#             self.start_indices[index] = self.data_count
#             self.data_count += memmap.shape[0]
#     def get_patch(self,hr,mask,hr_land,lr_inter):
#         i_start,i_end, j_start,j_end=self.loc_index
#         mask_data=torch.from_numpy(mask[:,i_start:i_end, j_start:j_end]).float()
#         land_data=torch.from_numpy(hr_land[:,i_start:i_end, j_start:j_end]).float()
#         lr_data=lr_inter[:,i_start:i_end, j_start:j_end].float()
#         ret = {
#             "HR":torch.from_numpy(hr[:,i_start:i_end, j_start:j_end]).float(),
#             "mask":mask_data,
#             "INTERPOLATED":torch.cat([lr_data,mask_data,land_data],axis=0),
#             "LAND":land_data
#             }
#         return ret

#     def __len__(self):
#         return self.data_count

#     def __getitem__(self, index):
#         memmap_index = bisect(self.start_indices, index) - 1
#         index_in_memmap = index - self.start_indices[memmap_index]

#         land_01_data=self.land_01
#         mask_data=self.mask_data
#         hr_target = self.target_hr[memmap_index][index_in_memmap]*mask_data

#         lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bicubic").squeeze(0)*mask_data

#         return self.get_patch(hr_target,mask_data,land_01_data,lr_inter)


class SR3_CNDataset_all(torch.utils.data.Dataset):
    def __init__(self, land_paths, mask_paths, lr_paths, var):
        var_list = {
            "u": 0,
            "v": 1,
            "t2m": 2,
            "sp": 3,
            "tp": 4,
        }
        index_var = var_list[var]
        self.variable_name = var
        self.target_lr = [
            np.load(path, mmap_mode="r+").transpose(0, 3, 1, 2)[
                :, index_var : index_var + 1
            ]
            for path in lr_paths
        ]
        self.land_01 = np.expand_dims(np.load(land_paths, mmap_mode="r+"), axis=0)
        self.mask_data = np.expand_dims(np.load(mask_paths, mmap_mode="r+"), axis=0)
        self.start_indices = [0] * len(self.target_lr)
        self.data_count = 0
        for index, memmap in enumerate(self.target_lr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
        self.max = torch.from_numpy(
            np.load(
                "/home/data/downscaling/downscaling_1023/data/train_dataset/max_new_10.npy",
                mmap_mode="r+",
            )
        ).float()
        self.min = torch.from_numpy(
            np.load(
                "/home/data/downscaling/downscaling_1023/data/train_dataset/min_new_10.npy",
                mmap_mode="r+",
            )
        ).float()

    def normalize_max_min(self, data):
        var_list = {
            "u": 0,
            "v": 1,
            "t2m": 2,
            "sp": 3,
            "tp": 4,
        }
        index_var = var_list[self.variable_name]

        max_ = self.max[index_var : index_var + 1]
        min_ = self.min[index_var : index_var + 1]

        # print(max_.max(),max_.min())
        # print((max_-min_).max(),(max_-min_).min())
        return (data - min_) / (max_ - min_ + 1e-6)

    def get_patch(self, mask, hr_land, lr_inter):
        mask_data = torch.from_numpy(mask).float()
        land_data = torch.from_numpy(hr_land).float()
        if self.variable_name in ["u", "v", "t2m", "sp"]:
            lr_data = self.normalize_max_min(lr_inter.float())
            ret = {
                "mask": mask_data,
                "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
                "LAND": land_data,
            }
        else:
            lr_data = lr_inter.float()
            ret = {
                "mask": mask_data,
                "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
                "LAND": land_data,
            }

        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        mask_data = self.mask_data
        land_01_data = self.land_01
        if self.variable_name == "tp":
            lr_inter = (
                interpolate(
                    torch.from_numpy(
                        np.expand_dims(
                            self.target_lr[memmap_index][index_in_memmap], axis=0
                        )
                    ).float(),
                    scale_factor=10,
                    mode="bilinear",
                ).squeeze(0)
                * mask_data
            )
        else:
            lr_inter = (
                interpolate(
                    torch.from_numpy(
                        np.expand_dims(
                            self.target_lr[memmap_index][index_in_memmap], axis=0
                        )
                    ).float(),
                    scale_factor=10,
                    mode="bicubic",
                ).squeeze(0)
                * mask_data
            )

        return self.get_patch(mask_data, land_01_data, lr_inter)


class SR3_Dataset_test(torch.utils.data.Dataset):
    def __init__(self, land_paths, mask_paths, lr_paths, var, patch_size, loc):
        index_list = []
        for i, i_start in enumerate(np.arange(0, 400, patch_size)):
            for j, j_start in enumerate(np.arange(0, 700, patch_size)):
                i_end = i_start + patch_size
                j_end = j_start + patch_size
                if i_end > 400:
                    i_end = 400
                    i_start = 400 - 128
                if j_end > 700:
                    j_end = 700
                    j_start = 700 - 128
                index_list.append((i_start, i_end, j_start, j_end))
        loc_dict = {}
        for i, index in enumerate(index_list):
            loc_dict[str(i)] = index
        var_list = {
            "u": 0,
            "v": 1,
            "t2m": 2,
            "sp": 3,
            "tp": 4,
        }
        index_var = var_list[var]
        self.loc_index = loc_dict[str(loc)]
        self.target_lr = [
            np.load(path, mmap_mode="r+").transpose(0, 3, 1, 2)[
                :, index_var : index_var + 1
            ]
            for path in lr_paths
        ]

        # [0,2,4,6,8]# 500 zrtuv #[6,8,4,0,2]u v t z r
        self.land_01 = np.expand_dims(np.load(land_paths, mmap_mode="r+"), axis=0)
        self.mask_data = np.expand_dims(np.load(mask_paths, mmap_mode="r+"), axis=0)
        self.start_indices = [0] * len(self.target_lr)
        self.data_count = 0
        self.patch_size = patch_size
        for index, memmap in enumerate(self.target_lr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

    def get_patch(self, mask, hr_land, lr_inter):
        i_start, i_end, j_start, j_end = self.loc_index
        mask_data = torch.from_numpy(mask[:, i_start:i_end, j_start:j_end]).float()
        land_data = torch.from_numpy(hr_land[:, i_start:i_end, j_start:j_end]).float()
        lr_data = lr_inter[:, i_start:i_end, j_start:j_end].float()
        ret = {
            "mask": mask_data,
            "INTERPOLATED": torch.cat([lr_data, mask_data, land_data], axis=0),
            "LAND": land_data,
        }
        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]

        land_01_data = self.land_01
        mask_data = self.mask_data

        lr_inter = (
            interpolate(
                torch.from_numpy(
                    np.expand_dims(
                        self.target_lr[memmap_index][index_in_memmap], axis=0
                    )
                ).float(),
                scale_factor=10,
                mode="bicubic",
            ).squeeze(0)
            * mask_data
        )

        return self.get_patch(mask_data, land_01_data, lr_inter)


class BigDataset_test(torch.utils.data.Dataset):
    def __init__(self, hr_paths, land_paths, mask_paths):
        self.target_hr = [
            np.load(path, mmap_mode="r+").transpose(0, 3, 1, 2) for path in hr_paths
        ]
        self.land_01 = np.expand_dims(np.load(land_paths, mmap_mode="r+"), axis=0)
        self.mask_data = np.expand_dims(np.load(mask_paths, mmap_mode="r+"), axis=0)
        self.start_indices = [0] * len(self.target_hr)
        self.data_count = 0
        # self.scale=scale
        # self.max_01=np.load(max_paths, mmap_mode='r+')
        for index, memmap in enumerate(self.target_hr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

    def get_patch(self, hr, mask, hr_land):
        mask_data = torch.from_numpy(mask).float()
        land_data = torch.from_numpy(hr_land).float()
        random_index = random.random()
        if random_index < 0:
            ret = {
                "HR": torch.from_numpy(hr).float(),
                "mask": mask_data,
                "INTERPOLATED": torch.cat([mask_data, land_data], axis=0),
                "LAND": land_data,
            }
        else:
            # patch_list=[256]
            ip = 256  # patch_list[random.randint(0, 2)]
            ih_hr, iw_hr = hr.shape[1:]
            ix = random.randrange(0, iw_hr - ip + 1)
            iy = random.randrange(0, ih_hr - ip + 1)
            mask_data = torch.from_numpy(mask[:, iy : iy + ip, ix : ix + ip]).float()
            land_data = torch.from_numpy(hr_land[:, iy : iy + ip, ix : ix + ip]).float()
            ret = {
                "HR": torch.from_numpy(hr[:, iy : iy + ip, ix : ix + ip]).float(),
                "mask": mask_data,
                "INTERPOLATED": torch.cat([mask_data, land_data], axis=0),
                "LAND": land_data,
            }
        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]

        land_01_data = self.land_01
        hr_target = self.target_hr[memmap_index][index_in_memmap]
        # physical=self.data_physical[memmap_index][index_in_memmap]
        mask_data = self.mask_data

        return self.get_patch(hr_target, mask_data, land_01_data)


class BigDataset_cascade_infer(torch.utils.data.Dataset):
    def __init__(self, lr_paths, mask_paths, mask_paths_2x, var):
        variable = {"u10": 0, "v10": 1, "sp": 2, "t2m": 3, "tp": 4}
        idx = variable[var]
        self.data_lr = [
            np.load(path, mmap_mode="r+").transpose(0, 3, 1, 2)[:, idx : idx + 1]
            for path in lr_paths
        ]
        self.mask_data = np.expand_dims(np.load(mask_paths, mmap_mode="r+"), axis=0)
        # 2倍
        self.mask_02 = np.expand_dims(np.load(mask_paths_2x, mmap_mode="r+"), axis=0)
        self.start_indices = [0] * len(self.data_lr)
        self.data_count = 0
        # self.scale=scale
        # self.patch_size=patch_size

        for index, memmap in enumerate(self.data_lr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        lr_data = self.data_lr[memmap_index][index_in_memmap]
        mask_data = torch.from_numpy(self.mask_data).float()
        mask_data_2x = torch.from_numpy(self.mask_02).float()

        inter = interpolate(
            torch.from_numpy(np.expand_dims(lr_data, axis=0)).float(),
            scale_factor=2,
            mode="bicubic",
        ).squeeze(0)
        ret = {
            "LR": torch.from_numpy(lr_data).float(),
            "INTERPOLATED": inter * mask_data_2x,  # /max_
            "mask": mask_data,
        }

        return ret


if __name__ == "__main__":
    import cv2

    # var, patch_size, area = "t2m", 2000, "china"
    var, patch_size, area = "u10", 256, "gansu"
    dataset = SR3_CNDataset_patch_preload(
        lr_root="/mnt/hwfile/ai4earth/wangjiong/weather_data/era5_np_float32",
        hr_root="/mnt/hwfile/ai4earth/wangjiong/weather_data/cma_land",
        land_paths="/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/assets/earth_data/ETOPO_2022_v1_1km_N60_0E70_140_surface.npy",
        mask_paths="/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/assets/earth_data/land_mask_1km_binary.npy",
        var=var,
        patch_size=patch_size,
        area=area,
        year_start="2012-01-01",
        year_end="2012-01-02-23",
        year_freq="6H",
        scale=3,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(len(dataloader))

    for i, data in enumerate(dataloader):
        print(">>>", i)
        print("HR", data["HR"].shape, torch.min(data["HR"]), torch.max(data["HR"]))
        print(
            "LR", 
            data["INTERPOLATED"].shape,
            torch.min(data["INTERPOLATED"][0, 0]),
            torch.max(data["INTERPOLATED"][0, 0]),
        )
        print("mask", data["mask"].shape, torch.min(data["mask"]), torch.max(data["mask"]))
        print("land", data["LAND"].shape, torch.min(data["LAND"]), torch.max(data["LAND"]))

        # import pdb
        # pdb.set_trace()

        hr_array = data["HR"].squeeze().numpy()
        cv2.imwrite(
            f"/mnt/petrelfs/wangjiong/ai4earth/Diffusion_4_downscaling/debug/{area}/{i}_hr_{var}_{area}_{patch_size}.png",
            (hr_array - np.min(hr_array))
            / (np.max(hr_array) - np.min(hr_array))
            * 255.0,
        )
        lr_array = data["INTERPOLATED"][0, 0].squeeze().numpy()
        cv2.imwrite(
            f"/mnt/petrelfs/wangjiong/ai4earth/Diffusion_4_downscaling/debug/{area}/{i}_lr_{var}_{area}_{patch_size}.png",
            (lr_array - np.min(lr_array))
            / (np.max(lr_array) - np.min(lr_array))
            * 255.0,
        )
        land_array = data["LAND"].squeeze().numpy()
        cv2.imwrite(
            f"/mnt/petrelfs/wangjiong/ai4earth/Diffusion_4_downscaling/debug/{area}/{i}_land_{var}_{area}_{patch_size}.png",
            (land_array - np.min(land_array))
            / (np.max(land_array) - np.min(land_array))
            * 255.0,
        )
        mask_array = data["mask"].squeeze().numpy()
        cv2.imwrite(
            f"/mnt/petrelfs/wangjiong/ai4earth/Diffusion_4_downscaling/debug/{area}/{i}_mask_{var}_{area}_{patch_size}.png",
            mask_array * 255.0
            # (mask_array - np.min(mask_array))
            # / (np.max(mask_array) - np.min(mask_array))
            # * 255.0,
        )
