# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
from typing import Any, Dict, Optional, Tuple
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import re
#
def sort_key(filename):
    # 使用正则表达式匹配文件名中 _ 后的数字
    match = re.search(r'.h5', filename)
    if match:
        # 将匹配到的数字转换为整数
        year = int(filename.split("/")[-1].split("_")[0])
        month = int(filename.split("_")[-1].split(".")[0])
        return int((year-2010)*12 + month)
    else:
        # 如果没有匹配到数字，返回一个默认值，例如0
        return 0

# # 自定义排序函数
# def sort_key(filename):
#     # 使用正则表达式匹配文件名中 _ 后的数字
#     match = re.search(r'.h5', filename)
#     if match:
#         # 将匹配到的数字转换为整数
#         month = int(filename.split("_")[-1].split(".")[0])
#         return month
#     else:
#         # 如果没有匹配到数字，返回一个默认值，例如0
#         return 0

class H5Dataset(Dataset):
    def __init__(self, 
                root_dir,
                mode,
                xb_list,
                lead_time,
                obs_list,
                obsmask_list,
                era5_list,
                start_idx,
                end_idx,
                variables,
                out_variables,
                daw,
                transforms,
                output_transforms,
        ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        start_idx_field = int(start_idx * len(xb_list))
        end_idx_field = int(end_idx * len(xb_list))
        xb_list = xb_list[start_idx_field:end_idx_field]
        # xb_list.sort(key=sort_key)
        era5_list = era5_list[start_idx_field:end_idx_field]
        # era5_list.sort(key=sort_key)
        start_idx_obs = int(start_idx * len(obs_list))
        end_idx_obs = int(end_idx * len(obs_list))
        obs_list = obs_list[start_idx_obs:end_idx_obs]
        # obs_list.sort(key=sort_key)
        obsmask_list = obsmask_list[start_idx_obs:end_idx_obs]
        # obsmask_list.sort(key=sort_key)
        self.lead_time = lead_time
        self.xb_list = [f for f in xb_list if ("climatology" not in f) and ("times" not in f)]
        self.obs_list = [f for f in obs_list if ("climatology" not in f) and ("times" not in f)]
        self.obsmask_list = [f for f in obsmask_list if ("climatology" not in f) and ("times" not in f)]
        self.era5_list = [f for f in era5_list if ("climatology" not in f) and ("times" not in f)]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.daw = daw
        self._get_files_stats()
        self.transforms = transforms
        self.output_transforms = output_transforms

    def _get_files_stats(self):
        self.n_samples_total = 0
        self.n_shard = len(self.xb_list)
        self.h5xb = [self._open_file(idx, self.xb_list) for idx in range(self.n_shard)]
        self.h5obs = [self._open_file(idx, self.obs_list) for idx in range(self.n_shard)]
        self.h5obsmask = [self._open_file(idx, self.obsmask_list) for idx in range(self.n_shard)]
        self.h5era5 = [self._open_file(idx, self.era5_list) for idx in range(self.n_shard)]
        self.in_chans = len(self.variables)
        self.shape_x = self.h5era5[0][self.variables[0]].shape[2] #just get rid of one of the pixels
        self.shape_y = self.h5era5[0][self.variables[0]].shape[3]
        self.n_samples_per_shards = [self.h5era5[i][self.variables[0]].shape[0] for i in range(self.n_shard)]
        logging.info("Number of examples per shard: {}".format(self.n_samples_per_shards))
        self.n_samples_total = sum(self.n_samples_per_shards)
        logging.info("Number of examples: {}. Fields Shape: {} x {} x {}".format(self.n_samples_total, self.in_chans, self.shape_x, self.shape_y))

    def _open_file(self, shard_idx, file_list):
        _file = h5py.File(file_list[shard_idx], 'r')
        return _file

    def __len__(self):
        return int(self.n_samples_total - self.daw - 1)

    def __getitem__(self, global_idx):
        if global_idx < 0:
            global_idx += self.__len__()
        total_idx = 0
        for i in range(self.n_shard):
            if (global_idx >= total_idx) and (global_idx < total_idx + self.n_samples_per_shards[i]):
                break
            else:
                total_idx += self.n_samples_per_shards[i]
        shard_idx = i #which month we are on
        local_idx = ((global_idx - total_idx) % self.n_samples_per_shards[i]) #which sample in that month we are on - determines indices for centering

        if local_idx + self.daw >= self.n_samples_per_shards[i] - 1:
            local_idx = local_idx - self.daw - 1
        
        mask = self.h5obsmask[shard_idx]["mask"][local_idx:local_idx+self.daw].astype(np.float32)
        mask_dict = {}
        for k in self.variables:
            if (k not in ["specific_humidity_50", "specific_humidity_200", "specific_humidity_250"]):# or ("geopotential" not in k):
                mask_dict[k] = mask
            else:
                mask_dict[k] = mask * 0
        
        #get the data
        xb = torch.from_numpy(np.concatenate([self.h5xb[shard_idx][k][local_idx].astype(np.float32) for k in self.variables], axis=0))
        obs = torch.from_numpy(np.nan_to_num(np.concatenate([self.h5obs[shard_idx][k][local_idx:local_idx+self.daw].astype(np.float32) for k in self.variables], axis=1)))
        obsmask = torch.from_numpy(np.nan_to_num(np.concatenate([mask_dict[k] for k in self.variables], axis=1).astype(np.float32)))
        era5 = torch.from_numpy(np.nan_to_num(np.concatenate([self.h5era5[shard_idx][k][local_idx].astype(np.float32) for k in self.variables], axis=0)))
        
        # logging.info("xb error", torch.sqrt(torch.mean((xb - era5)**2, dim=(-2,-1))))

        return self.transforms(xb), self.transforms(obs) * obsmask, obsmask, self.output_transforms(era5), self.variables, self.out_variables      

