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

class H5Dataset(Dataset):
    def __init__(self, 
                root_dir,
                mode,
                file_list,
                start_idx,
                end_idx,
                variables,
                out_variables,
                max_predict_ranges,
                iter_num,
                transforms,
                output_transforms,
        ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if ("climatology" not in f) and ("times" not in f)]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.max_predict_ranges = max_predict_ranges
        self.iter_num = iter_num
        self._get_files_stats()
        self.transforms = transforms
        self.output_transforms = output_transforms

    def _get_files_stats(self):
        self.n_samples_total = 0
        self.n_shard = len(self.file_list)
        self.h5dataset = [self._open_file(idx) for idx in range(self.n_shard)]
        self.in_chans = len(self.variables)
        self.shape_x = self.h5dataset[0][self.variables[0]].shape[2] #just get rid of one of the pixels
        self.shape_y = self.h5dataset[0][self.variables[0]].shape[3]
        self.n_samples_per_shards = [self.h5dataset[i][self.variables[0]].shape[0] for i in range(self.n_shard)]
        self.n_samples_total = sum(self.n_samples_per_shards)
        # for i in range(self.n_shard):
            # logging.info("Number of examples per shard: {}.".format(self.n_samples_per_shards[i]))
        logging.info("Number of examples: {}. Fields Shape: {} x {} x {}".format(self.n_samples_total, self.in_chans, self.shape_x, self.shape_y))

    def _open_file(self, shard_idx):
        _file = h5py.File(self.file_list[shard_idx], 'r')
        return _file

    def __len__(self):
        return self.n_samples_total // self.iter_num

    def __getitem__(self, global_idx):
        if global_idx < 0:
            global_idx += self.__len__()
        global_idx = int(self.iter_num * global_idx)
        total_idx = 0
        for i in range(self.n_shard):
            if (global_idx >= total_idx) and (global_idx < total_idx + self.n_samples_per_shards[i]):
                break
            else:
                total_idx += self.n_samples_per_shards[i]
        shard_idx = i #which month we are on
        local_idx = ((global_idx - total_idx) % self.n_samples_per_shards[i]) #which sample in that month we are on - determines indices for centering
        
        if self.h5dataset[shard_idx] is None:
            self.h5dataset[shard_idx] = self._open_file(shard_idx)

        if self.mode == "train":
            lead_times = np.random.choice([1, 3, 6, 12, 24], size=1)
        else:
            lead_times = 24 * np.ones(shape=1)
        
        if local_idx >= self.n_samples_per_shards[i] - int(self.iter_num * lead_times[0]) - 1:
            local_idx = self.n_samples_per_shards[i] - int(self.iter_num * lead_times[0]) - 1
        
        #get the data
        inputs = torch.from_numpy(np.concatenate([self.h5dataset[shard_idx][k][local_idx] for k in self.variables], axis=0).astype(np.float32))

        if self.iter_num == 1:
            output_shard_idx = shard_idx
            output_local_idx = local_idx + int(lead_times[0])

            if self.h5dataset[output_shard_idx] is None:
                self.h5dataset[output_shard_idx] = self._open_file(output_shard_idx)

            targets = torch.from_numpy(np.concatenate([self.h5dataset[output_shard_idx][k][output_local_idx] for k in self.out_variables], axis=0).astype(np.float32))

            return self.transforms(inputs), self.output_transforms(targets), \
                torch.from_numpy(lead_times).to(inputs.dtype) / 100, self.variables, self.out_variables, self.iter_num
        
        elif self.iter_num > 1:
            output_shard_idx = shard_idx
            targets = []
            for iter in range(1, self.iter_num + 1):
                output_local_idx = local_idx + int(iter * lead_times[0])

                if self.h5dataset[output_shard_idx] is None:
                    self.h5dataset[output_shard_idx] = self._open_file(output_shard_idx)

                targets.append(torch.from_numpy(np.concatenate([self.h5dataset[output_shard_idx][k][output_local_idx] for k in self.out_variables], axis=0).astype(np.float32)))
        
            targets = [self.output_transforms(target) for target in targets]
            targets = torch.stack(targets, dim=0)

            return self.transforms(inputs), targets, \
                torch.from_numpy(lead_times).to(inputs.dtype) / 100, self.variables, self.out_variables, self.iter_num            

