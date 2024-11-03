import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.datamodules.assimilate.h5dataset_osse import H5Dataset

def collate_fn(batch):
    xb = torch.stack([batch[i][0] for i in range(len(batch))])
    obs = torch.stack([batch[i][1] for i in range(len(batch))])
    obsmask = torch.stack([batch[i][2] for i in range(len(batch))])
    era5 = torch.stack([batch[i][3] for i in range(len(batch))])
    variables = batch[0][4]
    out_variables = batch[0][5]
    return (
        xb,
        obs,
        obsmask,
        era5,
        [v for v in variables],
        [v for v in out_variables],
    )

class AssimilateDataModule(LightningDataModule):
    def __init__(
            self,
            root_dir: str,
            start_idx: float,
            end_idx: float,
            variables: list,
            out_variables: list,
            full_obs: bool = False,
            daw: int = 0,
            partial: float = 0.1,
            lead_time: int = 24,
            seed : int= 1024,
            batch_size: int = 64,
            num_workers: int = 0,
            shuffle: bool = True,
            pin_memory: bool = True,
            prefetch_factor: int = 2,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if out_variables is None:
            self.hparams.out_variables = variables
        
        self.xb_listers_train = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, "background", f"lead_time_{lead_time}", "train")))
        self.xb_listers_val = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, "background", f"lead_time_{lead_time}", "val")))
        self.xb_listers_test = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, "background", f"lead_time_{lead_time}", "test")))

        self.obs_listers_train = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, "osse/obs", "train")))
        self.obs_listers_val = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, "osse/obs", "val")))
        self.obs_listers_test = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, "osse/obs", "test")))

        self.obsmask_listers_train = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, f"osse/obsmask/partial_{partial}", "train")))
        self.obsmask_listers_val = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, f"osse/obsmask/partial_{partial}", "val")))
        self.obsmask_listers_test = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, f"osse/obsmask/partial_{partial}", "test")))

        self.era5_listers_train = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, "era5_assimilate", "train")))
        self.era5_listers_val = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, "era5_assimilate", "val")))
        self.era5_listers_test = list(dp.iter.FileLister(os.path.join(self.hparams.root_dir, "era5_assimilate", "test")))

        self.train_data, self.val_data, self.test_data = None, None, None

        self.transforms = self.get_normalize(self.hparams.variables)
        self.output_transforms = self.get_normalize(self.hparams.out_variables)

    def get_normalize(self, variables: Optional[Dict] = None):
        if variables is None:
            variables = self.hparams.variables
        root_dir = self.hparams.root_dir
        normalize_mean = dict(np.load(os.path.join(root_dir, "train_forecast", "normalize_mean.npz")))
        mean = []
        for var in variables:
            if var != "total_precipitation":
                mean.append(normalize_mean[var])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(root_dir, "train_forecast", "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in variables])
        data_transforms = transforms.Normalize(normalize_mean, normalize_std)
        return data_transforms

    def get_lat_lon(self):
        # assume different data sources have the same lat and lon coverage
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon

    def _init_fn(self, worker_id):
        # 固定随机数
        np.random.seed(self.hparams.seed + worker_id)

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.train_data and not self.val_data and not self.test_data:
            self.train_data = H5Dataset(
                                root_dir=self.hparams.root_dir,
                                mode="train",
                                xb_list=self.xb_listers_train,
                                lead_time=self.hparams.lead_time,
                                obs_list=self.obs_listers_train,
                                obsmask_list=self.obsmask_listers_train,
                                era5_list=self.era5_listers_train,
                                full_obs=self.hparams.full_obs,
                                start_idx=self.hparams.start_idx,
                                end_idx=self.hparams.end_idx,
                                variables=self.hparams.variables,
                                out_variables=self.hparams.out_variables,
                                daw=self.hparams.daw,
                                transforms=self.transforms,
                                output_transforms=self.output_transforms,
                            )
                        
            self.val_data = H5Dataset(
                                root_dir=self.hparams.root_dir,
                                mode="val",
                                xb_list=self.xb_listers_val,
                                lead_time=self.hparams.lead_time,
                                obs_list=self.obs_listers_val,
                                obsmask_list=self.obsmask_listers_val,
                                era5_list=self.era5_listers_val,
                                full_obs=self.hparams.full_obs,
                                start_idx=self.hparams.start_idx,
                                end_idx=self.hparams.end_idx,
                                variables=self.hparams.variables,
                                out_variables=self.hparams.out_variables,
                                daw=self.hparams.daw,
                                transforms=self.transforms,
                                output_transforms=self.output_transforms,
                            )               

            self.test_data = H5Dataset(
                                root_dir=self.hparams.root_dir,
                                mode="test",
                                xb_list=self.xb_listers_test,
                                lead_time=self.hparams.lead_time,
                                obs_list=self.obs_listers_test,
                                obsmask_list=self.obsmask_listers_test,
                                era5_list=self.era5_listers_test,
                                full_obs=self.hparams.full_obs,
                                start_idx=self.hparams.start_idx,
                                end_idx=self.hparams.end_idx,
                                variables=self.hparams.variables,
                                out_variables=self.hparams.out_variables,
                                daw=self.hparams.daw,
                                transforms=self.transforms,
                                output_transforms=self.output_transforms,
                            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = collate_fn
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass