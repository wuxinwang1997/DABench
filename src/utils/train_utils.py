from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
# torch.set_printoptions(profile="full")
import torch.nn as nn
import numpy as np
import copy

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class WeightedL1Loss(nn.Module):
    def __init__(self, sum=False):
        super(WeightedL1Loss, self).__init__()
        self.sum = sum

    def lat(self, j: torch.Tensor, num_lat: int) -> torch.Tensor:
        return 90 - j * 180 / float(num_lat - 1)

    def latitude_weighting_factor(self, j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
        return num_lat * torch.cos(3.1416 / 180. * self.lat(j, num_lat)) / s

    def weighted_l1loss_channels(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
        num_lat = pred.shape[2]
        lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
        s = torch.sum(torch.cos(3.1416 / 180. * self.lat(lat_t, num_lat)))
        weight = torch.reshape(self.latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
        result = torch.mean(weight * torch.abs(pred - target), dim=(-1, -2))
        return result

    def weighted_l1loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        result = self.weighted_l1loss_channels(pred, target)
        return torch.mean(result, dim=0)

    def forward(self, pred, target):
        w_l1loss = self.weighted_l1loss(pred, target)
        if self.sum:
            return torch.sum(w_l1loss)
        else:
            return torch.mean(w_l1loss)

class GeostrophicLoss(nn.Module):
    def __init__(self, list_vars, lat_path, lon_path, mean_path, std_path, alpha):
        super(GeostrophicLoss, self).__init__()
        # 定义常数
        self.omega = 7.2921e-5  # 地球自转角速度
        self.g = 9.80665  # 重力加速度
        lat = np.load(lat_path)
        lats = np.reshape(lat, (1, 1, lat.shape[0], 1))
        self.lats = torch.tensor(lats)
        self.mask = torch.abs(self.lats)
        self.mask = torch.where((self.mask >= 30) & (self.mask <= 70), 1, 0)
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.list_vars = list_vars
        mean = np.ones((1, len(self.list_vars),1,1))
        std = np.ones((1, len(self.list_vars),1,1))
        for i in range(len(self.list_vars)):
            mean[0,i] = self.mean[self.list_vars[i]] * mean[0,i]
            std[0,i] = self.std[self.list_vars[i]] * std[0,i]
        self.mean = torch.tensor(mean, dtype=torch.float32, requires_grad=False)
        self.std = torch.tensor(std, dtype=torch.float32, requires_grad=False)
        self.alpha = alpha
    
    def geowind(self, z):
        f = 2 * self.omega * torch.sin(torch.deg2rad(self.lats.to(z.device, dtype=z.dtype)))
        # 计算梯度
        # 假设您有geopotential数据的经度和纬度步长
        dx = self.lats[0, 0, 2, 0] - self.lats[0, 0, 0, 0]  # 经度步长
        # 计算经度方向的梯度（沿纬度变化）
        dphidx = (z[:, :, 1:-1, 2:] - z[:, :, 1:-1, :-2]) * self.std[:,4:17].to(z.device, dtype=z.dtype) / dx.to(z.device,  dtype=z.dtype)
        dphidx = dphidx / (111e3 * torch.cos(torch.deg2rad(self.lats[:,:,1:-1].to(z.device,  dtype=z.dtype))))
        
        # 计算纬度方向的梯度（沿经度变化）
        dphidy = (z[:, :, 2:, 1:-1] - z[:, :, :-2, 1:-1]) * self.std[:,4:17].to(z.device, dtype=z.dtype) / dx.to(z.device,  dtype=z.dtype)
        dphidy = dphidy / 111e3
        
        # 计算风速风向
        gv = ((( 1 / f[:,:,1:-1,:]) * dphidx) - self.mean[:,30:43].to(z.device, dtype=z.dtype)) / self.std[:,30:43].to(z.device, dtype=z.dtype)
        gu = (((-1 / f[:,:,1:-1,:]) * dphidy) - self.mean[:,17:30].to(z.device, dtype=z.dtype)) / self.std[:,17:30].to(z.device, dtype=z.dtype)
        
        return gu, gv
        
    def forward(self, pred, target):
        z_p = torch.nn.functional.pad(pred[:,4:17], (1,1,0,0), mode="circular")
        gu_p, gv_p = self.geowind(z_p)

        z_t = torch.nn.functional.pad(target[:,4:17], (1,1,0,0), mode="circular")
        gu_t, gv_t = self.geowind(z_t)

        loss_pred = torch.nn.functional.l1_loss(pred, target)
        loss_geostrophic_wind_p = torch.nn.functional.l1_loss(self.mask[:,:,1:-1].to(pred.device) * gu_p, self.mask[:,:,1:-1].to(pred.device) * gu_t) \
            + torch.nn.functional.l1_loss(self.mask[:,:,1:-1].to(pred.device) * gv_p, self.mask[:,:,1:-1].to(pred.device) * gv_t)
        loss = loss_pred + self.alpha * loss_geostrophic_wind_p 
        
        return loss
        