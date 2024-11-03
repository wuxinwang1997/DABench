#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3, stochastic=False):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)
        self.stochastic = stochastic

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]
        """
        if self.stochastic == True:
            z = torch.randn(input_.shape).to(device)
            z = self.correlate_noise(z)
            z = (z-torch.mean(z))/torch.std(z)
            #z = torch.mul(self.regularize_variance(z),self.correlate_noise(z))
        """
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(input_.device),
                torch.autograd.Variable(torch.zeros(state_size)).to(input_.device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        """
        if self.stochastic == False:
            stacked_inputs = torch.cat((input_, prev_hidden), 1)
        else:
            stacked_inputs = torch.cat((torch.add(input_,z), prev_hidden), 1)
        """

        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

def compute_WeightedLoss(x2,w):
    #  fix normalizing factor ( Sum w = 1 != w~ bool index)
    if len(list(w.size()))>0:
        x2_msk = (x2 * w[None, :, None, None])[:, w>0, ...]
    else:
        x2_msk = x2[:, w==1, ...]
    x2_num = ~x2_msk.isnan() & ~x2_msk.isinf()
    if x2_num.sum() == 0:
        return torch.scalar_tensor(0., device=x2_num.device)
    # loss2 = x2_msk[x2_num].sum()
    loss2 = F.mse_loss(x2_msk[x2_num], torch.zeros_like(x2_msk[x2_num]))
    return loss2

def compute_spatio_temp_weighted_loss(x2, w):
    x2_w = (x2 * w[None, ...])
    non_zeros = (torch.ones_like(x2) * w[None, ...]) == 0.
    x2_num = ~x2_w.isnan() & ~x2_w.isinf() & ~non_zeros
    if x2_num.sum() == 0:
        return torch.scalar_tensor(0., device=x2_num.device)
    loss = F.mse_loss(x2_w[x2_num], torch.zeros_like(x2_w[x2_num]))
    return loss

# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL2Norm, self).__init__()

    def forward(self,x,w,eps=0.):
        loss_ = torch.nansum( x**2 , dim = -1)
        loss_ = torch.nansum( loss_ , dim = -1)
        if len(x.shape) == 5:
            loss_ = torch.nansum(loss_, dim=1)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / w.shape[0] )

        return loss_

class Model_WeightedL1Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()

    def forward(self,x,w,eps):

        loss_ = torch.nansum( torch.sqrt( eps**2 + x**2 ) , dim = -1)
        loss_ = torch.nansum( loss_ , dim = -1)
        if len(x.shape) == 5:
            loss_ = torch.nansum(loss_, dim=1)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / w.shape[0] )

        return loss_

# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdateLSTM(torch.nn.Module):
    def __init__(self,ShapeData,periodicBnd=False,DimLSTM=0,rateDropout=0.,stochastic=False):
        super(model_GradUpdateLSTM, self).__init__()

        with torch.no_grad():
            self.shape     = ShapeData
            if DimLSTM == 0 :
                self.dim_state  = 5*self.shape[0]
            else:
                self.dim_state  = DimLSTM
            self.PeriodicBnd = periodicBnd
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False

        self.convLayer     = self._make_ConvGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)

        self.dropout = torch.nn.Dropout(rateDropout)
        self.stochastic=stochastic
        self.lstm = ConvLSTM2d(self.shape[0],self.dim_state,3,stochastic=self.stochastic)

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(self.dim_state, self.shape[0], 1, padding=0,bias=False))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(torch.nn.Conv2d(self.dim_state, self.shape[0], (1,1), padding=0,bias=False))

        return torch.nn.Sequential(*layers)

    def _make_LSTMGrad(self):
        layers = []

        layers.append(ConvLSTM2d(self.shape[0],self.dim_state,3,stochastic=self.stochastic))

        return torch.nn.Sequential(*layers)

    def forward(self,hidden,cell,grad,gradnorm=1.0):
        # compute gradient
        grad  = grad / gradnorm

        grad  = self.dropout( grad )

        if hidden is None:
            hidden,cell = self.lstm(grad, None)
        else:
            hidden, cell = self.lstm(grad, [hidden,cell])

        grad = self.dropout( hidden )
        grad = self.convLayer( grad )

        return grad, hidden, cell


# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    def __init__(self ,m_NormObs, m_NormPhi, ShapeData, dim_obs=1,dim_obs_channel=0,dim_state=0):
        super(Model_Var_Cost, self).__init__()
        self.dim_obs_channel = dim_obs_channel
        self.dim_obs        = dim_obs
        if dim_state > 0 :
            self.dim_state      = dim_state
        else:
            self.dim_state      = ShapeData[0]

        # parameters for variational cost
        self.alphaObs    = torch.nn.Parameter(torch.Tensor(1. * np.ones((self.dim_obs,1))))
        self.alphaReg    = torch.nn.Parameter(torch.Tensor([1.]))
        if self.dim_obs_channel[0] == 0 :
            self.WObs           = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,ShapeData[0]))))
            self.dim_obs_channel  = ShapeData[0] * np.ones((self.dim_obs,))
        else:
            self.WObs            = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,np.max(self.dim_obs_channel)))))
        self.WReg    = torch.nn.Parameter(torch.Tensor(np.ones(self.dim_state,)))
        self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.dim_obs,))))
        self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))

        self.normObs   = m_NormObs
        self.normPrior = m_NormPhi

    def forward(self, dx, dy):

        loss = self.alphaReg**2 * self.normPrior(dx,self.WReg**2,self.epsReg)
        if self.dim_obs == 1 :
            loss +=  self.alphaObs[0]**2 * self.normObs(dy,self.WObs[0,:]**2,self.epsObs[0])
        else:
            for kk in range(0,self.dim_obs):
                loss +=  (
                    self.alphaObs[kk]**2
                    * self.normObs(
                        dy[kk],
                        self.WObs[kk,0:dy[kk].size(1)]**2,
                        self.epsObs[kk]
                    )
                )

        return loss

class Model_H(torch.nn.Module):
    def __init__(self, shape_data):
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array(shape_data)

    def forward(self, x, y, mask):
        dyout = (x - y) * mask

        return dyout

# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver_Grad_4DVarNN(nn.Module):
    NORMS = {
            'l1': Model_WeightedL1Norm,
            'l2': Model_WeightedL2Norm,
    }
    def __init__(self ,phi_r,mod_H, m_Grad, m_NormObs, m_NormPhi, shape_data, n_iter_grad, daw, dt, stochastic=False):
        super(Solver_Grad_4DVarNN, self).__init__()
        self.phi_r         = phi_r

        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        else:
            m_NormObs = self.NORMS[m_NormObs]()
        if m_NormPhi == None:
            m_NormPhi = Model_WeightedL2Norm()
        else:
            m_NormPhi = self.NORMS[m_NormPhi]()
        self.shape_data = shape_data
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormPhi, shape_data, mod_H.dim_obs, mod_H.dim_obs_channel)
        self.daw = daw
        self.dt = dt
        self.stochastic = stochastic
        self.preds = []
        with torch.no_grad():
            self.n_grad = int(n_iter_grad)

    def forward(self, x, yobs, mask, vars, out_vars):
        return self.solve(x, yobs, mask, vars, out_vars)

    def solve(self, x_0, obs, mask, vars, out_vars, hidden=None, cell=None, normgrad_=None):
        x_k = torch.mul(x_0,1.)
        x_k_plus_1 = None
        for _ in range(self.n_grad):
            x_k_plus_1, hidden, cell, normgrad_ = self.solver_step(x_k, obs, mask, vars, out_vars, hidden, cell, normgrad_)
            x_k = torch.mul(x_k_plus_1,1.)

        return x_k_plus_1

    def solver_step(self, x_k, obs, mask, vars, out_vars, hidden=None, cell=None, normgrad=None):
        _, var_cost_grad = self.var_cost(x_k, obs, mask, vars, out_vars)
        if normgrad is None:
            normgrad_= torch.sqrt(torch.mean(var_cost_grad**2, dim=(1,2,3), keepdim=True))
            normgrad_ = torch.where(torch.isnan(normgrad_), 1, normgrad_)
            normgrad_ = torch.where(normgrad_ == 0, 1, normgrad_)
            normgrad_ = torch.where(torch.isinf(normgrad_), 1, normgrad_)
        else:
            normgrad_= normgrad
            normgrad_ = torch.sqrt(torch.mean(var_cost_grad ** 2, dim=(1, 2, 3), keepdim=True))
            normgrad_ = torch.where(torch.isnan(normgrad_), 1, normgrad_)
            normgrad_ = torch.where(normgrad_ == 0, 1, normgrad_)
            normgrad_ = torch.where(torch.isinf(normgrad_), 1, normgrad_)
        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.n_grad
        x_k_plus_1 = x_k - grad
        return x_k_plus_1, hidden, cell, normgrad_

    def var_cost(self, xb, yobs, mask, vars, out_vars):
        preds = self.forecast(xb, yobs, vars, out_vars)
        
        dy = self.model_H(preds, yobs, mask)
        dx = xb - self.preds[1]

        loss = self.model_VarCost(dx, dy)
 
        var_cost_grad = torch.autograd.grad(loss, xb, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
        # var_cost_grad = torch.where(torch.isnan(var_cost_grad), 0, var_cost_grad)
        # var_cost_grad = torch.where(torch.isinf(var_cost_grad), 0, var_cost_grad)
        return loss, var_cost_grad
    
    def forecast(self, x0, yobs, vars, out_vars):
        self.preds = []
        self.preds.append(x0)
        for i in range(1, yobs.shape[1] // self.dt):
            if ((24 // self.dt) > 0) and (i % (24 // self.dt)) == 0:
                # Call the model for 24h forecast
                self.preds.append(self.phi_r(self.preds[i - 24 // self.dt],
                                             torch.from_numpy(24 * np.ones((1,1))).to(x0.device, dtype=torch.float32) / 100,
                                             vars,
                                             out_vars))
            elif ((12 // self.dt) > 0) and (i % (12 // self.dt)) == 0:
                # Call the model for 24h forecast
                self.preds.append(self.phi_r(self.preds[i - 12 // self.dt],
                                             torch.from_numpy(12 * np.ones((1,1))).to(x0.device, dtype=torch.float32) / 100,
                                             vars,
                                             out_vars))
            elif ((6 // self.dt) > 0) and (i % (6 // self.dt)) == 0:
                # Call the model for 24h forecast
                self.preds.append(self.phi_r(self.preds[i - 6 // self.dt],
                                             torch.from_numpy(6 * np.ones((1,1))).to(x0.device, dtype=torch.float32) / 100,
                                             vars,
                                             out_vars))
            elif ((3 // self.dt) > 0) and (i % (3 // self.dt)) == 0:
                # Call the model for 24h forecast
                self.preds.append(self.phi_r(self.preds[i - 3 // self.dt],
                                             torch.from_numpy(3 * np.ones((1,1))).to(x0.device, dtype=torch.float32) / 100,
                                             vars,
                                             out_vars))
            elif ((1 // self.dt) > 0) and (i % (1 // self.dt)) == 0:
                # Call the model for 24h forecast
                self.preds.append(self.phi_r(self.preds[i - 1 // self.dt],
                                             torch.from_numpy(1 * np.ones((1,1))).to(x0.device, dtype=torch.float32) / 100,
                                             vars,
                                             out_vars))
        preds = torch.stack(self.preds, dim=1)
        return preds