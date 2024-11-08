from typing import Any, Dict, Tuple
import copy
import torch
import pickle
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric
from src.utils.weighted_acc_rmse import weighted_rmse_torch, weighted_acc_torch, weighted_mae_torch, activity_torch

class FDVarNetLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        after_scheduler: torch.optim.lr_scheduler,
        mean_path: str,
        std_path: str,
        dict_vars: str,
        loss: object,
        pred_ckpt: str,
    ) -> None:
        """Initialize a `FDVarNetLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        weights_dict = torch.load(self.hparams.pred_ckpt, map_location="cpu")['state_dict']
        load_weights_dict = {k[4:]: v for k, v in weights_dict.items()
                            if self.net.phi_r.state_dict()[k[4:]].numel() == v.numel()}
        self.net.phi_r.load_state_dict(load_weights_dict, strict=True)
        for param in self.net.phi_r.named_parameters():
            param[1].requires_grad = False

        # loss function
        self.criterion = self.hparams.loss
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.dict_vars = dict_vars
        mult = np.ones(len(self.dict_vars))
        for i in range(len(self.dict_vars)):
            mult[i] = self.std[self.dict_vars[i]] * mult[i]
        self.mult = torch.tensor(mult, dtype=torch.float32, requires_grad=False)

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_rmse_u10_best = MinMetric()
        self.val_rmse_v10_best = MinMetric()
        self.val_rmse_z500_best = MinMetric()
        self.val_rmse_t850_best = MinMetric()
        self.val_rmse_q500_best = MinMetric()
        self.val_acc_u10_best = MaxMetric()
        self.val_acc_v10_best = MaxMetric()
        self.val_acc_z500_best = MaxMetric()
        self.val_acc_t850_best = MaxMetric()
        self.val_acc_q500_best = MaxMetric()
        self.val_activity_u10_best = MaxMetric()
        self.val_activity_v10_best = MaxMetric()
        self.val_activity_z500_best = MaxMetric()
        self.val_activity_t850_best = MaxMetric()
        self.val_activity_q500_best = MaxMetric()

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, vars, out_vars) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, y, mask, vars, out_vars)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_rmse_u10_best.reset()
        self.val_rmse_v10_best.reset()
        self.val_rmse_z500_best.reset()
        self.val_rmse_t850_best.reset()
        self.val_rmse_q500_best.reset()
        self.val_acc_u10_best.reset()
        self.val_acc_v10_best.reset()
        self.val_acc_z500_best.reset()
        self.val_acc_t850_best.reset()
        self.val_acc_q500_best.reset()
        self.val_activity_u10_best.reset()
        self.val_activity_v10_best.reset()
        self.val_activity_z500_best.reset()
        self.val_activity_t850_best.reset()
        self.val_activity_q500_best.reset()


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], phase: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        xb, obs, obs_mask, xt, clim, vars, out_vars = batch
        xb = xb.to(xt.device, dtype=xt.dtype)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(xb, requires_grad=True)
            xa = self.forward(state, obs.detach() * obs_mask, obs_mask, vars, out_vars)
            if (phase == 'val') or (phase == 'test'):
                xa = xa.detach()

            loss = self.criterion(xa, xt)
        torch.cuda.empty_cache()
        return loss, xa, xt, clim

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, clims = self.model_step(batch, "train")

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, clims = self.model_step(batch, "val")
        val_rmse = self.mult.to(self.device, dtype=preds.dtype) * weighted_rmse_torch(preds, targets)
        val_rmse = val_rmse.detach()
        val_acc = weighted_acc_torch(preds - clims.to(self.device, dtype=preds.dtype), 
                                     targets - clims.to(self.device, dtype=preds.dtype))
        val_acc = val_acc.detach()
        val_activity = activity_torch(preds, clims.to(self.device, dtype=preds.dtype), self.mult.to(self.device, dtype=preds.dtype))
        val_activity = val_activity.detach()

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_u10", val_rmse[self.dict_vars.index('10m_u_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_v10", val_rmse[self.dict_vars.index('10m_v_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_z500", val_rmse[self.dict_vars.index('geopotential_500')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_t850", val_rmse[self.dict_vars.index('temperature_850')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_q500", val_rmse[self.dict_vars.index('specific_humidity_500')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_u10", val_acc[self.dict_vars.index('10m_u_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_v10", val_acc[self.dict_vars.index('10m_v_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_z500", val_acc[self.dict_vars.index('geopotential_500')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_t850", val_acc[self.dict_vars.index('temperature_850')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_q500", val_acc[self.dict_vars.index('specific_humidity_500')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/activity_u10", val_activity[self.dict_vars.index('10m_u_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/activity_v10", val_activity[self.dict_vars.index('10m_v_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/activity_z500", val_activity[self.dict_vars.index('geopotential_500')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/activity_t850", val_activity[self.dict_vars.index('temperature_850')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/activity_q500", val_activity[self.dict_vars.index('specific_humidity_500')], on_step=False, on_epoch=True, prog_bar=True)
        
        return {'rmse': val_rmse, 'acc': val_acc, 'activity': val_activity, 'preds': preds, 'targets': targets}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_rmse, val_acc, val_activity = 0, 0, 0
        for out in validation_step_outputs:
            val_rmse += out['rmse'] / len(validation_step_outputs)
            val_acc += out['acc'] / len(validation_step_outputs)
            val_activity += out['activity'] / len(validation_step_outputs)

        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        val_rmse_u10 = val_rmse[self.dict_vars.index('10m_u_component_of_wind')]  # get current val rmse of u10
        self.val_rmse_u10_best(val_rmse_u10)  # update best so far val rmse of u10
        val_rmse_v10 = val_rmse[self.dict_vars.index('10m_v_component_of_wind')]  # get current val rmse of v10
        self.val_rmse_v10_best(val_rmse_v10)  # update best so far val rmse of v10
        val_rmse_z500 = val_rmse[self.dict_vars.index('geopotential_500')]  # get current val rmse of z500
        self.val_rmse_z500_best(val_rmse_z500)  # update best so far val rmse of z500
        val_rmse_t850 = val_rmse[self.dict_vars.index('temperature_850')]  # get current val rmse of t850
        self.val_rmse_t850_best(val_rmse_t850)  # update best so far val rmse of t850
        val_rmse_q500 = val_rmse[self.dict_vars.index('specific_humidity_500')]  # get current val rmse of q500
        self.val_rmse_q500_best(val_rmse_q500)  # update best so far val rmse of t850
        val_acc_u10 = val_acc[self.dict_vars.index('10m_u_component_of_wind')]  # get current val acc of u10
        self.val_acc_u10_best(val_acc_u10)  # update best so far val acc of u10
        val_acc_v10 = val_acc[self.dict_vars.index('10m_v_component_of_wind')]  # get current val acc of v10
        self.val_acc_v10_best(val_acc_v10)  # update best so far val acc of v10
        val_acc_z500 = val_acc[self.dict_vars.index('geopotential_500')]  # get current val acc of z500
        self.val_acc_z500_best(val_acc_z500)  # update best so far val acc of z500
        val_acc_t850 = val_acc[self.dict_vars.index('temperature_850')]  # get current val acc of t850
        self.val_acc_t850_best(val_acc_t850)  # update best so far val acc of t850
        val_acc_q500 = val_acc[self.dict_vars.index('specific_humidity_500')]  # get current val acc of q500
        self.val_acc_q500_best(val_acc_q500)  # update best so far val acc of q500
        val_activity_u10 = val_activity[self.dict_vars.index('10m_u_component_of_wind')]  # get current val acc of u10
        self.val_activity_u10_best(val_activity_u10)  # update best so far val acc of u10
        val_activity_v10 = val_activity[self.dict_vars.index('10m_v_component_of_wind')]  # get current val acc of v10
        self.val_activity_v10_best(val_activity_v10)  # update best so far val acc of v10
        val_activity_z500 = val_activity[self.dict_vars.index('geopotential_500')]  # get current val acc of z500
        self.val_activity_z500_best(val_activity_z500)  # update best so far val acc of z500
        val_activity_t850 = val_activity[self.dict_vars.index('temperature_850')]  # get current val acc of t850
        self.val_activity_t850_best(val_activity_t850)  # update best so far val acc of t850
        val_activity_q500 = val_activity[self.dict_vars.index('specific_humidity_500')]  # get current val acc of q500
        self.val_activity_q500_best(val_activity_q500)  # update best so far val acc of q500

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
        self.log("val/rmse_u10_best", self.val_rmse_u10_best.compute(), prog_bar=True)
        self.log("val/rmse_v10_best", self.val_rmse_v10_best.compute(), prog_bar=True)
        self.log("val/rmse_z500_best", self.val_rmse_z500_best.compute(), prog_bar=True)
        self.log("val/rmse_t850_best", self.val_rmse_t850_best.compute(), prog_bar=True)
        self.log("val/rmse_q500_best", self.val_rmse_q500_best.compute(), prog_bar=True)
        self.log("val/acc_u10_best", self.val_acc_u10_best.compute(), prog_bar=True)
        self.log("val/acc_v10_best", self.val_acc_v10_best.compute(), prog_bar=True)
        self.log("val/acc_z500_best", self.val_acc_z500_best.compute(), prog_bar=True)
        self.log("val/acc_t850_best", self.val_acc_t850_best.compute(), prog_bar=True)
        self.log("val/acc_q500_best", self.val_acc_q500_best.compute(), prog_bar=True)
        self.log("val/activity_u10_best", self.val_activity_u10_best.compute(), prog_bar=True)
        self.log("val/acticity_v10_best", self.val_activity_v10_best.compute(), prog_bar=True)
        self.log("val/acticity_z500_best", self.val_activity_z500_best.compute(), prog_bar=True)
        self.log("val/acticity_t850_best", self.val_activity_t850_best.compute(), prog_bar=True)
        self.log("val/acticity_q500_best", self.val_activity_q500_best.compute(), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.inference_mode(False):
            loss, preds, targets, clims = self.model_step(batch, "test")
            test_rmse = self.mult.to(self.device, dtype=preds.dtype) * weighted_rmse_torch(preds, targets)
            test_rmse = test_rmse.detach()
            test_acc = weighted_acc_torch(preds - clims.to(self.device, dtype=preds.dtype), 
                                        targets - clims.to(self.device, dtype=preds.dtype))
            test_acc = test_acc.detach()
            test_activity = activity_torch(preds, clims.to(self.device, dtype=preds.dtype), self.mult.to(self.device, dtype=preds.dtype))
            test_activity = test_activity.detach()

            # update and log metrics
            self.test_loss(loss)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/rmse_u10", test_rmse[self.dict_vars.index('10m_u_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/rmse_v10", test_rmse[self.dict_vars.index('10m_v_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/rmse_z500", test_rmse[self.dict_vars.index('geopotential_500')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/rmse_t850", test_rmse[self.dict_vars.index('temperature_850')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/rmse_q500", test_rmse[self.dict_vars.index('specific_humidity_500')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc_u10", test_acc[self.dict_vars.index('10m_u_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc_v10", test_acc[self.dict_vars.index('10m_v_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc_z500", test_acc[self.dict_vars.index('geopotential_500')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc_t850", test_acc[self.dict_vars.index('temperature_850')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc_q500", test_acc[self.dict_vars.index('specific_humidity_500')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/activity_u10", test_activity[self.dict_vars.index('10m_u_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/activity_v10", test_activity[self.dict_vars.index('10m_v_component_of_wind')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/activity_z500", test_activity[self.dict_vars.index('geopotential_500')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/activity_t850", test_activity[self.dict_vars.index('temperature_850')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/activity_q500", test_activity[self.dict_vars.index('specific_humidity_500')], on_step=False, on_epoch=True, prog_bar=True)
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer([{'params': self.net.model_Grad.parameters()},
                                            {'params': self.net.model_VarCost.parameters()}])
        if self.hparams.scheduler is not None:
            if self.hparams.after_scheduler is not None:
                after_scheduler = self.hparams.after_scheduler(optimizer=optimizer)
                scheduler = self.hparams.scheduler(optimizer=optimizer, after_scheduler=after_scheduler)
            else:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = FDVarNetLitModule(None, None, None, None)
