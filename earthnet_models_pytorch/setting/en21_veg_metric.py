
from typing import Tuple, Optional, Sequence, Union

import copy
import multiprocessing

from pytorch_lightning.metrics import Metric
import numpy as np
import torch


class RootMeanSquaredError(Metric):
    def __init__(self, compute_on_step: bool = False, dist_sync_on_step: bool = False, process_group = None, dist_sync_fn = None,):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    @torch.jit.unused
    def forward(self, *args, **kwargs):
        """
        Automatically calls ``update()``. Returns the metric value over inputs if ``compute_on_step`` is True.
        """
        # add current step
        with torch.no_grad():
            self.update(*args, **kwargs)
        self._forward_cache = None

        if self.compute_on_step:
            kwargs["just_return"] = True
            out_cache = self.update(*args, **kwargs)
            kwargs.pop("just_return", None)
            return out_cache
        
    def update(self, preds, targs, just_return = False):

        masks = targs["dynamic_mask"][0][:,-preds.shape[1]:,0,...].unsqueeze(2)
        lc = targs["landcover"].unsqueeze(1).repeat(1,preds.shape[1],1,1,1)
        masks = torch.where(masks.byte(), ((lc > 63).byte() & (lc < 105).byte()).type_as(masks), masks)
        targets = targs["dynamic"][0][:,-preds.shape[1]:,...]
        targets = ((targets[:,:,3,...] - targets[:,:,2,...])/(targets[:,:,3,...] + targets[:,:,2,...] + 1e-6)).unsqueeze(2)

        sum_squared_error = torch.pow(preds * masks - targets * masks, 2).sum((1,2,3,4))
        n_obs = (masks != 0).sum((1,2,3,4))
        if just_return:
            cubenames = targs["cubename"]
            rmse = torch.sqrt(sum_squared_error / n_obs)
            return [{"name":  cubenames[i], "rmse": rmse[i]} for i in range(len(cubenames))]
        else:
            self.sum_squared_error += sum_squared_error.sum()
            self.total += n_obs.sum()

    def compute(self):
        """
        Computes mean squared error over state.
        """
        return {"RMSE_Veg": torch.sqrt(self.sum_squared_error / self.total)}
