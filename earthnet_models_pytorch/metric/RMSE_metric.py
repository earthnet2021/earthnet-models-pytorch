from typing import Tuple, Optional, Sequence, Union

import copy
import multiprocessing
import sys

from torchmetrics import Metric
import numpy as np
import torch


class RootMeanSquaredError(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(
        self,
        lc_min: int,
        lc_max: int,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            # Advanced metric settings
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,  # distributed environment, if the metric should synchronize between different devices every time forward is called
            process_group=process_group,  # distributed environment, by default we synchronize across the world i.e. all processes being computed on. Specify exactly what devices should be synchronized over
            dist_sync_fn=dist_sync_fn,  # distributed environment, by default we use torch.distributed.all_gather() to perform the synchronization between devices.
        )

        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("current_scores", default=torch.tensor(0), dist_reduce_fx=None)

        self.lc_min = lc_min
        self.lc_max = lc_max
        print(
                f"Using Masked RootMeanSquaredError metric Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )

    # def forward(self, *args, **kwargs):
    #     # accumulate the metric
    #     with torch.no_grad():
    #       self.update(*args, **kwargs)
    #     # self._forward_cache = None # I don't know for what is this, I think is not anymore necessary?
    #     return self.compute()

    def update(self, preds, targs):
        """Any code needed to update the state given any inputs to the metric."""

        # Masks on the non vegetation pixels
        # Dynamic cloud mask available
        if len(targs["dynamic_mask"]) > 0:
            masks = targs["dynamic_mask"][0][:, -preds.shape[1] :, 0, ...].unsqueeze(2)

            # Landcover mask
            lc = targs["landcover"]

            if lc.ndim == 5:  # En23 has a weird dimension, temporary patch
                lc = lc[..., 0]

            masks = torch.where(
                masks.bool(),
                ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool())
                .type_as(masks)
                .unsqueeze(1)
                .repeat(1, preds.shape[1], 1, 1, 1),
                masks,
            )

        else:
            masks = (
                ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool())
                .type_as(preds)
                .unsqueeze(1)
            )
            # TODO what is that?
            if len(masks.shape) == 5:  # spacial dimentions
                masks = masks.repeat(1, preds.shape[1], 1, 1, 1)
            else:
                masks = masks.repeat(1, preds.shape[1], 1)

        # Targets
        targets = targs["dynamic"][0][:, -preds.shape[1] :, ...]
        targets = targets[:, :, 0, ...].unsqueeze(2)

        # MSE computation
        if len(masks.shape) == 5:
            sum_squared_error = torch.pow(preds * masks - targets * masks, 2).sum(
                (1, 2, 3, 4)
            )
            n_obs = (masks != 0).sum((1, 2, 3, 4))
        else:
            sum_squared_error = torch.pow(preds * masks - targets * masks, 2).sum(
                (1, 2)
            )
            n_obs = (masks != 0).sum((1, 2))

        # Update the states variables
        self.current_scores = sum_squared_error / n_obs
        self.sum_squared_error += sum_squared_error.sum()
        self.total += n_obs.sum()

    def compute(self):
        """
        Computes a final value from the state of the metric.
        Computes mean squared error over state.
        """
        return {"RMSE_Veg": torch.sqrt(self.sum_squared_error / self.total)}

    def compute_batch(self, targs):
        """
        Computes a final value for each sample of a batch over the state of the metric.
        """
        cubenames = targs["cubename"]
        return [
            {
                "name": cubenames[i],
                "rmse": self.current_scores[i],
            }  # "rmse" is Legacy, update logging before
            for i in range(len(cubenames))
        ]

