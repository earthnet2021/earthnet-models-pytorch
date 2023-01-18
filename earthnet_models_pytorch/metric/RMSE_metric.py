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
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
        lc_min=73,
        lc_max=104,
        comp_ndvi=True,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.lc_min = lc_min
        self.lc_max = lc_max

        self.comp_ndvi = comp_ndvi

    @torch.jit.unused
    def forward(self, *args, **kwargs):
        """
        Automatically calls ``update()``. Returns the metric value over inputs if ``compute_on_step`` is True.
        """
        # add current step
        with torch.no_grad():
            self.update(*args, **kwargs)  # accumulate the metrics
        self._forward_cache = None

        if self.compute_on_step:
            kwargs["just_return"] = True
            out_cache = self.update(*args, **kwargs)  # compute and return the rmse
            kwargs.pop("just_return", None)
            return out_cache

    def update(self, preds, targs, just_return=False):
        """Any code needed to update the state given any inputs to the metric."""

        lc = targs["landcover"]

        # Masks
        if len(targs["dynamic_mask"]) > 0:
            masks = targs["dynamic_mask"][0][:, -preds.shape[1] :, 0, ...].unsqueeze(2)
            masks = torch.where(
                masks.bool(),
                ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool())
                .type_as(masks)
                .unsqueeze(1)
                .repeat(1, preds.shape[1], 1, 1, 1),
                masks,
            )  # if error change bool by byte
        else:
            masks = (
                ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool())
                .type_as(preds)
                .unsqueeze(1)
            )  # if error change bool by byte
            if len(masks.shape) == 5:  # spacial dimentions
                masks = masks.repeat(1, preds.shape[1], 1, 1, 1)
            else:
                masks = masks.repeat(1, preds.shape[1], 1)

        # Targets
        targets = targs["dynamic"][0][:, -preds.shape[1] :, ...]
        # NDVI computation
        if targets.shape[2] >= 3 and self.comp_ndvi:
            targets = (
                (targets[:, :, 3, ...] - targets[:, :, 2, ...])
                / (targets[:, :, 3, ...] + targets[:, :, 2, ...] + 1e-6)
            ).unsqueeze(
                2
            ) 
        elif targets.shape[2] >= 3:
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

        if just_return:
            cubenames = targs["cubename"]
            rmse = torch.sqrt(sum_squared_error / n_obs)
            return [
                {"name": cubenames[i], "rmse": rmse[i]} for i in range(len(cubenames))
            ]
        else:
            self.sum_squared_error += sum_squared_error.sum()
            self.total += n_obs.sum()

    def compute(self):
        """
        Computes a final value from the state of the metric.
        Computes mean squared error over state.
        """
        return {"RMSE_Veg": torch.sqrt(self.sum_squared_error / self.total)}


class RMSE_ens21x(RootMeanSquaredError):
    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            lc_min=82,
            lc_max=104,
        )


class RMSE_ens22(RootMeanSquaredError):
    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            lc_min=2,
            lc_max=6,
            comp_ndvi=False,
        )
