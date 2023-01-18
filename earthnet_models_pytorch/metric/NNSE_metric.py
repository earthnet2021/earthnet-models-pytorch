from torchmetrics import Metric
import numpy as np
import torch


class NormalizedNashSutcliffeEfficiency(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(
        self,
        min_lc: int,
        max_lc: int,
        batch_size: int,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            # Advanced metric settings
            dist_sync_on_step=dist_sync_on_step,  # distributed environment, if the metric should synchronize between different devices every time forward is called
            process_group=process_group,  # distributed environment, by default we synchronize across the world i.e. all processes being computed on. Specify exactly what devices should be synchronized over
            dist_sync_fn=dist_sync_fn,  # distributed environment, by default we use torch.distributed.all_gather() to perform the synchronization between devices.
        )

        # Each state variable should be called using self.add_state(...)
        self.add_state("sum_nnse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state(
            "sum_nnse_sample", default=torch.zeros(batch_size), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_obs_sample", default=torch.zeros(batch_size), dist_reduce_fx="sum"
        )

        self.min_lc = min_lc
        self.max_lc = max_lc

    # @torch.jit.unused
    def forward(self, *args, **kwargs):
        """
        Automatically calls ``update()``. Returns the metric value over inputs if ``compute_on_step`` is True.
        """
        # add current step
        with torch.no_grad():
            self.update(*args, **kwargs)  # accumulate the metrics
        self._forward_cache = None

        self.update(*args, **kwargs)
        # print("forward", self.sum_nnse, self.n_obs)
        return self.compute()

    def update(self, preds, targs):
        """Any code needed to update the state given any inputs to the metric."""

        # Masks: to remove the non-vegetation pixels for the computation of the error
        # landcover class with classification of non-vegetation pixel via lc_min and lc_max
        lc = targs["landcover"]

        # if a mask is directly provided (e.g. cloud masking)
        if len(targs["dynamic_mask"]) > 0:
            masks = (
                (targs["dynamic_mask"][0][:, -preds.shape[1] :, ...] < 1.0)
                .bool()
                .type_as(preds)
            )
            # add the non-vegetation pixel to the mask
            masks = torch.where(
                masks.bool(),
                ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool())
                .type_as(masks)
                .unsqueeze(1)
                .repeat(1, preds.shape[1], 1, 1, 1),
                masks,
            )

        # else define a mask using the landcover type
        else:
            masks = (
                ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool())
                .type_as(preds)
                .unsqueeze(1)
            )

            # Reshape the mask to the shape of the data
            if len(masks.shape) == 5:  # cube dimentions
                masks = masks.repeat(1, preds.shape[1], 1, 1, 1)
            else:  # pixelwise models
                masks = masks.repeat(1, preds.shape[1], 1)

        # Targets
        try:
            if "target" in targs:
                targets = targs["target"][:, -preds.shape[1] :, ...]
        except:
            raise Exception("target is not defined")

        # Undetected NaN values in the original data, faulty transmission of Sentinel-2.
        masks = torch.where(
            torch.isnan(targets * masks), torch.zeros(1).bool().type_as(masks), masks
        )
        targetsmasked = torch.where(
            masks.bool(), targets, torch.zeros(1).type_as(targets)
        )
        predsmasked = torch.where(masks.bool(), preds, torch.zeros(1).type_as(preds))

        predsmasked = torch.where(
            torch.isnan(predsmasked), torch.zeros(1).type_as(predsmasked), predsmasked
        )
        targetsmasked = torch.where(
            torch.isnan(targetsmasked),
            torch.zeros(1).type_as(targetsmasked),
            targetsmasked,
        )

        # Metric computation
        if len(masks.shape) == 5:
            mse = torch.pow(predsmasked - targetsmasked, 2).sum((1, 2, 3, 4))
            var = torch.var(targetsmasked, (1, 2, 3, 4))
            nse = 1 - mse / (var + 1e-6)
            nnse = 1 / (2 - nse)
            n_obs = (masks != 0).sum((1, 2, 3, 4))
        else:
            # pixelwise models
            nse = 1 - torch.pow(predsmasked - targetsmasked, 2).sum((1, 2)) / torch.var(
                targetsmasked, (1, 2)
            )
            nnse = 1 / (2 - nse)
            n_obs = (masks != 0).sum((1, 2))

        # Update states
        if n_obs.sum() > 0 and var.sum() > 0:
            self.sum_nnse_sample += nnse
            self.n_obs_sample += n_obs

            self.sum_nnse += nnse.sum()
            self.n_obs += n_obs.sum()


    def compute(self):
        """
        Computes a final value over the state of the metric.
        """
        veg_score = 2 - 1 / (self.sum_nnse_sample / self.n_obs_sample)
        # we are computing a vegetation score, TODO "rmse" is Legacy, update logging before.
        return {"rmse": 2 - 1 / (self.sum_nnse / self.n_obs) + 1e-6}

    def compute_sample(self, targs):
        """
        Computes a final value for each sample over the state of the metric.
        """
        cubenames = targs["cubename"]
        veg_score = 2 - 1 / (self.sum_nnse_sample / self.n_obs_sample) + 1e-6
        # TODO "rmse" is Legacy, update logging before
        return [
            {
                "name": cubenames[i],
                "rmse": veg_score[i],
            }  # "rmse" is Legacy, update logging before
            for i in range(len(cubenames))
        ]
