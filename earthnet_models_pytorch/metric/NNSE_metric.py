from torchmetrics import Metric
import numpy as np
import torch


class NormalizedNashSutcliffeEfficiency(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(
        self,
        min_lc: int,
        max_lc: int,
        compute_on_cpu: bool = False,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            # Advanced metric settings
            compute_on_cpu=compute_on_cpu,  # will automatically move the metric states to cpu after calling update, making sure that GPU memory is not filling up.
            dist_sync_on_step=dist_sync_on_step,  # distributed environment, if the metric should synchronize between different devices every time forward is called
            process_group=process_group,  # distributed environment, by default we synchronize across the world i.e. all processes being computed on. Specify exactly what devices should be synchronized over
            dist_sync_fn=dist_sync_fn,  # distributed environment, by default we use torch.distributed.all_gather() to perform the synchronization between devices.
        )

        # Each state variable should be called using self.add_state(...)
        self.add_state("sum_nnse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")

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

        if self.compute_on_step:
            kwargs["just_return"] = True
            out_cache = self.update(*args, **kwargs)  # compute and return the rmse
            kwargs.pop("just_return", None)
            print("out_cache", out_cache)
            return out_cache

    def update(self, preds, targs, just_return=False):
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

        # Metric computation
        if len(masks.shape) == 5:
            nse = 1 - torch.pow(preds * masks - targets * masks, 2).sum(
                (1, 2, 3, 4)
            ) / torch.var(targets * masks, (1, 2, 3, 4))
            nnse = 1 / (2 - nse) + 1e-6
            n_obs = (masks != 0).sum((1, 2, 3, 4))
        else:
            # pixelwise models
            nse = 1 - torch.pow(preds * masks - targets * masks, 2).sum(
                (1, 2)
            ) / torch.var(targets * masks, (1, 2))
            nnse = 1 / (2 - nse)
            n_obs = (masks != 0).sum((1, 2))

        if just_return:
            cubenames = targs["cubename"]
            veg_score = 2 - 1 / (nnse / n_obs)
            return [
                {
                    "name": cubenames[i],
                    "rmse": veg_score[i],
                }  # "rmse" is Legacy, update logging before
                for i in range(len(cubenames))
            ]
        else:
            self.sum_nnse += nnse.sum()
            self.n_obs += n_obs.sum()

    def compute(self):
        """
        Computes a final value from the state of the metric.
        Computes mean squared error over state.
        """
        return {"Veg_score": 2 - 1 / (self.sum_nnse / self.n_obs)}
