import numpy as np
import torch
from torchmetrics import Metric


class NormalizedNashSutcliffeEfficiency(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
        lc_min=10.0,
        lc_max=40.0,
        ndvi_pred_idx=0,
        ndvi_targ_idx=0,
        mask_hq_only=True,
        **kwargs,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("nnse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(1e-6), dist_reduce_fx="sum")

        self.lc_min = lc_min
        self.lc_max = lc_max
        self.ndvi_pred_idx = ndvi_pred_idx
        self.ndvi_targ_idx = ndvi_targ_idx
        self.mask_hq_only = mask_hq_only
        self.compute_on_step = compute_on_step

    # @torch.jit.unused
    # def forward(self, *args, **kwargs):
    #     """
    #     Automatically calls ``update()``. Returns the metric value over inputs if ``compute_on_step`` is True.
    #     """
    #     # add current step
    #     with torch.no_grad():
    #         self.update(*args, **kwargs)  # accumulate the metrics
    #     self._forward_cache = None

    #     if self.compute_on_step:
    #         kwargs["just_return"] = True
    #         out_cache = self.update(*args, **kwargs)  # compute and return the rmse
    #         kwargs.pop("just_return", None)
    #         return out_cache

    def update(self, preds, batch):
        """Any code needed to update the state given any inputs to the metric.

        args:
        preds (torch.Tensor): Prediction tensor of correct length with NDVI at channel index self.ndvi_pred_idx
        batch: (dict): dictionary from dataloader. Expects NDVI target tensor to be under key "dynamic", first entry, channel index self.ndvi_targ_idx.
        """

        t_pred = preds.shape[1]

        lc = batch["landcover"]

        s2_mask = (
            (batch["dynamic_mask"][0][:, -t_pred:, ...] < 1.0).bool().type_as(preds)
        )  # b t c h w

        # lc_mask = ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()).type_as(preds)  # b c h w

        ndvi_targ = batch["dynamic"][0][:, -t_pred:, self.ndvi_targ_idx, ...].unsqueeze(
            2
        )  # b t c h w

        ndvi_pred = preds[:, :, self.ndvi_pred_idx, ...].unsqueeze(2)  # b t c h w

        sum_squared_error = (((ndvi_targ - ndvi_pred) * s2_mask) ** 2).sum(1)  # b c h w
        mean_ndvi_targ = (ndvi_targ * s2_mask).sum(1).unsqueeze(1) / (
            s2_mask.sum(1).unsqueeze(1) + 1e-8
        )  # b t c h w

        sum_squared_deviation = (((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2).sum(
            1
        )  # b c h w

        nse = 1 - sum_squared_error / (sum_squared_deviation + 1e-8)  # b c h w

        if self.mask_hq_only:
            lc_mask = (
                (lc >= self.lc_min).bool()
                & (lc <= self.lc_max).bool()
                & (ndvi_targ.min(1)[0] > 0.0)
                & (s2_mask.sum(1) >= 10)
                & (((sum_squared_deviation / s2_mask.sum(1)) ** 0.5) > 0.1)
            ).type_as(
                preds
            )  # b c h w
        else:
            lc_mask = ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()).type_as(
                preds
            )  # b c h w

        nnse = (1 / (2 - nse)) * lc_mask  # b c h w

        n_obs = lc_mask.sum((1, 2, 3))  # b

        self.current_scores = 2 - 1 / (nnse.sum((1, 2, 3)) / n_obs)  # b

        self.nnse_sum += nnse.sum()
        self.n_obs += n_obs.sum()

    def compute(self):
        """
        Computes a final value from the state of the metric.
        Computes mean squared error over state.
        """
        return {"veg_score": 2 - 1 / (self.nnse_sum / self.n_obs)}


    def compute_batch(self, targs):
        """
        Compute the final RMSE for each sample of a batch over the state of the metric.

        Parameters
        ----------
        targs : dict
            Dictionary containing the target data.

        Returns
        -------
        list
            List of dictionaries containing RMSE scores for each sample in the batch.
        """
        cubenames = targs["cubename"]
        return [
            {
                "name": cubenames[i],
                "veg_score": self.current_scores[i],
            }
            for i in range(len(cubenames))
        ]