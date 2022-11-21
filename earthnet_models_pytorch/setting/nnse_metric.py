
import numpy as np
import torch

from torchmetrics import Metric



class NNSEVeg(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(self, compute_on_step: bool = False, dist_sync_on_step: bool = False, process_group = None, dist_sync_fn = None, lc_min = 10., lc_max = 30., ndvi_pred_idx = 0, ndvi_targ_idx = 0):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("nnse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")  
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")

        self.lc_min = lc_min
        self.lc_max = lc_max
        self.ndvi_pred_idx = ndvi_pred_idx
        self.ndvi_targ_idx = ndvi_targ_idx
    
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
        
    def update(self, preds, batch, just_return = False):  
        '''Any code needed to update the state given any inputs to the metric.
        
            args:
            preds (torch.Tensor): Prediction tensor of correct length with NDVI at channel index self.ndvi_pred_idx
            batch: (dict): dictionary from dataloader. Expects NDVI target tensor to be under key "dynamic", first entry, channel index self.ndvi_targ_idx.
        '''

        t_pred = preds.shape[1]

        lc = batch["landcover"]

        s2_mask = batch["dynamic_mask"][0][:,-t_pred:,...]
        s2_lc_mask = torch.where((s2_mask < 1.).bool(), ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()).type_as(s2_mask).unsqueeze(1).repeat(1, preds.shape[1], 1, 1, 1), (s2_mask < 1.).type_as(s2_mask))

        ndvi_targ = batch["dynamic"][0][:, -t_pred:, self.ndvi_targ_idx,...].unsqueeze(2)

        ndvi_pred = preds[:,:,self.ndvi_pred_idx, ...].unsqueeze(2)

        nnse = 1 / (2 - (1 -(((ndvi_targ - ndvi_pred) * s2_lc_mask)**2).sum(1) / ((ndvi_targ - ndvi_targ.mean(1).unsqueeze(1))**2).sum(1)))

        n_obs = (s2_lc_mask != 0).sum((1,2,3,4))

            
        if just_return:
            cubenames = batch["cubename"]
            veg_score = nnse.sum((1,2,3)) / n_obs
            return [{"name":  cubenames[i], "veg_score": veg_score[i]} for i in range(len(cubenames))]
        else:
            self.nnse_sum += nnse.sum()
            self.total += n_obs.sum()

    def compute(self):  
        """
        Computes a final value from the state of the metric.
        Computes mean squared error over state.
        """
        return {"veg_score": self.nnse_sum / self.total}
