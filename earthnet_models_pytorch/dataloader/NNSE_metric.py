from torchmetrics import Metric
import numpy as np
import torch



class NormalizedNashSutcliffeEfficiency(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(self, lc_min: int, lc_max: int, compute_on_cpu: bool = False, dist_sync_on_step: bool = False, process_group = None, dist_sync_fn = None):
        super().__init__(
            # Advanced metric settings 
            compute_on_cpu=compute_on_cpu, # will automatically move the metric states to cpu after calling update, making sure that GPU memory is not filling up. 
            dist_sync_on_step=dist_sync_on_step, # distributed environment, if the metric should synchronize between different devices every time forward is called
            process_group=process_group, # distributed environment, by default we synchronize across the world i.e. all processes being computed on. Specify exactly what devices should be synchronized over
            dist_sync_fn=dist_sync_fn, # distributed environment, by default we use torch.distributed.all_gather() to perform the synchronization between devices. 
        )

        # Each state variable should be called using self.add_state(...)
        self.add_state("sum_nnse_error", default=torch.tensor(0.0), dist_reduce_fx="sum")  
        self.add_state("veg_score", default=torch.tensor(0), dist_reduce_fx="sum")

        self.lc_min = lc_min
        self.lc_max = lc_max
    
    @torch.jit.unused
    def forward(self, *args, **kwargs):
        """
        forward serves the dual purpose of both returning the metric on the current data and updating the internal metric state for accumulating over multiple batches.
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
        
        
    def update(self, preds, targs, just_return = False):  
        '''Any code needed to update the state given any inputs to the metric.'''
        lc = targs["landcover"]

        # Masks
        if len(targs["dynamic_mask"]) > 0:
            masks = targs["dynamic_mask"][0][:,-preds.shape[1]:,0,...].unsqueeze(2)
            masks = torch.where(masks.bool(), ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()).type_as(masks).unsqueeze(1).repeat(1, preds.shape[1], 1, 1, 1), masks)  # if error change bool by byte
        else:
            masks = ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()).type_as(preds).unsqueeze(1) # if error change bool by byte
            if len(masks.shape) == 5:  # spacial dimentions
                masks = masks.repeat(1, preds.shape[1], 1, 1, 1)
            else:
                masks = masks.repeat(1, preds.shape[1], 1)
        
        # Targets
        targets = targs["dynamic"][0][:,-preds.shape[1]:,...]
        if targets.shape[2] >= 3 and self.comp_ndvi:
            targets = ((targets[:,:,3,...] - targets[:,:,2,...])/(targets[:,:,3,...] + targets[:,:,2,...] + 1e-6)).unsqueeze(2)  # NDVI computation
        elif targets.shape[2] >= 3:
            targets = targets[:,:,0,...].unsqueeze(2)
        
        # MSE computation    
        if len(masks.shape) == 5:
            sum_squared_error = torch.pow(preds * masks - targets * masks, 2).sum((1,2,3,4))  
            n_obs = (masks != 0).sum((1,2,3,4))
        else:
            sum_squared_error = torch.pow(preds * masks - targets * masks, 2).sum((1,2))
            n_obs = (masks != 0).sum((1,2))
            
        if just_return:
            cubenames = targs["cubename"]
            rmse = torch.sqrt(sum_squared_error / n_obs)
            return [{"name":  cubenames[i], "rmse": rmse[i]} for i in range(len(cubenames))]
        else:
            self.sum_squared_error += sum_squared_error.sum()
            self.total += n_obs.sum()

    def compute(self):  
        """
        Computes a final value from the state of the metric.
        Computes mean squared error over state.
        """
        return {"RMSE_Veg": torch.sqrt(self.sum_squared_error / self.total)}


class NNSE_ens21x(NormalizedNashSutcliffeEfficiency):

    def __init__(self, compute_on_cpu: bool = False, dist_sync_on_step: bool = False, process_group = None, dist_sync_fn = None):
        super().__init__(
            compute_on_cpu=compute_on_cpu,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            lc_min = 82,
            lc_max = 104
        )

class NNSE_ens22(NormalizedNashSutcliffeEfficiency):

    def __init__(self, compute_on_cpu: bool = False, dist_sync_on_step: bool = False, process_group = None, dist_sync_fn = None):
        super().__init__(
            compute_on_cpu=compute_on_cpu,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            lc_min = 2,
            lc_max = 6
        )