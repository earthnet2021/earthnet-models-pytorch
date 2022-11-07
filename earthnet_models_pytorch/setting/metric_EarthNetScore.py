
from typing import Tuple, Optional, Sequence, Union

import copy
import multiprocessing

from torchmetrics import Metric
import earthnet as en
import numpy as np
import torch

calc = en.parallel_score.CubeCalculator

def parallel_calc(row):
    cubename, preds, targs, masks, ndvi_preds, ndvi_targs, ndvi_masks = row
    debug_info = {}
    mad, debug_info["MAD"] = calc.MAD(preds, targs, masks)
    
    ols, debug_info["OLS"] = calc.OLS(ndvi_preds, ndvi_targs, ndvi_masks)

    emd, debug_info["EMD"] = calc.EMD(ndvi_preds, ndvi_targs, ndvi_masks)

    ssim, debug_info["SSIM"] = calc.SSIM(preds, targs, masks)
    return {"name": cubename, "MAD": mad, "OLS": ols, "EMD": emd, "SSIM": ssim}#, "debug_info": debug_info}


class EarthNetScore(Metric):
    def __init__(self, dist_sync_on_step=False, compute_on_step = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step = compute_on_step)

        self.add_state("data", default=list())

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
            self._to_sync = self.dist_sync_on_step

            # save context before switch
            self._cache = {attr: getattr(self, attr) for attr in self._defaults.keys()}

            # call reset, update, compute, on single batch
            self.reset()
            self.update(*args, **kwargs)
            for el in self.data:
                el.update({"ENS": self.__harmonic_mean([el["MAD"],el["OLS"],el["EMD"],el["SSIM"]])})
            self._forward_cache = self.data

            # restore context
            for attr, val in self._cache.items():
                setattr(self, attr, val)
            self._to_sync = True
            self._computed = None

            return self._forward_cache

    def update(self, preds, targs):

        preds = copy.deepcopy(preds).detach().cpu().numpy()
        masks = copy.deepcopy(targs["dynamic_mask"][0]).detach().cpu().numpy()[:,-preds.shape[1]:,...]
        cubenames = copy.deepcopy(targs["cubename"])
        targs = copy.deepcopy(targs["dynamic"][0]).detach().cpu().numpy()[:,-preds.shape[1]:,...]

        ndvi_preds = ((preds[:,:,3,...] - preds[:,:,2,...])/(preds[:,:,3,...] + preds[:,:,2,...] + 1e-6))[:,:,np.newaxis,...]
        ndvi_targs = ((targs[:,:,3,...] - targs[:,:,2,...])/(targs[:,:,3,...] + targs[:,:,2,...] + 1e-6))[:,:,np.newaxis,...]
        ndvi_masks = masks[:,:,0,...][:,:,np.newaxis,...]

        preds = np.transpose(preds, (0,3,4,2,1))
        masks = np.transpose(masks, (0,3,4,2,1))
        targs = np.transpose(targs, (0,3,4,2,1))

        ndvi_preds = np.transpose(ndvi_preds, (0,3,4,2,1))
        ndvi_masks = np.transpose(ndvi_masks, (0,3,4,2,1))
        ndvi_targs = np.transpose(ndvi_targs, (0,3,4,2,1))

        rows = [[cubenames[i], preds[i,...], targs[i,...], masks[i,...], ndvi_preds[i,...], ndvi_targs[i,...], ndvi_masks[i,...]] for i in range(preds.shape[0])]

        with multiprocessing.Pool(preds.shape[0]) as p:
            self.data += list(p.map(parallel_calc, rows))

    def compute(self):

        if isinstance(self.data[0], list):
            self.data = [j for d in self.data for j in d]

        data = {}
        for scores in self.data:
            if scores["name"] not in data:
                data[scores["name"]] = [scores]
            else:
                data[scores["name"]].append(scores)

        scores = []
        for cube in data:
            best_sample = self.__get_best_sample(data[cube])
            scores.append([best_sample["MAD"],best_sample["OLS"],best_sample["EMD"],best_sample["SSIM"]])

        scores = np.array(scores, dtype = np.float64)
        mean_scores = np.nanmean(scores, axis = 0).tolist()
        ens = self.__harmonic_mean(mean_scores)

        return {
                "EarthNetScore": ens,
                "Value (MAD)": mean_scores[0],
                "Trend (OLS)": mean_scores[1],
                "Distribution (EMD)": mean_scores[2],
                "Perceptual (SSIM)": mean_scores[3]
            }
            

    def __harmonic_mean(self, vals: Sequence[float]) -> Union[float, None]:
        """

        Calculates the harmonic mean of a list of values, safe for NaNs

        Args:
            vals (list): List of Floats
        Returns:
            float: harmonic mean
        """        
        vals = list(filter(None, vals))
        if len(vals) == 0:
            return None
        else:
            return min(1,len(vals)/sum([1/(v+1e-8) for v in vals]))
                
    def __get_best_sample(self, samples: Sequence[dict]) -> dict:
        """Gets best prediction out of 1 to n predictions. Safe to NaNs

        Args:
            samples (Sequence[dict]): List of dicts with subscores per sample

        Returns:
            dict: dict with subscores of best sample
        """        
        ens = np.array([self.__harmonic_mean([sample["MAD"],sample["OLS"],sample["EMD"],sample["SSIM"]]) for sample in samples], dtype = np.float64)

        try:
            min_idx = np.nanargmax(ens)
        except ValueError:
            min_idx = 0
        
        return samples[min_idx]
