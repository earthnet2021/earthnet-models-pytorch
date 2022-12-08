
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as distrib

from earthnet_models_pytorch.task.shedule import WeightShedule


def setup_loss(args):
    if "pixelwise" in args and args["pixelwise"]:  
        return PixelwiseLoss(args)
    else:
        return BaseLoss(args)


class MaskedDistance(nn.Module):
    """Loss with a cloud mask to compute only on the vegetation pixel"""
    def __init__(self, distance_type="L2", rescale = False):
        super(MaskedDistance, self).__init__()
        self.distance_type = distance_type

    def forward(self, preds, targets, mask):
        assert(preds.shape == targets.shape)
        predsmasked = preds * mask
        targetsmasked = targets * mask  

        if self.distance_type == "L2":
            return F.mse_loss(predsmasked,targetsmasked, reduction='sum')/ ((mask > 0).sum() + 1)
        
        elif self.distance_type == "L1":
            return F.l1_loss(predsmasked,targetsmasked, reduction='sum')/ ((mask > 0).sum() + 1)
        
        elif self.distance_type == "nnse":
            nse = 1 - F.mse_loss(predsmasked,targetsmasked, reduction='sum')/ (torch.var(targetsmasked)**2 * ((mask > 0).sum() + 1))
            return 1 / (2 - nse)


class PixelwiseLoss(nn.Module):
    """Loss for pixelwise model"""
    def __init__(self, setting: dict):
        super().__init__()

        self.distance = MaskedDistance(**setting["args"])
        self.min_lc = 82 if "min_lc" not in setting else setting["min_lc"]  
        self.max_lc = 104 if "max_lc" not in setting else setting["max_lc"]  

    def forward(self, preds, batch, aux, current_step = None):
        logs = {}

        targs = batch["dynamic"][0][:,-preds.shape[1]:,...]  # the number of sample to predict

        lc = batch["landcover"] 
        
        masks = ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool()).type_as(preds).unsqueeze(1).repeat(1, preds.shape[1], 1, 1, 1)  # mask for outlayers using lc threshold 

        masks = torch.where(masks.bool(), (preds >= 0).type_as(masks), masks)

        dist = self.distance(preds, targs, masks)
        
        logs["distance"] = dist 

        loss = dist

        logs["loss"] = loss  
        return loss, logs


class BaseLoss(nn.Module): 
    def __init__(self, setting: dict):
        super().__init__()
        self.distance = MaskedDistance(**setting["args"])
        self.lambda_state = WeightShedule(**setting["state_shedule"]) # speed of the learning rate ? (I think)
        self.lambda_infer = WeightShedule(**setting["inference_shedule"])
        self.lambda_l2_res =  WeightShedule(**setting["residuals_shedule"])
        self.dist_scale = 1 if "dist_scale" not in setting else setting["dist_scale"]
        self.min_lc = setting["min_lc"]
        self.max_lc = setting["max_lc"]


    def forward(self, preds, batch, aux, current_step = None):   
        logs = {}
        if "target" in batch:
            # todo check shape + mask + lc
            targs = batch['target'][:,-preds.shape[1]:,...]
        else:
            targs = batch["dynamic"][0][:,-preds.shape[1]:,...]  # todo legacy, change the previous dataloaders

        if len(batch["dynamic_mask"]) > 0:
            masks = batch["dynamic_mask"][0][:,-preds.shape[1]:,...]
        else:
            masks = None
        if self.ndvi:
            # NDVI computation
            if preds.shape[2] == 1:
                targs = targs[:,:,0,...].unsqueeze(2)
            if masks is not None:
                masks = masks[:,:,0,...].unsqueeze(2)
            lc = batch["landcover"]
            if masks is None:
                masks = ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool()).type_as(preds).unsqueeze(1).repeat(1, preds.shape[1], 1, 1, 1)
            else:
                masks = torch.where(masks.bool(), ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool()).type_as(masks).unsqueeze(1).repeat(1, preds.shape[1], 1, 1, 1), masks)
            
            masks = torch.where(masks.bool(), (preds >= 0).type_as(masks), masks)

        dist = self.distance(preds, targs, masks) 

        logs["distance"] = dist

        loss = dist * self.dist_scale
        
        # for variationnal models
        if aux: 
            variationnal_loss(self, aux, logs, current_step, loss)
        
        logs["loss"] = loss
        return loss, logs


def make_normal_from_raw_params(raw_params, scale_stddev=1, dim=-1, eps=1e-8):
    """
    Creates a normal distribution from the given parameters.

    Parameters
    ----------
    raw_params : torch.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    dim : int104 if "max_lc" not in setting else
        Dimensions of raw_params so that the first half corresponds to the mean, and the second half to the scale.
    eps : float
        Minimum possible value of the final scale parameter.

    Returns
    -------
    torch.distributions.Normal
        Normal distributions with the input mean and eps + softplus(raw scale) * scale_stddev as scale.
    """
    loc, raw_scale = torch.chunk(raw_params, 2, dim)
    assert loc.shape[dim] == raw_scale.shape[dim]
    scale = F.softplus(raw_scale) + eps
    normal = distrib.Normal(loc, scale * scale_stddev)
    return normal

def variationnal_loss(self, aux, logs, current_step, loss):
    if "state_params" in aux:  
        state_normal = make_normal_from_raw_params(aux["state_params"])  # create a normal distribution from the given parametres
        kld_state = distrib.kl_divergence(state_normal, distrib.Normal(0,1)).mean()
        lambda_state = self.lambda_state(current_step)
        loss += lambda_state * kld_state  # why ?
        logs["kld_state"] = kld_state
        logs["lambda_kld_state"] = lambda_state
    if set(["infer_q_params", "infer_p_params"]).issubset(set(aux)):
        assert(len(aux["infer_q_params"]) == len(aux["infer_p_params"]))  # what's mean ?
        for i, (q_params, p_params) in enumerate(zip(aux["infer_q_params"],aux["infer_p_params"])): 
            infer_q_normal = make_normal_from_raw_params(q_params)
            infer_p_normal = make_normal_from_raw_params(p_params)
            kld_infer = distrib.kl_divergence(infer_q_normal, infer_p_normal).mean()
            lambda_infer = self.lambda_infer(current_step)
            loss += lambda_infer * kld_infer
            logs["kld_infer_{}".format(i)] = kld_infer
            logs["lambda_kld_infer"] = lambda_infer
    if "updates" in aux:
        l2_res = torch.norm(aux["updates"], p=2, dim=2).mean()
        lambda_l2_res = self.lambda_l2_res(current_step)
        loss += lambda_l2_res * l2_res
        logs["l2_res"] = l2_res
        logs["lambda_l2_res"] = lambda_l2_res
    return aux, logs, current_step, loss