import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as distrib
import sys
import numpy as np

from earthnet_models_pytorch.task.shedule import WeightShedule


def setup_loss(args):
    if (
        "pixelwise" in args and args["pixelwise"]
    ):  # TODO check legacy but maybe remove the second argument
        return PixelwiseLoss(args)
    elif "variationnal" in args and args["variationnal"]:
        return PixelwiseLoss(args)
    else:
        return BaseLoss(args)


class MaskedDistance(nn.Module):
    """Loss with a cloud mask to compute only on the vegetation pixel"""

    def __init__(self, distance_type="L2"):
        super(MaskedDistance, self).__init__()
        self.distance_type = distance_type

    def forward(self, preds, targs, mask):
        assert preds.shape == targs.shape

        targsmasked = torch.where(mask.bool(), targs, torch.zeros(1).type_as(targs))
        predsmasked = torch.where(mask.bool(), preds, torch.zeros(1).type_as(preds))

        predsmasked = torch.where(
            torch.isnan(predsmasked), torch.zeros(1).type_as(predsmasked), predsmasked
        )
        targsmasked = torch.where(
            torch.isnan(targsmasked), torch.zeros(1).type_as(targsmasked), targsmasked
        )

        if self.distance_type == "L2":

            return F.mse_loss(predsmasked, targsmasked, reduction="sum") + 1e-6 / (
                (mask > 0).sum() + 1
            )

        elif self.distance_type == "L1":
            return F.l1_loss(predsmasked, targsmasked, reduction="sum") + 1e-6 / (
                (mask > 0).sum() + 1
            )

        elif self.distance_type == "nnse":
            nse = (
                1
                - F.mse_loss(predsmasked, targsmasked, reduction="sum")
                + 1e-6 / (torch.var(targsmasked) ** 2 * ((mask > 0).sum() + 1))
            )
            return 1 / (2 - nse)


class PixelwiseLoss(nn.Module):
    """Loss for pixelwise model"""

    def __init__(self, setting: dict):
        super().__init__()
        self.distance = MaskedDistance(**setting["args"])
        self.min_lc = 82 if "min_lc" not in setting else setting["min_lc"]
        self.max_lc = 104 if "max_lc" not in setting else setting["max_lc"]

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}

        targs = batch["dynamic"][0][
            :, -preds.shape[1] :, ...
        ]  # the number of sample to predict

        lc = batch["landcover"]

        masks = (
            ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool())
            .type_as(preds)
            .unsqueeze(1)
            .repeat(1, preds.shape[1], 1, 1, 1)
        )  # mask for outlayers using lc threshold

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
        # self.scaler = TODO add torch.cuda.amp.GradScaler()
        self.min_lc = setting["min_lc"]
        self.max_lc = setting["max_lc"]

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}

        if "target" in batch:
            # todo check shape + mask + lc
            targs = batch["target"][:, -preds.shape[1] :, ...]

        # TODO legacy, update previous dataloaders then remove the else
        else:
            targs = batch["dynamic"][0][:, -preds.shape[1] :, ...]
            if preds.shape[2] == 1:  # can probably be removed
                targs = targs[:, :, 0, ...].unsqueeze(2)

        # Masks
        if len(batch["dynamic_mask"]) > 0:
            masks = (
                (batch["dynamic_mask"][0][:, -preds.shape[1] :, ...] < 1.0)
                .bool()
                .type_as(preds)
            )
            # Undetected NaN values in the original data, faulty transmission of Sentinel-2.
            masks = torch.where(
                torch.isnan(targs * masks), torch.zeros(1).bool().type_as(masks), masks
            )
        else:
            masks = None

        # Mask non vegetation landcover
        lc = batch["landcover"]
        if masks is None:
            masks = (
                ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool())
                .type_as(preds)
                .unsqueeze(1)
                .repeat(1, preds.shape[1], 1, 1, 1)
            )
        else:
            masks = torch.where(
                masks.bool(),
                ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool())
                .type_as(masks)
                .unsqueeze(1)
                .repeat(1, preds.shape[1], 1, 1, 1),
                masks,
            )

        # Mask non-vegetation pixel (because vegetation pixels have a positive NDVI)
        masks = torch.where(
            masks.bool(),
            (targs >= 0).bool().type_as(masks),
            masks,
        )

        loss = self.distance(preds, targs, masks)
        
        logs["loss"] = loss

        return loss, logs


class VariationnalLoss(nn.Module):
    def __init__(self, setting: dict):
        super().__init__()
        self.distance = MaskedDistance(**setting["args"])
        self.scaler = (
            1 if "dist_scale" not in setting else setting["dist_scale"]
        )  # scaler = torch.cuda.amp.GradScaler()
        self.min_lc = setting["min_lc"]
        self.max_lc = setting["max_lc"]

        # Variationnal models TODO write a separated loss for variationnal model
        self.lambda_state = (
            None
            if "state_shedule" not in setting
            else WeightShedule(**setting["state_shedule"])
        )  # speed of the learning rate ? (I think)
        self.lambda_infer = (
            None
            if "inference_shedule" not in setting
            else WeightShedule(**setting["inference_shedule"])
        )
        self.lambda_l2_res = (
            None
            if "residuals_shedule" not in setting
            else WeightShedule(**setting["residuals_shedule"])
        )

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}
        if "target" in batch:
            # todo check shape + mask + lc
            targs = batch["target"][:, -preds.shape[1] :, ...]

        # TODO legacy, update previous dataloaders then remove the else
        else:
            targs = batch["dynamic"][0][:, -preds.shape[1] :, ...]
            if preds.shape[2] == 1:  # can probably be removed
                targs = targs[:, :, 0, ...].unsqueeze(2)

        # Masks
        if len(batch["dynamic_mask"]) > 0:
            masks = batch["dynamic_mask"][0][:, -preds.shape[1] :, ...]
            if masks is not None and masks.shape[2] > 1:  # Legacy...
                masks = masks[:, :, 0, ...].unsqueeze(2)
        else:
            masks = None

        lc = batch["landcover"]
        if masks is None:
            masks = (
                ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool())
                .type_as(preds)
                .unsqueeze(1)
                .repeat(1, preds.shape[1], 1, 1, 1)
            )
        else:
            masks = torch.where(
                masks.bool(),
                ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool())
                .type_as(masks)
                .unsqueeze(1)
                .repeat(1, preds.shape[1], 1, 1, 1),
                masks,
            )

        masks = torch.where(masks.bool(), (preds >= 0).type_as(masks), masks)

        loss = self.distance(preds, targs, masks)

        logs["distance"] = loss

        if "state_params" in aux:
            state_normal = make_normal_from_raw_params(
                aux["state_params"]
            )  # create a normal distribution from the given parametres
            kld_state = distrib.kl_divergence(state_normal, distrib.Normal(0, 1)).mean()
            lambda_state = self.lambda_state(current_step)
            loss += lambda_state * kld_state
            logs["kld_state"] = kld_state
            logs["lambda_kld_state"] = lambda_state
        if set(["infer_q_params", "infer_p_params"]).issubset(set(aux)):
            assert len(aux["infer_q_params"]) == len(
                aux["infer_p_params"]
            )  # what's mean ?
            for i, (q_params, p_params) in enumerate(
                zip(aux["infer_q_params"], aux["infer_p_params"])
            ):
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
