import sys

import torch
import torch.distributions as distrib
import torch.nn.functional as F
from torch import nn


def make_normal_from_raw_params(raw_params, scale_stddev=1, dim=-1, eps=1e-8):
    """
    Creates a normal distribution from the given parameters.

    Parameters
    ----------
    raw_params : torch.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    dim : int
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


class MaskedLoss(nn.Module):
    def __init__(self, distance_type="L2", rescale=False):
        super(MaskedLoss, self).__init__()
        self.distance_type = distance_type

    def forward(self, preds, targets, mask):
        assert preds.shape == targets.shape
        predsmasked = preds * mask
        targetsmasked = targets * mask

        if self.distance_type == "L2":
            return F.mse_loss(predsmasked, targetsmasked, reduction="sum") / (
                (mask > 0).sum() + 1
            )  # (input, target, reduction: Specifies the reduction to apply to the output)
        elif self.distance_type == "L1":
            return F.l1_loss(predsmasked, targetsmasked, reduction="sum") / (
                (mask > 0).sum() + 1
            )


LOSSES = {"masked": MaskedLoss}


class PixelwiseLoss(nn.Module):
    def __init__(self, setting: dict):
        super().__init__()

        self.distance = LOSSES[setting["name"]](**setting["args"])
        self.lc_min = 82 if "lc_min" not in setting else setting["lc_min"]
        self.lc_max = 104 if "lc_max" not in setting else setting["lc_max"]
        print(
            f"Using Masked L2 NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
        )

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}

        targs = batch["dynamic"][0][
            :, -preds.shape[1] :, ...
        ]  # the number of sample to predict

        lc = batch["landcover"]

        masks = (
            ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool())
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


class MaskedPixelwiseLoss(nn.Module):
    def __init__(
        self,
        lc_min=None,
        lc_max=None,
        context_length=None,
        target_length=None,
        ndvi_pred_idx=0,
        ndvi_targ_idx=0,
        pred_mask_value=None,
        scale_by_std=False,
        extra_aux_loss_term=None,
        extra_aux_loss_weight=1,
        setting=None,
        **kwargs,
    ):
        super().__init__()

        self.lc_min = (
            lc_min if lc_min else None
        )  # landcover boudaries of vegetation (to select only pixel with vegetation)
        self.lc_max = lc_max if lc_max else None
        self.use_lc = lc_min & lc_max
        if not self.use_lc:
            print(
                f"WARNING. The boundaries of the landcover map are not definite. Loss calculated on all pixels including non-vegetation pixels."
            )
        self.context_length = context_length
        self.target_length = target_length
        self.ndvi_pred_idx = ndvi_pred_idx  # index of the NDVI band
        self.ndvi_targ_idx = ndvi_targ_idx  # index of the NDVI band
        self.pred_mask_value = pred_mask_value
        self.scale_by_std = scale_by_std
        if self.scale_by_std:
            print(
                f"Using Masked L2/Std NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )
        else:
            print(
                f"Using Masked L2 NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )

        self.extra_aux_loss_term = extra_aux_loss_term
        self.extra_aux_loss_weight = extra_aux_loss_weight
        self.setting = setting

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}
        # Targets
        targets = batch["dynamic"][0][
            :, self.context_length : self.context_length + self.target_length, 0, ...
        ].unsqueeze(2)

        # Masks on the non vegetation pixels
        # Dynamic cloud mask available
        if len(batch["dynamic_mask"]) > 0:
            s2_mask = (
                (
                    batch["dynamic_mask"][0][
                        :,
                        self.context_length : self.context_length + self.target_length,
                        self.ndvi_targ_idx,
                        ...,
                    ].unsqueeze(2)
                    < 1.0
                )
                .bool()
                .type_as(preds)
            )

        # Landcover mask
        lc = batch["landcover"]
        if self.setting == "en23":
            lc_bool = (lc <= self.lc_min).bool() | (lc >= self.lc_max).bool()
        else:
            lc_mask = (lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()

        lc_mask = (
            lc_bool.type_as(s2_mask).unsqueeze(1).repeat(1, preds.shape[1], 1, 1, 1)
        )
        mask = s2_mask * lc_mask

        # MSE computation
        sum_squared_error = torch.pow((preds - targets) * mask, 2).sum()
        n_obs = mask.sum()  # sum of pixel with vegetation
        loss = sum_squared_error / (n_obs + 1e-8)
        logs["loss"] = loss
        return loss, logs


class MaskedL2NDVILoss(nn.Module):
    def __init__(
        self,
        lc_min=None,
        lc_max=None,
        context_length=None,
        target_length=None,
        ndvi_pred_idx=0,
        ndvi_targ_idx=0,
        pred_mask_value=None,
        scale_by_std=False,
        weight_by_std=False,
        extra_aux_loss_term=None,
        extra_aux_loss_weight=1,
        mask_hq_only=False,
        **kwargs,
    ):
        super().__init__()

        self.lc_min = (
            lc_min if lc_min else None
        )  # landcover boudaries of vegetation (to select only pixel with vegetation)
        self.lc_max = lc_max if lc_max else None
        self.use_lc = lc_min & lc_max
        if not self.use_lc:
            print(
                f"WARNING. The boundaries of the landcover map are not definite. Loss calculated on all pixels including non-vegetation pixels."
            )
        self.context_length = context_length
        self.target_length = target_length
        self.ndvi_pred_idx = ndvi_pred_idx  # index of the NDVI band
        self.ndvi_targ_idx = ndvi_targ_idx  # index of the NDVI band
        self.pred_mask_value = pred_mask_value
        self.scale_by_std = scale_by_std
        self.weight_by_std = weight_by_std
        if self.scale_by_std:
            print(
                f"Using Masked L2/Std NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )
        else:
            print(
                f"Using Masked L2 NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )

        self.extra_aux_loss_term = extra_aux_loss_term
        self.extra_aux_loss_weight = extra_aux_loss_weight
        self.mask_hq_only = mask_hq_only

    def forward(self, preds, batch, aux, current_step=None):
        # Mask
        # Cloud mask
        s2_mask = (
            (
                batch["dynamic_mask"][0][
                    :,
                    self.context_length : self.context_length + self.target_length,
                    ...,
                ]
                < 1.0
            )
            .bool()
            .type_as(preds)
        )  # b t c h w

        # Landcover mask
        lc = batch["landcover"]
        lc_mask = ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()).type_as(
            preds
        )  # b c h w
        ndvi_targ = batch["dynamic"][0][
            :,
            self.context_length : self.context_length + self.target_length,
            self.ndvi_targ_idx,
            ...,
        ].unsqueeze(
            2
        )  # b t c h w

        ndvi_pred = preds[:, -ndvi_targ.shape[1] :, self.ndvi_pred_idx, ...].unsqueeze(
            2
        )  # b t c h w

        sum_squared_error = (((ndvi_targ - ndvi_pred) * s2_mask) ** 2).sum(1)  # b c h w
        mse = sum_squared_error / (s2_mask.sum(1) + 1e-8)  # b c h w

        if self.scale_by_std:
            mean_ndvi_targ = (ndvi_targ * s2_mask).sum(1).unsqueeze(1) / (
                s2_mask.sum(1).unsqueeze(1) + 1e-8
            )  # b t c h w
            sum_squared_deviation = (((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2).sum(
                1
            )  # b c h w
            mse = sum_squared_error / sum_squared_deviation.clip(
                min=0.01
            )  # mse b c h w
        elif self.weight_by_std:
            mean_ndvi_targ = (ndvi_targ * s2_mask).sum(1).unsqueeze(1) / (
                s2_mask.sum(1).unsqueeze(1) + 1e-8
            )  # b t c h w
            sum_squared_deviation = (((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2).sum(
                1
            )  # b c h w
            mse = sum_squared_error * (
                ((sum_squared_deviation / (s2_mask.sum(1) + 1e-8)) ** 0.5) / 0.1
            ).clip(
                min=0.01, max=100.0
            )  # b c h w

        if self.pred_mask_value:  # what is that?
            pred_mask = (
                (ndvi_pred != self.pred_mask_value).bool().type_as(preds).max(1)[0]
            )
            mse_lc = (mse * lc_mask * pred_mask).sum() / (
                (lc_mask * pred_mask).sum() + 1e-8
            )
        elif self.use_lc:
            mse_lc = (mse * lc_mask).sum() / (lc_mask.sum() + 1e-8)
        else:
            mse_lc = mse.mean()

        logs = {"loss": mse_lc}

        if self.extra_aux_loss_term:
            extra_loss = aux[self.extra_aux_loss_term]
            logs["mse_lc"] = mse_lc
            logs[self.extra_aux_loss_term] = extra_loss
            mse_lc += self.extra_aux_loss_weight * extra_loss
            logs["loss"] = mse_lc

        return mse_lc, logs


def setup_loss(args):
    if args["name"] == "MaskedL2NDVILoss":
        return MaskedL2NDVILoss(**args)
    elif args["name"] == "MaskedPixelwiseLoss":
        return MaskedPixelwiseLoss(**args)
    elif args["name"] == "PixelwiseLoss":
        return PixelwiseLoss(args)
