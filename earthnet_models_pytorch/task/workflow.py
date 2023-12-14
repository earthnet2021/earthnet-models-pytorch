from typing import Optional, Union
from pathlib import Path

import argparse
import ast
import copy
import json
import torch
from torch import nn
import numpy as np
import xarray as xr
import pytorch_lightning as pl


from earthnet_models_pytorch.utils import str2bool, log_viz
from earthnet_models_pytorch.task import setup_loss, SHEDULERS
from earthnet_models_pytorch.metric import METRICS


class SpatioTemporalTask(pl.LightningModule):
    def __init__(self, model: nn.Module, hparams: argparse.Namespace):
        super().__init__()

        if hasattr(self, "save_hyperparameters"):
            # SpatioTemporalTask herite from the LightningModule, save the parameters in a file
            self.save_hyperparameters(copy.deepcopy(hparams))
        else:
            self.hparams = copy.deepcopy(hparams)

        if hparams.pred_dir is None:
            self.pred_dir = (
                Path(self.logger.log_dir) / "predictions"
                if self.logger is not None
                else Path.cwd() / "experiments" / "predictions"
            )  # logger: hyperparameter of LightningModule for the Trainer
        else:
            self.pred_dir = Path(self.hparams.pred_dir)

        self.model = model
        self.loss = setup_loss(hparams.loss)

        self.context_length = hparams.context_length
        self.target_length = hparams.target_length

        #self.lc_min = hparams.lc_min
        #self.lc_max = hparams.lc_max

        self.n_stochastic_preds = hparams.n_stochastic_preds


        self.shedulers = []
        for shedule in self.hparams.shedulers:
            self.shedulers.append(
                (shedule["call_name"], SHEDULERS[shedule["name"]](**shedule["args"]))
            )


        self.metric = METRICS[self.hparams.metric](**self.hparams.metric_kwargs)
        

    @staticmethod
    def add_task_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):  # Optional[X] is equivalent to Union[X, None].
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(
            parents=parent_parser, add_help=False
        )  # parents - A list of ArgumentParser objects whose arguments should also be included

        # Path of the directory to save the prediction
        parser.add_argument("--pred_dir", type=str, default=None)

        # Name of the dataset, involves major differences in the variables available and the tasks to be performed.
        parser.add_argument("--setting", type=str, default="en21-std")

        # Dictionnary of the loss name and the distance norm used.
        parser.add_argument(
            "--loss",
            type=ast.literal_eval,
            default='{"name": "masked", "args": {"distance_type": "L1"}}',
        )
        # Metric used for the test set and the validation set.
        parser.add_argument("--metric", type=str, default="RMSE")
        parser.add_argument('--metric_kwargs', type = ast.literal_eval, default = '{}')

        # Context and target length for temporal model. A temporal model use a context period to learn the temporal dependencies and predict the target period.
        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)

        # Landcover bounds. Used as mask on the non-vegetation pixel.
        #parser.add_argument("--lc_min", type=int, default=10)
        #parser.add_argument("--lc_max", type=int, default=20)

        # Number of stochastic prediction for statistical models.
        parser.add_argument("--n_stochastic_preds", type=int, default=1)

        # Number of batches to be displayed in the logger
        parser.add_argument("--n_log_batches", type=int, default=2)

        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)

        # optimizer: Function that adjusts the attributes of the neural network, such as weights and learning rates.
        parser.add_argument(
            "--optimization",
            type=ast.literal_eval,
            default='{"optimizer": [{"name": "Adam", "args:" {"lr": 0.0001, "betas": (0.9, 0.999)} }], "lr_shedule": [{"name": "multistep", "args": {"milestones": [25, 40], "gamma": 0.1} }]}',
        )
        # Sheduler: methods to adjust the learning rate based on the number of epochs
        parser.add_argument("--shedulers", type=ast.literal_eval, default="[]")

        parser.add_argument("--compute_metric_on_test", type=str2bool, default=False)
        return parser

    def forward(
        self, data, pred_start: int = 0, preds_length: Optional[int] = None, kwargs={}
    ):
        """
        data is a dict with tensors
        pred_start is the first index that shall be predicted, defaults to zero.
        preds_length is the length of the prediction, could also be None.
        kwargs are optional keyword arguments parsed to the model, right now these are model shedulers.
        """
        data["global_step"] = self.global_step
        return self.model(
            data,
            pred_start=pred_start,
            preds_length=preds_length,
            **kwargs,
        )

    def configure_optimizers(self):
        "define and load optimizers and shedulers"
        optimizers = [
            getattr(torch.optim, o["name"])(self.parameters(), **o["args"])
            for o in self.hparams.optimization["optimizer"]
        ]

        # torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs.
        shedulers = [
            getattr(torch.optim.lr_scheduler, s["name"])(optimizers[i], **s["args"])
            for i, s in enumerate(self.hparams.optimization["lr_shedule"])
        ]
        return optimizers, shedulers

    def training_step(self, batch, batch_idx):
        """compute and return the training loss and some additional metrics for e.g. the progress bar or logger"""

        # Learning rate scheduling should be applied after optimizer’s update
        kwargs = {}
        for shedule_name, shedule in self.shedulers:
            kwargs[shedule_name] = shedule(self.global_step)


        # Predictions generation
        preds, aux = self(batch, kwargs=kwargs)
        loss, logs = self.loss(preds, batch, aux, current_step=self.global_step)

        # Logs
        for shedule_name in kwargs:
            if len(kwargs[shedule_name]) > 1:
                for i, shed_val in enumerate(kwargs[shedule_name]):
                    logs[f"{shedule_name}_i"] = shed_val
            else:
                logs[shedule_name] = kwargs[shedule_name]
        logs["batch_size"] = torch.tensor(
            self.hparams.train_batch_size, dtype=torch.float32
        )
        # Metric logging method
        self.log_dict(logs)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform one evaluation epoch over the validation set. Operates on a single batch of data from the validation set. In this step you d might generate examples or calculate anything of interest like accuracy.
        Return  a List of dictionaries with metrics logged during the validation phase
        """

        data = copy.deepcopy(batch)

        batch_size = torch.tensor(self.hparams.val_batch_size, dtype=torch.int64)

        # Select only the context data for the model
        data["dynamic"][0] = data["dynamic"][0][:, :self.context_length, ...]

        loss_logs = []  # list of loss values
        viz_logs = []  # list of (preds, scores)

        # nb of predictions for statistical models
        for i in range(self.n_stochastic_preds):
            # Predictions of the model
            preds, aux = self(
                data, pred_start=self.context_length, preds_length=self.target_length
            )

            # Loss computation
            mse_lc, logs = self.loss(preds, batch, aux)

            if np.isfinite(mse_lc.cpu().detach().numpy()):
                loss_logs.append(logs)

            # Update of the metric
            self.metric.update(preds, batch)

            # compute the scores for the n_batches that can be visualized in the logger
            if batch_idx < self.hparams.n_log_batches:
                scores = self.metric.compute_batch(batch)
                viz_logs.append((preds, scores))

        mean_logs = {
            log_name: torch.tensor(
                [log[log_name] for log in loss_logs],
                dtype=torch.float32,
                device=self.device,
            ).mean()
            for log_name in loss_logs[0]
        }

        # loss_val
        self.log_dict(
            {log_name + "_val": mean_logs[log_name] for log_name in mean_logs},
            sync_dist=True,
            batch_size=batch_size,
        )

        # Visualisation of the prediction for the n first batches
        if batch_idx < self.hparams.n_log_batches and len(preds.shape) == 5:
            if self.logger is not None and preds.shape[2] == 1:
                log_viz(
                    self.logger.experiment,
                    viz_logs,
                    batch,
                    batch_idx,
                    self.current_epoch,
                    setting=self.hparams.setting,
                )

    def validation_epoch_end(self, validation_step_outputs):
        current_scores = self.metric.compute()
        self.log_dict(current_scores, sync_dist=True)
        self.metric.reset()  # lagacy? To remove, shoudl me managed by the logger?
        if (
            self.logger is not None
            and type(self.logger.experiment).__name__ != "DummyExperiment"
            and self.trainer.is_global_zero
        ):
            current_scores["epoch"] = self.current_epoch
            current_scores = {
                k: str(v.detach().cpu().item())
                if isinstance(v, torch.Tensor)
                else str(v)
                for k, v in current_scores.items()
            }
            outpath = Path(self.logger.log_dir) / "validation_scores.json"
            if outpath.is_file():
                with open(outpath, "r") as fp:
                    past_scores = json.load(fp)
                scores = past_scores + [current_scores]
            else:
                scores = [current_scores]

            with open(outpath, "w") as fp:
                json.dump(scores, fp)

    def test_step(self, batch, batch_idx):
        """Operates on a single batch of data from the test set. In this step you generate examples or calculate anything of interest such as accuracy."""
        scores = []

        data = copy.deepcopy(batch)

        #data["dynamic"][0] =  data["dynamic"][0][:,:self.context_length,...]  # selection only the context data
        #if len(data["dynamic_mask"]) > 0:
        #    data["dynamic_mask"][0] = data["dynamic_mask"][0][:,:self.context_length,...]

        for i in range(self.n_stochastic_preds):
            preds, aux = self(
                batch, pred_start=self.context_length, preds_length=self.target_length
            )

            lc = batch["landcover"]

            static = batch["static"][0]

            for j in range(preds.shape[0]):
                if self.hparams.setting in ["en21x", "en23"]:
                    # Targets
                    targ_path = Path(batch["filepath"][j])
                    targ_cube = xr.open_dataset(targ_path)

                    lat = targ_cube.lat
                    lon = targ_cube.lon

                    ndvi_preds = preds[j, :, 0, ...].detach().cpu().numpy()
                    pred_cube = xr.Dataset(
                        {
                            "ndvi_pred": xr.DataArray(
                                data=ndvi_preds,
                                coords={
                                    "time": targ_cube.time.isel(
                                        time=slice(4, None, 5)
                                    ).isel(
                                        time=slice(
                                            self.context_length,
                                            self.context_length + self.target_length,
                                        )
                                    ),
                                    "lat": lat,
                                    "lon": lon,
                                },
                                dims=["time", "lat", "lon"],
                            )
                        }
                    )

                    pred_dir = self.pred_dir
                    pred_path = pred_dir / targ_path.parent.stem / targ_path.name
                    pred_path.parent.mkdir(parents=True, exist_ok=True)

                    if pred_path.is_file():
                        pred_path.unlink()

                    if not pred_path.is_file():
                        pred_cube.to_netcdf(
                            pred_path, encoding={"ndvi_pred": {"dtype": "float32"}}
                        )

                elif self.hparams.setting in ["en21xold", "en22"]:
                    # Targets
                    targ_path = Path(batch["filepath"][j])
                    targ_cube = xr.open_dataset(targ_path)
                    tile = targ_path.name[:5]

                    hrd = (
                        preds[j, ...]
                        .permute(2, 3, 1, 0)
                        .squeeze(2)
                        .detach()
                        .cpu()
                        .numpy()
                        if "full" not in aux
                        else aux["full"][j, ...]
                        .permute(2, 3, 1, 0)
                        .detach()
                        .cpu()
                        .numpy()
                    )  # h, w, c, t

                    # Masks
                    masks = (
                        (lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()
                    ).type_as(
                        preds
                    )  # mask for outlayers using lc threshold   # mask for outlayers using lc threshold
                    masks = masks[j, ...].permute(1, 2, 0).detach().cpu().numpy()

                    static = static[j, ...].permute(1, 2, 0).detach().cpu().numpy()

                    # Paths
                    if self.n_stochastic_preds == 1:
                        if targ_path.parents[1].name == "sim_extremes":
                            pred_dir = self.pred_dir / targ_path.parents[0].name
                        else:
                            pred_dir = self.pred_dir
                        pred_path = pred_dir / targ_path.name
                    else:
                        if targ_path.parents[1].name == "sim_extremes":
                            pred_dir = self.pred_dir / tile / targ_path.parents[0].name
                        else:
                            pred_dir = self.pred_dir / tile
                        pred_path = pred_dir / f"pred_{i+1}_{targ_path.name}"

                    # TODO save all preds for one cube in same file....
                    pred_dir.mkdir(parents=True, exist_ok=True)

                    # Axis
                    y = targ_cube["y"].values
                    x = targ_cube["x"].values

                    pred_cube = xr.Dataset(
                        {
                            "ndvi_pred": xr.DataArray(
                                data=hrd,
                                coords={
                                    "time": targ_cube.time.isel(
                                        time=slice(
                                            self.context_length,
                                            self.context_length + self.target_length,
                                        )
                                    ),
                                    "latitude": y,
                                    "longitude": x,
                                },
                                dims=["latitude", "longitude", "time"],
                            )
                        }
                    )

                    if not pred_path.is_file():
                        pred_cube.to_netcdf(
                            pred_path,
                            encoding={
                                "ndvi_pred": {"dtype": "float32"},

                            },
                        )

                else:
                    cubename = batch["cubename"][j]
                    cube_dir = self.pred_dir / cubename[:5]
                    cube_dir.mkdir(parents=True, exist_ok=True)
                    cube_path = cube_dir / f"pred{i+1}_{cubename}"

                    targ_path = Path(batch["filepath"][j])
                    targ_cube = xr.open_dataset(targ_path)
                    lon = targ_cube["lon"].values
                    lat = targ_cube["lat"].values
                    hrd = (
                        preds[j, :, 0, ...].permute(1, 2, 0).detach().cpu().numpy()
                        if "full" not in aux
                        else aux["full"][j, ...]
                        .permute(2, 3, 1, 0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    pred_cube = xr.Dataset(
                        {
                            "ndvi_pred": xr.DataArray(
                                data=hrd,
                                coords={
                                    "time": targ_cube.time.isel(
                                        time=slice(
                                            4 + 5 * self.context_length,
                                            4 + 5 * self.target_length,
                                            5,
                                        )
                                    ),
                                    "latitude": lat,
                                    "longitude": lon,
                                },
                                dims=["latitude", "longitude", "time"],
                            )
                        }
                    )
                    if not cube_path.is_file():
                        pred_cube.to_netcdf(
                            cube_path,
                            encoding={"ndvi_pred": {"dtype": "float32"}},
                        )

            if self.hparams.compute_metric_on_test:
                self.metric.update(preds, batch)
                scores.append(self.metric.compute_batch(batch))
        return scores

    def test_epoch_end(self, test_step_outputs):
        """Called at the end of a test epoch with the output of all test steps."""
        if self.hparams.compute_metric_on_test:
            self.pred_dir.mkdir(parents=True, exist_ok=True)
            print(self.pred_dir)
            with open(
                self.pred_dir / f"individual_scores_{self.global_rank}.json", "w"
            ) as fp:
                json.dump(
                    [
                        {
                            k: v if isinstance(v, str) else v.item()
                            for k, v in test_step_outputs[i][j][l].items()
                        }
                        for i in range(len(test_step_outputs))
                        for j in range(len(test_step_outputs[i]))
                        for l in range(len(test_step_outputs[i][j]))
                    ],
                    fp,
                )

            scores = self.metric.compute()
            if self.trainer.is_global_zero:
                with open(self.pred_dir / "total_score.json", "w") as fp:
                    json.dump(
                        {
                            k: v if isinstance(v, str) else v.item()
                            for k, v in scores.items()
                        },
                        fp,
                    )

    def teardown(self, stage):
        if stage == "test" and self.hparams.compute_metric_on_test:
            if self.global_rank == 0:
                data = []
                for path in self.pred_dir.glob("individual_scores_*.json"):
                    with open(path, "r") as fp:
                        data += json.load(fp)
                    path.unlink(missing_ok=True)
                if len(data) == 0:
                    return
                out_names = []
                out = []
                for d in data:
                    if d["name"] not in out_names:
                        out.append(d)
                        out_names.append(d["name"])
                with open(self.pred_dir / f"individual_scores.json", "w") as fp:
                    json.dump(out, fp)
        return
