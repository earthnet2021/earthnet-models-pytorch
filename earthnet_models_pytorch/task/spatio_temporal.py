
from typing import Optional, Union

import argparse
import ast
import copy
import json

import numpy as np
import xarray as xr

import torch

import pytorch_lightning as pl 

from torch import nn

from pathlib import Path

from earthnet_models_pytorch.utils import str2bool, log_viz
from earthnet_models_pytorch.task import setup_loss, SHEDULERS
from earthnet_models_pytorch.metric import METRICS

class SpatioTemporalTask(pl.LightningModule):

    def __init__(self, model: nn.Module, hparams: argparse.Namespace):
        super().__init__()
        
        if hasattr(self, "save_hyperparameters"):  # SpatioTemporalTask herite from the LightningModule
            self.save_hyperparameters(copy.deepcopy(hparams))
        else:
            self.hparams = copy.deepcopy(hparams)
        self.model = model

        if hparams.pred_dir is None:
            self.pred_dir = Path(self.logger.log_dir)/"predictions" if self.logger is not None else Path.cwd()/"experiments"/"predictions"  # logger: hyperparameter of LightningModule for the Trainer
        else:
            self.pred_dir = Path(self.hparams.pred_dir)

        self.loss = setup_loss(hparams.loss)

        self.context_length = hparams.context_length
        self.target_length = hparams.target_length

        self.min_lc = hparams.min_lc
        self.max_lc = hparams.max_lc

        self.n_stochastic_preds = hparams.n_stochastic_preds 

        self.current_filepaths = []

        self.metric = METRICS[self.hparams.setting]()
        self.ndvi_pred = (self.hparams.setting == "en21-veg") #TODO: Legacy, remove this...
        
        self.pred_mode = {"en21-veg": "ndvi", "en21-std": "rgb", "en21x": "kndvi", "en21x-px": "kndvi", "en22": "kndvi"}[self.hparams.setting]

        self.model_shedules = []
        for shedule in self.hparams.model_shedules:
            self.model_shedules.append((shedule["call_name"], SHEDULERS[shedule["name"]](**shedule["args"])))


    @staticmethod
    def add_task_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):  # Optional[X] is equivalent to Union[X, None].
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)  # parents - A list of ArgumentParser objects whose arguments should also be included

        parser.add_argument('--pred_dir', type = str, default = None)

        parser.add_argument('--loss', type = ast.literal_eval, default = '{"name": "masked", "args": {"distance_type": "L1"}}')

        parser.add_argument('--context_length', type = int, default = 10)
        parser.add_argument('--target_length', type = int, default = 20)

        parser.add_argument('--min_lc', type = int, default = 10)
        parser.add_argument('--max_lc', type = int, default = 20)

        parser.add_argument('--n_stochastic_preds', type = int, default = 10)

        parser.add_argument('--n_log_batches', type = int, default = 2)

        parser.add_argument('--train_batch_size', type = int, default = 1)
        parser.add_argument('--val_batch_size', type = int, default = 1)
        parser.add_argument('--test_batch_size', type = int, default = 1)

        parser.add_argument('--optimization', type = ast.literal_eval, default = '{"optimizer": [{"name": "Adam", "args:" {"lr": 0.0001, "betas": (0.9, 0.999)} }], "lr_shedule": [{"name": "multistep", "args": {"milestones": [25, 40], "gamma": 0.1} }]}')

        parser.add_argument('--model_shedules', type = ast.literal_eval, default = '[]')

        parser.add_argument('--setting', type = str, default = "en21-std")

        parser.add_argument('--compute_metric_on_test', type = str2bool, default = False)
        return parser

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None, kwargs = {}):
        """
        data is a dict with tensors
        pred_start is the first index that shall be predicted, defaults to zero.
        n_preds is the length of the prediction, could also be None.
        kwargs are optional keyword arguments parsed to the model, right now these are model shedulers.
        """        
        return self.model(data, pred_start = pred_start, n_preds = n_preds, **kwargs)

    def configure_optimizers(self):
        optimizers = [getattr(torch.optim,o["name"])(self.parameters(), **o["args"]) for o in self.hparams.optimization["optimizer"]] # This gets any (!) torch.optim optimizer
        # torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs. 
        shedulers = [getattr(torch.optim.lr_scheduler,s["name"])(optimizers[i], **s["args"]) for i, s in enumerate(self.hparams.optimization["lr_shedule"])] # This gets any(!) torch.optim.lr_scheduler - but only those with standard callback will work (i.e. not the Plateau one)
        return optimizers, shedulers

    
    def training_step(self, batch, batch_idx):
        '''compute and return the training loss and some additional metrics for e.g. the progress bar or logger'''
        kwargs = {}
        for (shedule_name, shedule) in self.model_shedules:
            kwargs[shedule_name] = shedule(self.global_step)  # adjust the learning rate
        
        # Predictions generation
        preds, aux = self(batch, n_preds = self.context_length+self.target_length, kwargs = kwargs) 
        loss, logs = self.loss(preds, batch, aux, current_step = self.global_step)

        # Logs
        for shedule_name in kwargs:
            logs[shedule_name] = kwargs[shedule_name]
        logs['batch_size'] = torch.tensor(self.hparams.train_batch_size, dtype=torch.float32)
        self.log_dict(logs)  
        
        return loss

    def validation_step(self, batch, batch_idx):
        '''Operates on a single batch of data from the validation set. In this step you d might generate examples or calculate anything of interest like accuracy.'''
        data = copy.deepcopy(batch)

        data["dynamic"][0] =  data["dynamic"][0][:,:self.context_length,...]  # selection only the context data
        if len(data["dynamic_mask"]) > 0:
            data["dynamic_mask"][0] = data["dynamic_mask"][0][:,:self.context_length,...]

        all_logs = []
        all_viz = []
        
        for i in range(self.n_stochastic_preds):  # several predictions 
            preds, aux = self(data, pred_start = self.context_length, n_preds = self.target_length)  # output model
            all_logs.append(self.loss(preds, batch, aux)[1])

            # if self.loss.distance.rescale:
            #    preds = ((preds - 0.2)/0.6)  

            if batch_idx < self.hparams.n_log_batches:
                self.metric.compute_on_step = True
                scores = self.metric(preds, batch)  # Returns the metric value over inputs if ``compute_on_step`` is True
                self.metric.compute_on_step = False
                all_viz.append((preds, scores))
            else:
                self.metric(preds, batch)

        mean_logs = {l: torch.tensor([log[l].mean() for log in all_logs], dtype=torch.float32, device = self.device).mean() for l in all_logs[0]} 

        batch_size = torch.tensor(self.hparams.val_batch_size, dtype=torch.float32)
        
        # loss_val
        self.log_dict({l+"_val": mean_logs[l] for l in mean_logs}, sync_dist=True, batch_size=batch_size)  

        if batch_idx < self.hparams.n_log_batches and len(preds.shape) == 5:
            if self.logger is not None and preds.shape[2] == 1:
                log_viz(self.logger.experiment, all_viz, batch, batch_idx, self.current_epoch, mode = self.pred_mode) #, lc_min = 82 if not self.hparams.setting == "en22" else 2, lc_max = 104 if not self.hparams.setting == "en22" else 6)

    def validation_epoch_end(self, validation_step_outputs):
        current_scores = self.metric.compute()
        self.log_dict(current_scores, sync_dist=True)
        self.metric.reset()
        if self.logger is not None and type(self.logger.experiment).__name__ != "DummyExperiment" and self.trainer.is_global_zero:
            current_scores["epoch"] = self.current_epoch
            current_scores = {k: str(v.detach().cpu().item())  if isinstance(v, torch.Tensor) else str(v) for k,v in current_scores.items()}
            outpath = Path(self.logger.log_dir)/"validation_scores.json"
            if outpath.is_file():
                with open(outpath, "r") as fp:
                    past_scores = json.load(fp)
                scores = past_scores + [current_scores]
            else:
                scores = [current_scores]        

            with open(outpath, "w") as fp:
                json.dump(scores, fp)


    def test_step(self, batch, batch_idx):
        '''Operates on a single batch of data from the test set. In this step you d normally generate examples or calculate anything of interest such as accuracy.'''
        scores = []

        for i in range(self.n_stochastic_preds):
            preds, aux = self(batch, pred_start = self.context_length, n_preds = self.target_length)

            lc = batch["landcover"]
            
            static = batch["static"][0]

            for j in range(preds.shape[0]):
                if self.hparams.setting in ["en21x", "en22"]:
                    # Targets
                    targ_path = Path(batch["filepath"][j])
                    targ_cube = xr.open_dataset(targ_path)
                    tile = targ_path.name[:5]

                    hrd = preds[j,...].permute(2,3,1,0).squeeze(2).detach().cpu().numpy() if "full" not in aux else aux["full"][j,...].permute(2,3,1,0).detach().cpu().numpy()  # h, w, c, t
                    
                    # Masks
                    masks = ((lc >= self.min_lc).bool() & (lc <= self.max_lc).bool()).type_as(preds)  # mask for outlayers using lc threshold   # mask for outlayers using lc threshold           
                    masks = masks[j,...].permute(1,2,0).detach().cpu().numpy() 

                    landcover = lc[j,...].permute(1,2,0).detach().cpu().numpy() 
                    static = static[j,...].permute(1,2,0).detach().cpu().numpy()
                    geom = targ_cube.geom
                 
                    # Paths
                    if self.n_stochastic_preds == 1:
                        if targ_path.parents[1].name == "sim_extremes": 
                            pred_dir = self.pred_dir/targ_path.parents[0].name
                        else:
                            pred_dir = self.pred_dir
                        pred_path = pred_dir/targ_path.name
                    else:
                        if targ_path.parents[1].name == "sim_extremes":
                            pred_dir = self.pred_dir/tile/targ_path.parents[0].name
                        else:
                            pred_dir = self.pred_dir/tile
                        pred_path = pred_dir/f"pred_{i+1}_{targ_path.name}"

                    # TODO save all preds for one cube in same file....
                    pred_dir.mkdir(parents = True, exist_ok = True)

                    # Axis
                    y = targ_cube["y"].values
                    x = targ_cube["x"].values


                    # Saving
                    targ_cube["ndvi_target"] = (targ_cube.nir - targ_cube.red)/(targ_cube.nir+targ_cube.red+1e-6)
                    pred_cube = xr.Dataset({"ndvi_pred": xr.DataArray(data = hrd, coords = {"time": targ_cube.time.isel(time = slice(self.context_length,self.context_length+self.target_length)), "latitude": y, "longitude": x}, dims = ["latitude", "longitude", "time"])})
                    pred_cube["ndvi_target"] = xr.DataArray(data = targ_cube["ndvi_target"].isel(time = slice(self.context_length,self.context_length+self.target_length)).values, coords = {"time": targ_cube.time.isel(time = slice(self.context_length,self.context_length+self.target_length)), "latitude": y, "longitude": x}, dims = ["latitude", "longitude", "time"])
                    pred_cube["mask"] = xr.DataArray(data = masks[:,:,0], coords = {"latitude": y, "longitude": x}, dims = ["latitude", "longitude"])
                    pred_cube["landcover"] = xr.DataArray(data = landcover[:,:,0], coords = {"latitude": y, "longitude": x}, dims = ["latitude", "longitude"])
                    pred_cube["geomorphons"] = xr.DataArray(data = geom, coords = {"latitude": y, "longitude": x}, dims = ["latitude", "longitude"])
                    pred_cube["SRTM"] = xr.DataArray(data = static[:,:,0], coords = {"latitude": y, "longitude": x}, dims = ["latitude", "longitude"])
                    
                    if not pred_path.is_file():
                      pred_cube.to_netcdf(pred_path, encoding={"ndvi_pred":{"dtype": "float32"}, "ndvi_target":{"dtype": "float32"}, "mask":{"dtype": "float32"}, "landcover":{"dtype": "float32"}, "geomorphons":{"dtype": "float32"}, "SRTM":{"dtype": "float32"}})
                    
                    
                    # Vitus code
                    # pred_cube = xr.Dataset({"kndvi_pred": xr.DataArray(data = (np.tanh(1) * hrd[:,:,0,:]).clip(0, np.tanh(1)), coords = {"time": targ_cube.time.isel(time = slice(9,45)), "y": y, "x": x}, dims = ["y", "x", "time"])})
                    # pred_cube["kndvi"] = xr.DataArray(data = targ_cube["kndvi"].isel(time = slice(9,45)).values, coords = {"time": targ_cube.time.isel(time = slice(9,45)), "y": y, "x": x}, dims = ["y", "x", "time"])

                    # if not pred_path.is_file():
                       # pred_cube.to_netcdf(pred_path)
                                       
                else:
                    cubename = batch["cubename"][j]
                    cube_dir = self.pred_dir/cubename[:5]
                    cube_dir.mkdir(parents = True, exist_ok = True)
                    cube_path = cube_dir/f"pred{i+1}_{cubename}"
                    np.savez_compressed(cube_path, highresdynamic = preds[j,...].permute(2,3,1,0).detach().cpu().numpy() if "full" not in aux else aux["full"][j,...].permute(2,3,1,0).detach().cpu().numpy())

            if self.hparams.compute_metric_on_test:
                self.metric.compute_on_step = True
                scores.append(self.metric(preds, batch))
                self.metric.compute_on_step = False
            
        return scores
    
    def test_epoch_end(self, test_step_outputs):
        '''Called at the end of a test epoch with the output of all test steps.'''
        if self.hparams.compute_metric_on_test:
            self.pred_dir.mkdir(parents = True, exist_ok = True)
            with open(self.pred_dir/f"individual_scores_{self.global_rank}.json", "w") as fp:
                json.dump([{k: v if isinstance(v, str) else v.item() for k,v in test_step_outputs[i][j][l].items()} for i in range(len(test_step_outputs)) for j in range(len(test_step_outputs[i])) for l in range(len(test_step_outputs[i][j]))], fp)
            
            scores = self.metric.compute()
            if self.trainer.is_global_zero:
                with open(self.pred_dir/"total_score.json", "w") as fp:
                    json.dump({k: v if isinstance(v, str) else v.item() for k,v in scores.items()}, fp)
    
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
                with open(self.pred_dir/f"individual_scores.json", "w") as fp:
                    json.dump(out, fp)
        return
   