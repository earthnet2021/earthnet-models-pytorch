
from typing import Optional, Union

import argparse
import ast
import copy
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as clr
import xarray as xr

import torch
import torchvision

import pytorch_lightning as pl 

from torch import nn

from pathlib import Path
import shutil

from earthnet_models_pytorch.utils import str2bool, log_viz
from earthnet_models_pytorch.task import setup_loss, SHEDULERS
from earthnet_models_pytorch.setting import METRICS

class SpatioTemporalTask(pl.LightningModule):

    def __init__(self, model: nn.Module, hparams: argparse.Namespace):
        super().__init__()

        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(copy.deepcopy(hparams))
        else:
            self.hparams = copy.deepcopy(hparams)
        self.model = model

        if hparams.pred_dir is None:
            self.pred_dir = Path(self.logger.log_dir)/"predictions" if self.logger is not None else Path.cwd()/"experiments"/"predictions"
        else:
            self.pred_dir = Path(self.hparams.pred_dir)

        self.loss = setup_loss(hparams.loss)

        self.context_length = hparams.context_length
        self.target_length = hparams.target_length

        self.n_stochastic_preds = hparams.n_stochastic_preds

        self.current_filepaths = []

        self.metric = METRICS[self.hparams.setting]()
        self.ndvi_pred = (self.hparams.setting == "en21-veg") #TODO: Legacy, remove this...
        self.pred_mode = {"en21-veg": "ndvi", "en21-std": "rgb", "en21x": "kndvi", "en21x-px": "kndvi", "en22": "kndvi"}[self.hparams.setting]

        self.model_shedules = []
        for shedule in self.hparams.model_shedules:
            self.model_shedules.append((shedule["call_name"], SHEDULERS[shedule["name"]](**shedule["args"])))


    @staticmethod
    def add_task_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument('--pred_dir', type = str, default = None)

        parser.add_argument('--loss', type = ast.literal_eval, default = '{"name": "masked", "args": {"distance_type": "L1"}}')

        parser.add_argument('--context_length', type = int, default = 10)
        parser.add_argument('--target_length', type = int, default = 20)
        parser.add_argument('--n_stochastic_preds', type = int, default = 10)

        parser.add_argument('--n_log_batches', type = int, default = 2)

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
        shedulers = [getattr(torch.optim.lr_scheduler,s["name"])(optimizers[i], **s["args"]) for i, s in enumerate(self.hparams.optimization["lr_shedule"])] # This gets any(!) torch.optim.lr_scheduler - but only those with standard callback will work (i.e. not the Plateau one)
        return optimizers, shedulers

    
#     def tbptt_split_batch(self, batch, split_size):
#         splits = []

#         split_times = [0,15,21,27,33,39,45]

#         for i in range(6):
#             splits.append({k: batch[k] if "dynamic" not in k else [v[:, split_times[i]:split_times[i+1],...] for v in batch[k]] for k in batch})
# ˇ        return splits

    
    def training_step(self, batch, batch_idx):
        
        kwargs = {}
        for (shedule_name, shedule) in self.model_shedules:
            kwargs[shedule_name] = shedule(self.global_step)

        preds, aux = self(batch, n_preds = self.context_length+self.target_length, kwargs = kwargs)

        loss, logs = self.loss(preds, batch, aux, current_step = self.global_step)

        for arg in kwargs:
            logs[arg] = kwargs[arg]

        self.log_dict(logs)

        return loss

    def validation_step(self, batch, batch_idx):

        data = copy.deepcopy(batch)

        data["dynamic"][0] =  data["dynamic"][0][:,:self.context_length,...]
        if len(data["dynamic_mask"]) > 0:
            data["dynamic_mask"][0] = data["dynamic_mask"][0][:,:self.context_length,...]

        all_logs = []
        all_viz = []
        for i in range(self.n_stochastic_preds):
            preds, aux = self(data, pred_start = self.context_length, n_preds = self.target_length)
            all_logs.append(self.loss(preds, batch, aux)[1])
            if self.loss.distance.rescale:
                preds = ((preds - 0.2)/0.6)
            if batch_idx < self.hparams.n_log_batches:
                self.metric.compute_on_step = True
                scores = self.metric(preds, batch)
                self.metric.compute_on_step = False
                all_viz.append((preds, scores))
            else:
                self.metric(preds, batch)
        
        mean_logs = {l: torch.tensor([log[l] for log in all_logs], dtype=torch.float32, device = self.device).mean() for l in all_logs[0]}
        self.log_dict({l+"_val": mean_logs[l] for l in mean_logs}, sync_dist=True)

        if batch_idx < self.hparams.n_log_batches and len(preds.shape) == 5:
            if self.logger is not None:
                log_viz(self.logger.experiment, all_viz, batch, batch_idx, self.current_epoch, mode = self.pred_mode, lc_min = 82 if not self.hparams.setting == "en22" else 2, lc_max = 104 if not self.hparams.setting == "en22" else 6)

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
        scores = []
        for i in range(self.n_stochastic_preds):
            preds, aux = self(batch, pred_start = self.context_length, n_preds = self.target_length)
            if self.loss.distance.rescale:
                preds = ((preds - 0.2)/0.6)
            for j in range(preds.shape[0]):
                if self.hparams.setting in ["en21x", "en22"]:
                    targ_path = Path(batch["filepath"][j])
                    targ_cube = xr.open_dataset(targ_path)
                    tile = targ_path.name[:5]
                    hrd = preds[j,...].permute(2,3,1,0).detach().cpu().numpy() if "full" not in aux else aux["full"][j,...].permute(2,3,1,0).detach().cpu().numpy()
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
                    y = targ_cube["y"].values
                    y[0] = y[1] + 20
                    y[-1] = y[-2] - 20
                    x = targ_cube["x"].values
                    x[0] = x[1] - 20
                    x[-1] = x[-2] + 20
                    pred_cube = xr.Dataset({"kndvi_pred": xr.DataArray(data = (np.tanh(1) * hrd[:,:,0,:]).clip(0, np.tanh(1)), coords = {"time": targ_cube.time.isel(time = slice(9,45)), "y": y, "x": x}, dims = ["y", "x", "time"])})
                    pred_cube["kndvi"] = xr.DataArray(data = targ_cube["kndvi"].isel(time = slice(9,45)).values, coords = {"time": targ_cube.time.isel(time = slice(9,45)), "y": y, "x": x}, dims = ["y", "x", "time"])
                    if not pred_path.is_file():
                        pred_cube.to_netcdf(pred_path)
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

    # def log_viz(self, viz_data, batch, batch_idx): #TODO factor functionality out, remove duplicity
    #     tensorboard_logger = self.logger.experiment
    #     targs = batch["dynamic"][0]
    #     if "landcover" in batch:
    #         lc = batch["landcover"]
    #         lc = 1 - (lc > 63).byte() & (lc < 105).byte()
    #     masks = batch["dynamic_mask"][0].byte()
    #     for i, (preds, scores) in enumerate(viz_data):
    #         for j in range(preds.shape[0]):
    #             # Predictions RGB
                
    #             if not self.ndvi_pred:
    #                 rgb = torch.cat([preds[j,:,2,...].unsqueeze(1)*10000,preds[j,:,1,...].unsqueeze(1)*10000,preds[j,:,0,...].unsqueeze(1)*10000],dim = 1)
    #                 grid = torchvision.utils.make_grid(rgb, nrow = 10, normalize = True, range = (0,5000))
    #                 text = f"Cube: {scores[j]['name']} ENS: {scores[j]['ENS']:.4f} MAD: {scores[j]['MAD']:.4f} OLS: {scores[j]['OLS']:.4f} EMD: {scores[j]['EMD']:.4f} SSIM: {scores[j]['SSIM']:.4f}"
    #                 text = torch.tensor(self.text_phantom(text, width = grid.shape[-1]), dtype=torch.float32, device = self.device).type_as(grid).permute(2,0,1)
    #                 grid = torch.cat([grid, text], dim = -2)
    #                 tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} RGB Preds, Sample: {i}", grid, self.current_epoch)
    #                 ndvi = self.ndvi_colorize((preds[j,:,3,...] - preds[j,:,2,...])/(preds[j,:,3,...] + preds[j,:,2,...]+1e-6), mask = None if "landcover" not in batch else lc[j,...].repeat(preds.shape[1],1,1))
    #                 grid = torchvision.utils.make_grid(ndvi, nrow = 10)
    #             else:
    #                 ndvi = self.ndvi_colorize(preds[j,...].squeeze(), mask = None if "landcover" not in batch else lc[j,...].repeat(preds.shape[1],1,1))
    #                 text = f"Cube: {scores[j]['name']} RMSE: {scores[j]['rmse']:.4f}"
    #                 grid = torchvision.utils.make_grid(ndvi, nrow = 10)
    #                 text = torch.tensor(self.text_phantom(text, width = grid.shape[-1]), dtype=torch.float32, device = self.device).type_as(grid).permute(2,0,1)
    #             grid = torch.cat([grid, text], dim = -2)
    #             tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} NDVI Preds, Sample: {i}", grid, self.current_epoch)
    #             ndvi = (preds[j,:,3,...] - preds[j,:,2,...])/(preds[j,:,3,...] + preds[j,:,2,...]+1e-6) if not self.ndvi_pred else preds[j,...].squeeze()
    #             ndvi_chg = (ndvi[1:,...]-ndvi[:-1,...]+1)/2
    #             grid = torchvision.utils.make_grid(ndvi_chg.unsqueeze(1), nrow = 10)
    #             grid = torch.cat([grid, text], dim = -2)
    #             tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} NDVI Change, Sample: {i}", grid, self.current_epoch)
    #             # Images
    #             rgb = torch.cat([targs[j,:,2,...].unsqueeze(1)*10000,targs[j,:,1,...].unsqueeze(1)*10000,targs[j,:,0,...].unsqueeze(1)*10000],dim = 1)
    #             if i == 0:
    #                 grid = torchvision.utils.make_grid(rgb, nrow = 10, normalize = True, range = (0,5000))
    #                 self.logger.experiment.add_image(f"Cube: {batch_idx*preds.shape[0] + j} RGB Targets", grid, self.current_epoch)
    #                 ndvi = self.ndvi_colorize((targs[j,:,3,...] - targs[j,:,2,...])/(targs[j,:,3,...] + targs[j,:,2,...]+1e-6), mask = None if "landcover" not in batch else lc[j,...].repeat(targs.shape[1],1,1), clouds = masks[j,:,0,...])
    #                 grid = torchvision.utils.make_grid(ndvi, nrow = 10)
    #                 self.logger.experiment.add_image(f"Cube: {batch_idx*preds.shape[0] + j} NDVI Targets", grid, self.current_epoch)

    # def text_phantom(self, text, width): #TODO move to generic function
    #     # Create font
    #     pil_font = ImageFont.load_default()#
    #     text_width, text_height = pil_font.getsize(text)

    #     # create a blank canvas with extra space between lines
    #     canvas = Image.new('RGB', [width, text_height], (255, 255, 255))

    #     # draw the text onto the canvas
    #     draw = ImageDraw.Draw(canvas)
    #     offset = ((width - text_width) // 2 , 0)
    #     white = "#000000"
    #     draw.text(offset, text, font=pil_font, fill=white)

    #     # Convert the canvas into an array with values in [0, 1]
    #     return (255 - np.asarray(canvas)) / 255.0

    # def ndvi_colorize(self, data, mask = None, clouds = None): #TODO move to generic function or take from earthnet.tk
    #     # mask has 1 channel
    #     # clouds has 0 channel
    #     t,h,w = data.shape
    #     in_data = copy.deepcopy(data.reshape(-1)).detach().cpu().numpy()
    #     if mask is not None:
    #         in_data = np.ma.array(in_data, mask = copy.deepcopy(mask.reshape(-1)).detach().cpu().numpy())            
    #     cmap = clr.LinearSegmentedColormap.from_list('custom blue', ["#cbbe9a","#fffde4","#bccea5","#66985b","#2e6a32","#123f1e","#0e371a","#01140f","#000d0a"], N=256)
    #     cmap.set_bad(color='red')
    #     if clouds is None:
    #         return torch.as_tensor(cmap(in_data)[:,:3], dtype = data.dtype, device = data.device).reshape(t,h,w,3).permute(0,3,1,2)
    #     else:
    #         out = torch.as_tensor(cmap(in_data)[:,:3], dtype = data.dtype, device = data.device).reshape(t,h,w,3).permute(0,3,1,2)
    #         return torch.stack([torch.where(clouds, out[:,0,...],torch.zeros_like(out[:,0,...], dtype = data.dtype, device = data.device)), torch.where(clouds, out[:,1,...],torch.zeros_like(out[:,1,...], dtype = data.dtype, device = data.device)), torch.where(clouds, out[:,2,...],0.1*torch.ones_like(out[:,2,...], dtype = data.dtype, device = data.device))], dim = 1)