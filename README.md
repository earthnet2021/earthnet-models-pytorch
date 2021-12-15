# earthnet-models-pytorch

## Requirements

```
conda create -n emp python=3.8
conda activate emp
conda install -c conda-forge mamba
mamba install pytorch torchvision tensorboard
mamba install numpy matplotlib pillow xarray zarr netcdf4
pip install pytorch-lightning earthnet segmentation-models-pytorch
```

## Installation

```
pip install git+https://github.com/vitusbenson/earthnet-models-pytorch.git
```
or:
```
git clone https://github.com/vitusbenson/earthnet-models-pytorch.git
cd earthnet-models-pytorch
pip install -e .
```

## Debug

## Train

## Tune

## Test

## Plot

## API

### **earthnet_models_pytorch.task**

The task is a PyTorch Lightning module that implements the training, validation and testing loops as well as the optimization and logging necessary. Tasks include:
- spatio-temporal
- pixelwise train-only gradients
- pixelwise test-time gradients and statistical models

**spatio-temporal**

The required data has
- dynamic
- dynamic_mask
- static
- static_mask
- filepath
- cubename
- (landcover)

The model forward takes a `batch` dict, `pred_start` as the first index to be predicted of the first tensor in the dynamic list and `n_preds` as the prediction length.

A setting must implement a metric with a Lightning metrics interface.


### **earthnet_models_pytorch.setting**

The setting is a combination of a Lightning DataModule and a Lightning Metric. Settings include:
- EN21 std
- EN21 veg
- Europe veg

Important is that the respective lists and dicts in the init are filled and that possible global arguments are mapped in the `parse.py`.


### **earthnet_models_pytorch.model**

The model is a PyTorch nn.Module with build-in parser for hyperparameters (much like a Lightning Module). The forward must align with the requirements of the appropriate task.