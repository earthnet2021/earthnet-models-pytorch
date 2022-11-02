# earthnet-models-pytorch

A PyTorch lightning library for Earth surface forecasting.

This library contains models, dataloaders and scripts for Earth surface forecasting in the context of research surrounding the [EarthNet](www.earthnet.tech) challenge.

It is currently under development, thus do expect bugs and please report them!

The library is build on [PyTorch](www.pytorch.org), a Python deep learning library, and [PyTorch Lightning](https://www.pytorchlightning.ai/), a PyTorch wrapper reducing boilerplate code and adding functionality to scale experiments.

In earthnet-models-pytorch there is three main components:

    1. Model - plain PyTorch models just implementing simple forward passes.
    2. Setting - Dataset and Metrics for a particular problem
    3. Task - Abstraction for the training, validation & test loops, tying together models and settings, normally both models and settings are task-specific.


## Requirements

We recommend using Anaconda for managing dependencies of this library. The following bash commands create a suitable environment. Please note the PyTorch installation requirements for your system, see (https://pytorch.org/) - esp. cudatoolkit might have to be installed with a different cuda version.

```
conda create -n emp python=3.10
conda activate emp
conda install -c conda-forge mamba
mamba install -c pytorch -c conda-forge pytorch torchvision torchaudio cudatoolkit=11.3 tensorboard
mamba install -c conda-forge numpy matplotlib pillow xarray zarr netcdf4
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

The design process of a new model or feature is supported in earthnet-models-pytorch by a debug option. Since models often require both a lot of data and GPUs to test the complete training cycle, we use this debug option rather than classic unit testing.
In order to use it, we need to set up a `config.yaml` containing all configs for the different components. See the `configs` folder for examples. It is recommended to save the setting in a folder structure `configs/<setting>/<model>/<feature>/base.yaml`. If done in this way, the earthnet-models-pytorch logger automatically detects the correct naming for later.

We can check if a model works as desired by running:
```
debug.py path/to/setting.yaml
```

It starts with a fast dev run in PyTorch lightning, which is essentially just performing two train, validation and test loops. It then also overfits a model on 4 batches for 1000 epochs, to check if gradients flow properly and the model does indeed learn. 

Note, the debug and all other scripts are registered by PyPI, so it does not matter from which directory they are started and we dont need to use python for them, they should always work.

## Train

In order to train a model we again need to set up a `config.yaml`, see above regarding for more details.

Then we just do:
```
train.py path/to/setting.yaml
```

It trains the model as specified in the config.

## Tune

Hyperparameter tuning. Explanation tbd.

## Test

The script for testing a trained model works as follows:
```
test.py path/to/setting.yaml path/to/checkpoint track --pred_dir path/to/predictions/directory/
```

Here we replace `track` by the track that we want to test on; this depends on the particular setting you choose. For example in `en21-std` there are 4 tracks: `iid`, `ood`, `ex` and `sea`.

## Plot

Plotting functionality. Explanation tbd.

## API

### **earthnet_models_pytorch.task**

The task is a PyTorch Lightning module that implements the training, validation and testing loops as well as the optimization and logging necessary. Tasks include:
- spatio-temporal
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

A setting must implement a metric with a Lightning metrics interface. Note: The Lightning metrics interface as used in this library right now is currently beeing deprecated, thus we will also rework this part.


### **earthnet_models_pytorch.setting**

The setting is a combination of a Lightning DataModule and a Lightning Metric. Settings include:

- `en21-std`; the setting of the EarthNet2021 challenge.
- `en21-veg`; Only predicting vegetation on EarthNet2021 data with additional S2GLC Landcover data.
- `en21x`; The EarthNet2021x data, which is reworked data from the EarthNet2021 challenge, now focusing on vegetation forecasting in Europe.
- `en21x-px`; Same as `en21x`, but using pixelwise data from a `.csv` to efficiently create batches for pixelwise models.
- `en22`; The EarthNet2022 data from the DeepCube UC1 project. Similar to `en21x`.


Important is that the respective lists and dicts in the init are filled and that possible global arguments are mapped in the `parse.py`.


### **earthnet_models_pytorch.model**

The model is a PyTorch nn.Module with build-in parser for hyperparameters (much like a Lightning Module). The forward must align with the requirements of the appropriate task.
