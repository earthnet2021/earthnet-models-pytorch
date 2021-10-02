"""LocalRNN
"""

from typing import Optional, Union

import argparse
import ast

import timm
import torch
import torchvision

import segmentation_models_pytorch as smp

from torch import nn

from earthnet_models_pytorch.utils import str2bool



class LocalRNN(nn.Module):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams

        self.ndvi_pred = (hparams.setting in ["en21-veg", "europe-veg", "en21x"])

        self.state_encoder = getattr(smp, self.hparams.state_encoder_name)(**self.hparams.state_encoder_args)

        self.update_encoder = timm.create_model(self.hparams.update_encoder_name, pretrained=True, in_chans = self.hparams.update_encoder_inchannels, num_classes=self.hparams.update_encoder_nclasses)

        self.rnn = torch.nn.GRU(input_size = self.hparams.update_encoder_nclasses, hidden_size = self.hparams.state_encoder_args["classes"], num_layers = 1, batch_first = True)

        self.head = torch.nn.Linear(in_features = self.hparams.state_encoder_args["classes"], out_features = 1 if self.ndvi_pred else 4)

        self.sigmoid = nn.Sigmoid()



    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--state_encoder_name", type = str, default = "FPN")
        parser.add_argument("--state_encoder_args", type = ast.literal_eval, default = '{"encoder_name": "timm-efficientnet-b4", "encoder_weights": "noisy-student", "in_channels": 191, "classes": 256}')
        parser.add_argument("--update_encoder_name", type = str, default = "efficientnet_b1")
        parser.add_argument("--update_encoder_inchannels", type = int, default = 28)
        parser.add_argument("--update_encoder_nclasses", type = int, default = 128)
        parser.add_argument("--setting", type = str, default = "en21x")
        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)

        return parser


    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        
        n_preds = 0 if n_preds is None else n_preds

        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]

        b, t, c, h, w = hr_dynamic_inputs.shape

        hr_dynamic_inputs = hr_dynamic_inputs.reshape(b, t*c, h, w)

        static_inputs = data["static"][0]

        state_inputs = torch.cat((hr_dynamic_inputs, static_inputs), dim = 1)

        state = self.state_encoder(state_inputs)

        _, c_s, _, _ = state.shape

        state = state.reshape(b, c_s, h * w).transpose(1,2).reshape(1, b*h*w, c_s)

        meso_dynamic_inputs = data["dynamic"][1][:,c_l:,...]

        _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape

        meso_dynamic_inputs = meso_dynamic_inputs.reshape(b*t_m, c_m, h_m, w_m)

        update = self.update_encoder(meso_dynamic_inputs)

        _, c_u = update.shape

        update = update.reshape(b,t_m,c_u).unsqueeze(1).repeat(1,h*w, 1, 1).reshape(b*h*w,t_m,c_u)

        out, _ = self.rnn(update.contiguous(), state.contiguous())

        out = self.sigmoid(self.head(out))

        _, _, c_o = out.shape

        out = out.reshape(b,h*w,t_m,c_o).reshape(b,h,w,t_m,c_o).permute(0,3,4,1,2)

        return out, {}















