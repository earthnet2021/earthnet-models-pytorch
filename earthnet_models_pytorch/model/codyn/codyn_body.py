"""

Design principles:

Modules that do nothing are "None", save checking via if None

submodules have input_shapes, output_shapes and get_shapes attributes

shapes ist ein eigener datentyp

dynamic encoders have output_skips attribute


"""

from typing import Optional, Union

import argparse
import ast
import copy

import torch

from torch import nn

from earthnet_models_pytorch.model.codyn.decode import setup_decoder
from earthnet_models_pytorch.model.codyn.dynamics import setup_dynamics
from earthnet_models_pytorch.model.codyn.encode import setup_encoders
from earthnet_models_pytorch.model.codyn.extract import setup_extraction
from earthnet_models_pytorch.model.codyn.fuse import setup_fusion

from earthnet_models_pytorch.utils import str2bool


class CodynBody(nn.Module):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = copy.deepcopy(hparams)

        self.dynamic_encoders = nn.ModuleList(setup_encoders(self.hparams.dynamic_encoders))

        self.static_encoders = nn.ModuleList(setup_encoders(self.hparams.static_encoders))

        self.content_extraction = setup_extraction(self.hparams.content_extraction)

        self.content_fusion = setup_fusion(self.hparams.content_fusion)

        self.dynamics_core = setup_dynamics(self.hparams.dynamics_core)

        self.decoder = setup_decoder(self.hparams.decoder)


    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--dynamic_encoders", type = ast.literal_eval, default = '[{"name": "vgg128", "args": {"in_channels": 4, "latent_channels": 128, "in_filters": 32, "output_skips": True}, "name": "4x80", "args": {"in_channels": 6, "latent_channels": 64, "in_filters": 16, "output_skips": False}}]')

        parser.add_argument("--static_encoders", type = ast.literal_eval, default = '[{"name": "vgg128", "args": {"in_channels": 1, "latent_channels": 128, "in_filters": 32, "output_skips": True}}]')

        parser.add_argument("--content_extraction", type = ast.literal_eval, default = '{"name": "sum", "args": {"context_length": 10}}')

        parser.add_argument("--content_fusion", type = ast.literal_eval, default = '{}')

        parser.add_argument("--dynamics_core", type = ast.literal_eval, default = '{}')

        parser.add_argument("--decoder", type = ast.literal_eval, default = '{"name": "vgg128", "args": {"output_channels": 4, "latent_channels": 384, "last_filters": 32, "skip": True}}')

        parser.add_argument("--skip_dropout", type = str2bool, default = False)

        parser.add_argument("--noisy_pixel_mask", type = str2bool, default = False)
        parser.add_argument("--ndvi_pred", type = str2bool, default = False)

        return parser        

    def forward(self, data: dict, pred_start: int = 0, n_preds: Optional[int] = None, p_dropout: float = 0):
        assert(len(self.dynamic_encoders)<=len(data["dynamic"]))

        dynamic_encodings, dynamic_content_skips, dynamic_mask_encodings, dynamic_mask_skips = [], [], [], []
        for idx, encoder in enumerate(self.dynamic_encoders):
            dynamic_input = data["dynamic"][idx]
            if idx < len(data["dynamic_mask"]):
                dynamic_mask = data["dynamic_mask"][idx]
                if self.hparams.noisy_pixel_mask:
                    b, t, c, h, w = dynamic_input.shape
                    for i in range(b):
                        all_pixels = dynamic_input[i,...].permute(1,0,2,3)
                        all_pixels = all_pixels[dynamic_mask[i,...].permute(1,0,2,3) == 1].reshape(c,-1)
                        all_pixels = torch.cat(int(1+dynamic_input[i,...].nelement()/all_pixels.nelement())*[all_pixels], dim = -1)
                        all_pixels = all_pixels[:,torch.randperm(all_pixels.shape[-1], device = dynamic_input.device)]
                        all_pixels = all_pixels[:,:t*h*w].reshape(c,t,h,w).permute(1,0,2,3)
                        dynamic_input[i,...] = torch.where(dynamic_mask[i,...] == 0, all_pixels, dynamic_input[i,...])
            else:
                dynamic_mask = None
            
            if self.hparams.ndvi_pred and idx == 0:
                if dynamic_input.shape[2] != 1:
                    dynamic_input = torch.cat([((dynamic_input[:,:,3,...] - dynamic_input[:,:,2,...])/(dynamic_input[:,:,3,...] + dynamic_input[:,:,2,...] + 1e-6)).unsqueeze(2),dynamic_input], dim = 2)
                    if dynamic_mask is not None:
                        dynamic_mask = torch.cat([dynamic_mask, dynamic_mask[:,:,0,...].unsqueeze(2)], dim = 2)

            tmp_content_encoding, tmp_content_skips, tmp_mask_encoding, tmp_mask_skips = encoder(dynamic_input, dynamic_mask)
            dynamic_encodings.append(tmp_content_encoding)
            if encoder.output_skips:
                dynamic_content_skips.append(tmp_content_skips)
                dynamic_mask_encodings.append(tmp_mask_encoding)
                dynamic_mask_skips.append(tmp_mask_skips) # Encoder shall return None mask if not used...

        static_content_encodings, static_content_skips, static_mask_encodings, static_mask_skips = [], [], [], []
        for idx, encoder in enumerate(self.static_encoders):
            static_input = data["static"][idx]
            if idx < len(data["static_mask"]):
                static_mask = data["static_mask"][idx]
            else:
                static_mask = None
            tmp_content_encoding, tmp_content_skips, tmp_mask_encoding, tmp_mask_skips = encoder(static_input, static_mask)

            static_content_encodings.append(tmp_content_encoding)
            static_content_skips.append(tmp_content_skips)
            static_mask_encodings.append(tmp_mask_encoding)
            static_mask_skips.append(tmp_mask_skips)
        
        extracted_content_encoding, extracted_content_skips, extracted_mask_encoding, extracted_mask_skips = self.content_extraction(enc = [dynamic_encodings[j] for j in range(len(self.dynamic_encoders)) if self.dynamic_encoders[j].output_skips], skips = dynamic_content_skips, mask_enc = dynamic_mask_encodings, mask_skips = dynamic_mask_skips)
        
        content_encoding, content_skips = self.content_fusion(extracted_content_encoding, extracted_content_skips, extracted_mask_encoding, extracted_mask_skips, static_content_encodings, static_content_skips, static_mask_encodings, static_mask_skips)

        # HERE SKIPS DROPOUT
        if self.hparams.skip_dropout and self.training:
            content_skips = [(torch.rand(1, device = c.device) >= p_dropout) * c for c in content_skips]
            

        dynamics, aux_loss_pars = self.dynamics_core(dynamic_encodings, pred_start = pred_start, n_preds = n_preds)

        preds = self.decoder(content_skips, content_encoding, dynamics)

        return preds, aux_loss_pars
