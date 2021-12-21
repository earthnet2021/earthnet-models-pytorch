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
from earthnet_models_pytorch.model.rnn import MLP


class LocalRNN(nn.Module):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams
        
        # ndvi_pred true if task is NDVI prediction instead of RGBNir)
        self.ndvi_pred = (hparams.setting in ["en21-veg", "europe-veg", "en21x", "en22"])
        
        # Instantiates a model from segmentation_models_pytorch as state encoder
        # Uses hparams from config.yalm e.g. "state_encoder_name: FPN, state_encoder_args: encoder_name: timm-efficientnet-b3 (...)"
        self.state_encoder = getattr(smp, self.hparams.state_encoder_name)(**self.hparams.state_encoder_args)
        
        # If model is MLP uses model.rr.MLP class, else creates model from timm
        if self.hparams.update_encoder_name == "MLP":
            # Instantiates the update_model from model.rrn.mlp
            # ninp: number of inputs, nhid: number of dimmensions in hidden layers, nout: number of outputs
            self.update_encoder = MLP(ninp = self.hparams.update_encoder_inchannels, nhid = 128, nout = self.hparams.update_encoder_nclasses, nlayers = 2, activation = "relu")
            hidden_size = self.hparams.state_encoder_args["classes"] - 64 # RNN hidden size is 64 dims smaller than number of outputs of state encoder. ??1
            in_features = hidden_size
        else:
            # Creates model from timm using the hparms given
            self.update_encoder = timm.create_model(self.hparams.update_encoder_name, pretrained=True, in_chans = self.hparams.update_encoder_inchannels, num_classes=self.hparams.update_encoder_nclasses - self.hparams.update_encoder_inchannels)
            hidden_size = self.hparams.state_encoder_args["classes"] # as many dimmensions in the hidden layers as state_encoder outputs ??
            in_features = 2*self.hparams.state_encoder_args["classes"] # Twice the input features as state_encoder outputs ??
        
        # Recurrent model as Gated recurrent unit, input dimms are update_encoder output dimms
        self.rnn = torch.nn.GRU(input_size = self.hparams.update_encoder_nclasses, hidden_size = hidden_size, num_layers = 1, batch_first = True)
        
        # Output layer. 1 feature channel if NDVI, 4 if RGBNir
        self.head = torch.nn.Linear(in_features = in_features, out_features = 1 if self.ndvi_pred else 4)

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
        parser.add_argument("--train_lstm_npixels", type = int, default = 256)
        parser.add_argument("--setting", type = str, default = "en21x")
        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)
        parser.add_argument("--use_dem", type = str2bool, default = True)
        parser.add_argument("--use_soilgrids", type = str2bool, default = True)
        parser.add_argument("--lc_min", type = int, default = 82)
        parser.add_argument("--lc_max", type = int, default = 104)
        parser.add_argument("--val_n_splits", type = int, default = 20)

        return parser


    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        '''
        Model foward pass behaviour
        '''
        
        # Number of predictions, by default 1.
        n_preds = 0 if n_preds is None else n_preds
    
        # Context_length is that defined by task if mode istraining, otherwise is pred_start (default 0)
        c_l = self.hparams.context_length if self.training else pred_start
        
        # High resolution [0] dynamic inputs only to be used from t 0 to t context_lenght
        hr_dynamic_inputs = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]
        
        # last frame of spectral inputs before prediction_length
        last_dynamic_input = hr_dynamic_inputs[:,-1,...]
        
        # batch, time, channels, height, width of high-res dynamic inputs
        b, t, c, h, w = hr_dynamic_inputs.shape
        
        # Flattens time and channels for High-res inputs
        hr_dynamic_inputs = hr_dynamic_inputs.reshape(b, t*c, h, w)
        
        # high res static inputs
        static_inputs = data["static"][0]
        
        # Concatenates DEM and Soil to High-res dynamic
        if self.hparams.use_dem and self.hparams.use_soilgrids:
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs), dim = 1)
        elif self.hparams.use_dem:
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs[:,0,...][:,None,...]), dim = 1)
        elif self.hparams.use_soilgrids:
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs[:,1:,...]), dim = 1)
        else:
            state_inputs = hr_dynamic_inputs
        
        # Feeds context (highres dynamic and static) to the state encoder.
        state = self.state_encoder(state_inputs)
        
        # Last state frame replaced by last high-ress input ??
        state[:,-1,...] = last_dynamic_input[:,0,...]
        
        # c_s : channels * time of context state
        _, c_s, _, _ = state.shape
        
        # Determine whether the dataset low-res dynamic variables [1] *meteo* are spatial (5 dims) or scalar (3 dims)
        meso_dynamic_inputs = data["dynamic"][1][:,c_l:,...]
        if len(meso_dynamic_inputs.shape) == 5:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape
        else:
            _, t_m, c_m = meso_dynamic_inputs.shape

        # MLP as update_encoder
        if self.hparams.update_encoder_name == "MLP":
            
            # If meso_dynamics has spatial dims, reduces to center value in height and width
            if len(meso_dynamic_inputs.shape) == 5:
                meso_dynamic_inputs = meso_dynamic_inputs[:,:,:,(h_m//2- 1):(h_m//2),(w_m//2- 1):(w_m//2)].mean((3,4))
                
            # meso_dynamic variables are spatially expanded&repeated to match high-res variables
            meso_dynamic_inputs = meso_dynamic_inputs.reshape(b, 1, 1, t_m, c_m).repeat(1, h, w, 1, 1).reshape(b*h*w,t_m, c_m)
            
            # location_input is spatio-temporal_context_input, limited to 64 features, repeated for temporal lenght of mesodynamics, reshaped to batch*height*width, time, 64 features ?? are these related to ??1
            location_input = state[:,None,:64,:,:].repeat(1,t_m, 1, 1,1).permute(0,3,4,1,2).reshape(b*h*w, t_m, 64)
            
            # concatenates spatio-temporal_context_input and meso_dynamic_input
            update_inputs = torch.cat([meso_dynamic_inputs,location_input], dim = -1)
            
            # state as [1,batch*height*width,channels*time]
            state = state.reshape(b, c_s, h * w).transpose(1,2).reshape(1, b*h*w, c_s)[:,:,64:]
            
            # if mode is training detect vegetated pixels from lc, if too few select 4, leave these for training the model (update_inputs) ??
            if self.training:
                # idxs tensor of randomly premuted integers from 0 to b*h*w to uniquely identify pixels
                idxs = torch.randperm(b*h*w).type_as(state).long()
                
                # what is lc now ??
                lc = data["landcover"].reshape(-1)
                
                # idxs  ??
                idxs = idxs[(lc >= self.hparams.lc_min).byte() & (lc <= self.hparams.lc_max).byte()][:self.hparams.train_lstm_npixels*b]
                # idxs = torch.randint(low = 0, high = b*h*w, size = (self.hparams.train_lstm_npixels*b, )).type_as(update).long()
                if len(idxs) == 0:
                    print(f"Detected cube without vegetation: {data['cubename']}")
                    idxs = [1,2,3,4]

                update_inputs = update_inputs[idxs, :, :]

                state = state[:,idxs,:]

            update = self.update_encoder(update_inputs)

        # If MLP is not the update_encoder
        else:
            # reshapes meso_dynamic to batch*time
            meso_dynamic_inputs = meso_dynamic_inputs.reshape(b*t_m, c_m, h_m, w_m)
            
            # feeds the update_encoder with meso_dynamic_inputs ...
            update = self.update_encoder(meso_dynamic_inputs)
            
            # ... yet concats the model output with the meso_dynamic_input center pixel value ??
            update = torch.cat([update, meso_dynamic_inputs[:,:,(h_m//2- 1):(h_m//2),(w_m//2- 1):(w_m//2)].mean((2,3))], dim = 1)

            _, c_u = update.shape

            update = update.reshape(b,t_m,c_u).unsqueeze(1).repeat(1,h*w, 1, 1).reshape(b*h*w,t_m,c_u)

            state = state.reshape(b, c_s, h * w).transpose(1,2).reshape(1, b*h*w, c_s)
            
            # repeated identical task, move this and line 156-175 into a function?
            if self.training:
                idxs = torch.randperm(b*h*w).type_as(update).long()
                lc = data["landcover"].reshape(-1)
                idxs = idxs[(lc >= self.hparams.lc_min).byte() & (lc <= self.hparams.lc_max).byte()][:self.hparams.train_lstm_npixels*b]
                # idxs = torch.randint(low = 0, high = b*h*w, size = (self.hparams.train_lstm_npixels*b, )).type_as(update).long()
                if len(idxs) == 0:
                    print(f"Detected cube without vegetation: {data['cubename']}")
                    idxs = [1,2,3,4]

                state = state[:,idxs,:]
                update = update[idxs,:,:]

        if self.training:
            out, _ = self.rnn(update.contiguous(), state.contiguous())
        else:
            out_arr = []
            states = torch.chunk(state, self.hparams.val_n_splits, dim = 1)
            updates = torch.chunk(update, self.hparams.val_n_splits, dim = 0)
            
            # contiguous() returns a copy of the tensor as if it had been perfectly copied in memory ??
            for i in range(self.hparams.val_n_splits):
                out_arr.append(self.rnn(updates[i].contiguous(),states[i].contiguous())[0])
            out = torch.cat(out_arr, dim = 0)

        if not self.hparams.update_encoder_name == "MLP":
            out = torch.cat([out,state.repeat(t_m,1,1).permute(1,0,2)], dim = -1)

        out = self.sigmoid(self.head(out))

        _, _, c_o = out.shape
        
        # only output predictions for pixels with vegetation, others are -1
        if self.training:

            tmp_out = -torch.ones((b*h*w, t_m, c_o)).type_as(out)
            tmp_out[idxs, :, :] = out
            
            out = tmp_out
        
        # reshapes and permutes back to [batch, time, channels, height, width]
        out = out.reshape(b,h*w,t_m,c_o).reshape(b,h,w,t_m,c_o).permute(0,3,4,1,2)

        return out, {}















