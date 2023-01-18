"""ContextConvLSTM
"""

from turtle import forward
from typing import Optional, Union, List

import argparse
import ast
from grpc import dynamic_ssl_server_credentials
from pip import main
import torch.nn as nn
import torch
import sys
from earthnet_models_pytorch.utils import str2bool


class ContextConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # 4 for the 4 split in the ConvLSTM
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class CNN(nn.Module):
    def __init__(self, kernel_size, bias):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=24,  # TODO nb of weather variables
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.Sigmoid(),
        )
        self.norm = torch.nn.BatchNorm2d(24)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=24,
                out_channels=12,  # TODO nb of weather variables
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=2,  # TODO nb of weather variables
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.ReLU(),
        )

    def forward(self, topology, weather):
        x = self.conv1(topology)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_features[0], dim_features[1]),
            nn.ReLU(),
            nn.Linear(dim_features[1], dim_features[2]),
            nn.ReLU(),
            nn.Linear(dim_features[2], dim_features[3]),
            nn.ReLU(),
        )

    def forward(self, weather):
        return self.model(weather)


class ContextConvLSTM(nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams

        self.ndvi_pred = hparams.setting in ["en21-veg", "europe-veg", "en21x", "en22"]

        input_encoder = 1  # # TODO find a better solution nb of channel 39 = 5 r,b,g, nr, ndvi + 1 dem + 33 weather
        input_decoder = 1  # 33 weather

        if self.hparams.method == "CNN":
            self.CNN = CNN(kernel_size=self.hparams.kernel_size, bias=self.hparams.bias)
            input_encoder = 3
            input_decoder = 3

        elif hparams.method == "MLP":
            dim_features = [24, 24, 10, 1]
            self.MLP = MLP(dim_features)
            input_encoder = 2
            input_decoder = 2

        if self.hparams.input == "NDVI+T":
            input_encoder = 2
            input_decoder = 2
        elif self.hparams.input == "NDVI+T+W":
            # input_encoder = 1 + 1 + 15
            # input_decoder = 1 + 1 + 15
            input_encoder = 1 + 1 + 24
            input_decoder = 1 + 1 + 24
        elif self.hparams.input == "NDVI+T+W+lc":
            input_encoder = 1 + 1 + 24 + 1
            input_decoder = 1 + 1 + 24 + 1
        elif self.hparams.input == "RGBNR":
            input_encoder = 5 + 25
            input_decoder = 1 + 25
        else:
            input_encoder = 1
            input_decoder = 1

        self.encoder_1_convlstm = ContextConvLSTMCell(
            input_dim=input_encoder,
            hidden_dim=self.hparams.hidden_dim[0],
            kernel_size=self.hparams.kernel_size,
            bias=self.hparams.bias,
        )

        self.encoder_2_convlstm = ContextConvLSTMCell(
            input_dim=self.hparams.hidden_dim[0],
            hidden_dim=self.hparams.hidden_dim[1],
            kernel_size=self.hparams.kernel_size,
            bias=self.hparams.bias,
        )

        self.decoder_1_convlstm = ContextConvLSTMCell(
            input_dim=input_decoder,
            hidden_dim=self.hparams.hidden_dim[0],
            kernel_size=self.hparams.kernel_size,
            bias=self.hparams.bias,
        )

        self.decoder_2_convlstm = ContextConvLSTMCell(
            input_dim=self.hparams.hidden_dim[0],
            hidden_dim=self.hparams.hidden_dim[1],
            kernel_size=self.hparams.kernel_size,
            bias=self.hparams.bias,
        )

        padding = self.hparams.kernel_size // 2, self.hparams.kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.hparams.hidden_dim[1],
            out_channels=1,
            kernel_size=self.hparams.kernel_size,
            padding=padding,
            bias=self.hparams.bias,
        )

        self.activation_output = nn.Sigmoid()

        if hparams.method == "bigger":
            self.encoder_3_convlstm = ContextConvLSTMCell(
                input_dim=self.hparams.hidden_dim[1],
                hidden_dim=self.hparams.hidden_dim[2],
                kernel_size=self.hparams.kernel_size,
                bias=self.hparams.bias,
            )
            self.encoder_4_convlstm = ContextConvLSTMCell(
                input_dim=self.hparams.hidden_dim[2],
                hidden_dim=self.hparams.hidden_dim[3],
                kernel_size=self.hparams.kernel_size,
                bias=self.hparams.bias,
            )

            self.decoder_3_convlstm = ContextConvLSTMCell(
                input_dim=self.hparams.hidden_dim[1],
                hidden_dim=self.hparams.hidden_dim[2],
                kernel_size=self.hparams.kernel_size,
                bias=self.hparams.bias,
            )
            self.decoder_4_convlstm = ContextConvLSTMCell(
                input_dim=self.hparams.hidden_dim[2],
                hidden_dim=self.hparams.hidden_dim[3],
                kernel_size=self.hparams.kernel_size,
                bias=self.hparams.bias,
            )

            self.conv = nn.Conv2d(
                in_channels=self.hparams.hidden_dim[3],
                out_channels=1,
                kernel_size=self.hparams.kernel_size,
                padding=padding,
                bias=self.hparams.bias,
            )

    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        # TODO remove the useless features
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        # parser.add_argument("--input_dim", type=int, default=6)
        parser.add_argument(
            "--hidden_dim", type=ast.literal_eval, default=[64, 64, 64, 64]
        )  # TODO find a better type ? list(int)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--bias", type=str2bool, default=True)
        parser.add_argument("--return_all_layers", type=str2bool, default=False)
        parser.add_argument("--setting", type=str, default="en22")
        parser.add_argument("--context_length", type=int, default=9)
        parser.add_argument("--target_length", type=int, default=36)
        parser.add_argument("--lc_min", type=int, default=82)
        parser.add_argument("--lc_max", type=int, default=104)
        parser.add_argument("--method", type=str, default=None)
        parser.add_argument("--input", type=str, default=None)
        parser.add_argument("--skip_connections", type=str2bool, default=False)
        parser.add_argument("--add_conv", type=str2bool, default=False)
        return parser

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):

        # c_l = self.hparams.context_length if self.training else pred_start
        c_l = 35
        # Data
        hr_dynamics = data["dynamic"][0][
            :, (c_l - self.hparams.context_length) : c_l, ...
        ]
        if self.hparams.input == "RGBNR":
            target = hr_dynamics[:, :, :, ...]
        else:
            target = hr_dynamics[:, :, 0, ...].unsqueeze(2)
        weather = data["dynamic"][1].unsqueeze(3).unsqueeze(4)
        topology = data["static"][0]
        landcover = data["landcover"]

        # Shape
        b, t, _, h, w = data["dynamic"][0].shape

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, height=h, width=w)
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(
            batch_size=b, height=h, width=w
        )
        if self.hparams.method == "bigger":
            h_t3, c_t3 = self.encoder_3_convlstm.init_hidden(
                batch_size=b, height=h, width=w
            )
            h_t4, c_t4 = self.encoder_4_convlstm.init_hidden(
                batch_size=b, height=h, width=w
            )
        output = []

        # encoding network
        for t in range(self.hparams.context_length):

            if self.hparams.input == "NDVI+T":
                input = torch.cat((target[:, t, ...], topology), dim=1)
            elif self.hparams.input == "NDVI+T+W":
                weather_t = weather[:, t, ...].repeat(1, 1, 128, 128)
                input = torch.cat((target[:, t, ...], topology), dim=1)
                input = torch.cat((input, weather_t), dim=1)
            elif self.hparams.input == "NDVI+T+W+lc":
                weather_t = weather[:, t, ...].repeat(1, 1, 128, 128)
                input = torch.cat((target[:, t, ...], topology), dim=1)
                input = torch.cat((input, landcover), dim=1)
                input = torch.cat((input, weather_t), dim=1)
            elif self.hparams.input == "RGBNR":
                weather_t = weather[:, t, ...].repeat(1, 1, 128, 128)
                input = torch.cat((target[:, t, ...], topology), dim=1)
                input = torch.cat((input, weather_t), dim=1)
            else:
                input = target[:, t, :, :]

            # First block
            h_t, c_t = self.encoder_1_convlstm(input_tensor=input, cur_state=[h_t, c_t])

            # second block
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )

            if self.hparams.method == "bigger":
                h_t3, c_t3 = self.encoder_3_convlstm(
                    input_tensor=h_t2, cur_state=[h_t3, c_t3]
                )
                h_t4, c_t4 = self.encoder_4_convlstm(
                    input_tensor=h_t3, cur_state=[h_t4, c_t4]
                )

        # First prediction
        if self.hparams.method == "bigger":
            pred = self.conv(h_t4)
        else:
            pred = self.conv(h_t2)

        if self.hparams.skip_connections:
            if self.hparams.input == "RGBNR":

                pred = pred + target[:, -1, 0, ...].unsqueeze(1)
            else:
                pred = pred + target[:, -1, ...]

        if self.hparams.method == "MLP":
            pred = pred + self.MLP(weather[:, t, ...].squeeze(3).squeeze(2)).unsqueeze(
                2
            ).unsqueeze(3).repeat(1, 1, 128, 128)

        pred = self.activation_output(pred)

        # forecasting network
        for t in range(self.hparams.target_length):
            # for skip connection
            pred_previous = torch.clone(pred)

            # Input
            if self.hparams.input == "NDVI+T":
                pred = torch.cat((pred, topology), dim=1)
            elif self.hparams.input == "NDVI+T+W":
                weather_t = weather[:, c_l + t, ...].repeat(1, 1, 128, 128)
                pred = torch.cat((pred, topology), dim=1)
                pred = torch.cat((pred, weather_t), dim=1)
            elif self.hparams.input == "NDVI+T+W+lc":
                weather_t = weather[:, c_l + t, ...].repeat(1, 1, 128, 128)
                pred = torch.cat((pred, landcover), dim=1)
                pred = torch.cat((pred, topology), dim=1)
                pred = torch.cat((pred, weather_t), dim=1)
            elif self.hparams.input == "RGBNR":
                weather_t = weather[:, c_l + t, ...].repeat(1, 1, 128, 128)
                pred = torch.cat((pred, topology), dim=1)
                pred = torch.cat((pred, weather_t), dim=1)

            # first block
            if self.hparams.method == "stack":
                h_t, c_t = self.encoder_1_convlstm(
                    input_tensor=pred, cur_state=[h_t, c_t]
                )
                # Second block
                h_t2, c_t2 = self.encoder_2_convlstm(
                    input_tensor=h_t, cur_state=[h_t2, c_t2]
                )
            else:
                h_t, c_t = self.decoder_1_convlstm(
                    input_tensor=pred, cur_state=[h_t, c_t]
                )

                # Second block
                h_t2, c_t2 = self.decoder_2_convlstm(
                    input_tensor=h_t, cur_state=[h_t2, c_t2]
                )

            pred = h_t2

            if self.hparams.method == "bigger":
                h_t3, c_t3 = self.decoder_3_convlstm(
                    input_tensor=h_t2, cur_state=[h_t3, c_t3]
                )
                h_t4, c_t4 = self.decoder_4_convlstm(
                    input_tensor=h_t3, cur_state=[h_t4, c_t4]
                )
                pred = h_t4

            pred = self.conv(pred)

            if self.hparams.skip_connections:
                pred = pred + pred_previous

            if self.hparams.method == "MLP":
                pred = pred + self.MLP(
                    weather[:, c_l + t, ...].squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)

            # Output
            pred = self.activation_output(pred)
            output += [pred.unsqueeze(1)]

        output = torch.cat(output, dim=1)  # .unsqueeze(2)
        return output, {}
