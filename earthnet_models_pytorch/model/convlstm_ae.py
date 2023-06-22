"""ConvLSTM_ae
    ConvLSTM with an encoding-decoding architecture
"""
from typing import Optional, Union

import argparse
import ast
import torch.nn as nn
import torch
from earthnet_models_pytorch.utils import str2bool


class ConvLSTMCell(nn.Module):
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


class ConvLSTMAE(nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams

        self.encoder_1_convlstm = ConvLSTMCell(
            input_dim=self.hparams.num_inputs,
            hidden_dim=self.hparams.hidden_dim[0],
            kernel_size=self.hparams.kernel_size,
            bias=self.hparams.bias,
        )

        self.encoder_2_convlstm = ConvLSTMCell(
            input_dim=self.hparams.hidden_dim[0],
            hidden_dim=self.hparams.hidden_dim[1],
            kernel_size=self.hparams.kernel_size,
            bias=self.hparams.bias,
        )

        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=42, #self.hparams.num_inputs,
            hidden_dim=self.hparams.hidden_dim[0],
            kernel_size=self.hparams.kernel_size,
            bias=self.hparams.bias,
        )

        self.decoder_2_convlstm = ConvLSTMCell(
            input_dim=self.hparams.hidden_dim[0],
            hidden_dim=self.hparams.hidden_dim[1],
            kernel_size=self.hparams.kernel_size,
            bias=self.hparams.bias,
        )

        padding = self.hparams.kernel_size // 2, self.hparams.kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.hparams.hidden_dim[1],
            out_channels=self.hparams.num_outputs,
            kernel_size=self.hparams.kernel_size,
            padding=padding,
            bias=self.hparams.bias,
        )

        self.activation_output = nn.Sigmoid()

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

        parser.add_argument(
            "--hidden_dim", type=ast.literal_eval, default=[64, 64, 64, 64]
        )  
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--bias", type=str2bool, default=True)
        parser.add_argument("--setting", type=str, default="en22")
        parser.add_argument("--num_inputs", type=int, default=5 + 3 + 24)
        parser.add_argument("--num_outputs", type=int, default=1)
        parser.add_argument("--context_length", type=int, default=9)
        parser.add_argument("--target_length", type=int, default=36)
        parser.add_argument("--skip_connections", type=str2bool, default=False)
        return parser

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):

        context_length = self.hparams.context_length if self.training or (pred_start < self.hparams.context_length) else pred_start

        # k = torch.tensor([1])
        # TODO definir correctement la valeur de k + regarder shape decay k = max step ? + remove teacher forcing dans val et test
        # teacher_forcing_decay = k / (k + torch.exp(step / k))
        # teacher_forcing = torch.bernoulli(teacher_forcing_decay)

        # Data
        # sentinel 2 bands
        sentinel = data["dynamic"][0]
        weather = data["dynamic"][1].unsqueeze(3).unsqueeze(4)
        static = data["static"][0] if data["static"][0].ndim == 4 else data["static"][0][... ,0]

        # Shape: batch size, temporal size, number of channels, height, width
        b, t, _, h, w = sentinel.shape

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, height=h, width=w)
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(
            batch_size=b, height=h, width=w
        )

        output = []

        # Encoding network
        for t in range(context_length):
            input = torch.cat((sentinel[:, t, ...], static), dim=1)
            # WARNING only for En23
            weather_t = (
                weather[:, t : t + 5, ...]
                .view(weather.shape[0], 1, -1, 1, 1)
                .squeeze(1)
                .repeat(1, 1, 128, 128)
            )
            input = torch.cat((input, weather_t), dim=1)
            # First block
            h_t, c_t = self.encoder_1_convlstm(input_tensor=input, cur_state=[h_t, c_t])
            # second block
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )

        # First prediction
        pred = self.conv(h_t2)
        # add the last frame of the context period
        if self.hparams.skip_connections:
            pred = pred + sentinel[:, t, 0, ...].unsqueeze(1)

        pred = self.activation_output(pred)
        # forecasting network
        for t in range(self.hparams.target_length):
            # if teacher_forcing:
            #    pred = target[:, c_l + t, ...]

            # Skip connection
            pred_previous = torch.clone(pred)

            # Input
            weather_t = (
                weather[:, context_length + t : context_length + t + 5, ...]
                .view(weather.shape[0], 1, -1, 1, 1)
                .squeeze(1)
                .repeat(1, 1, 128, 128)
            )
            pred = torch.cat((pred, static), dim=1)
            pred = torch.cat((pred, weather_t), dim=1)
            # first block
            h_t, c_t = self.decoder_1_convlstm(input_tensor=pred, cur_state=[h_t, c_t])

            # Second block
            h_t2, c_t2 = self.decoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )

            pred = h_t2
            pred = self.conv(pred)
            if self.hparams.skip_connections:
                pred = pred + pred_previous

            # Output
            pred = (self.activation_output(pred) - 0.5) * 2
            output += [pred.unsqueeze(1)]
        output = torch.cat(output, dim=1)  # .unsqueeze(2)
        return output, {}
