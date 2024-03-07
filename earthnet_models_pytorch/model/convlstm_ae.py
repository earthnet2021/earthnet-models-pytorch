"""ConvLSTM_ae
    ConvLSTM with an encoding-decoding architecture
"""

import argparse
import ast
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
from earthnet_models_pytorch.model.layer_utils import inverse_permutation
from earthnet_models_pytorch.utils import str2bool

# Mapping of class labels to indices
class_mapping = {
    10: 0,
    20: 1,
    30: 2,
    40: 3,
    50: 4,
    60: 5,
    70: 6,
    80: 7,
    90: 8,
    95: 9,
    100: 10,
}


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
        """
        Forward pass of the ConvLSTM cell.
        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor.
        cur_state: tuple
            Tuple containing the current hidden state (h) and cell state (c).

        Returns
        -------
        torch.Tensor
            Next hidden state (h_next).
        torch.Tensor
            Next cell state (c_next).
        """

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
        """
        Initialize the hidden and cell states with zeros.
        Parameters
        ----------
        batch_size: int
            Batch size.
        height: int
            Height of the input tensor.
        width: int
            Width of the input tensor.

        Returns
        -------
        torch.Tensor
            Initial hidden state (h) filled with zeros.
        torch.Tensor
            Initial cell state (c) filled with zeros.
        """
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


# SigmoidRescaler class to rescale output between -1 and 1
class SigmoidRescaler(nn.Module):
    def __init__(self):
        super(SigmoidRescaler, self).__init__()

    def forward(self, x):
        # Apply the sigmoid function
        sigmoid_output = torch.sigmoid(x)

        # Rescale the sigmoid output between -1 and 1
        rescaled_output = 2 * sigmoid_output - 1

        return rescaled_output


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
            input_dim=(
                self.hparams.num_inputs - 4
                if self.hparams.decoder_input_subtract_s2bands
                else self.hparams.num_inputs
            ),  # nb of s2 bands.
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

        if self.hparams.target == "ndvi":
            self.activation_output = nn.Sigmoid()
        elif self.hparams.target == "anomalie_ndvi":
            self.activation_output = SigmoidRescaler()
        else:
            KeyError("The target is not defined.")

    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        """
        Add model-specific arguments to the command-line argument parser.

        Parameters
        ----------
        parent_parser: Optional[Union[argparse.ArgumentParser, list]]
            Parent argument parser (optional).

        Returns
        -------
        argparse.ArgumentParser
            Argument parser with added model-specific arguments.
        """
        # Create a new argument parser or use the parent parser
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        # Add model-specific arguments
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
        parser.add_argument("--teacher_forcing", type=str2bool, default=False)
        parser.add_argument("--target", type=str, default=False)
        parser.add_argument("--use_weather", type=str2bool, default=True)
        parser.add_argument("--spatial_shuffle", type=str2bool, default=False)
        parser.add_argument(
            "--decoder_input_subtract_s2bands", type=str2bool, default=True
        )
        parser.add_argument("--weather_is_aggregated", type=str2bool, default=False)
        return parser

    def forward(
        self,
        data,
        pred_start: int = 0,
        preds_length: Optional[int] = None,
    ):
        """
        Forward pass of the ConvLSTMAE model.

        Parameters
        ----------
        data: dict
            Dictionary containing the input data.
        pred_start: int, optional
            Starting index of predictions (default is 0).
        preds_length: Optional[int], optional
            Length of predictions (default is None).
        step: Optional[int], optional
            Step parameter for teacher forcing (default is None).

        Returns
        -------
        torch.Tensor
            Output tensor containing the model predictions.
        dict
            Empty dictionary as the second return value (not used in this implementation).
        """

        # Determine the context length for prediction
        context_length = (
            self.hparams.context_length
            if self.training or (pred_start < self.hparams.context_length)
            else pred_start
        )

        step = data["global_step"]
        # Calculate teacher forcing coefficient
        if self.hparams.teacher_forcing:
            k = torch.tensor(0.1 * 10 * 12000)

            teacher_forcing_decay = k / (k + torch.exp(step + (120000 / 2) / k))
        else:
            teacher_forcing_decay = 0

        # Extract data components

        # sentinel 2 bands
        sentinel = data["dynamic"][0][:, :context_length, ...]

        # Extract the target for the teacher forcing method
        if self.hparams.teacher_forcing and self.training:
            target = data["dynamic"][0][
                :, context_length : context_length + self.hparams.target_length, ...
            ]

        weather = data["dynamic"][1].unsqueeze(3).unsqueeze(4)

        # Prepare landcover data
        # landcover = data["landcover"]
        # for key in list(class_mapping.keys()):
        #     landcover[landcover == key] = class_mapping.get(key)
        #
        # landcover = (
        #     nn.functional.one_hot(landcover.to(torch.int64), 11)
        #     .transpose(1, 4)
        #     .squeeze(4)
        # )
        #
        # # Concatenate static data with landcover
        # static = torch.cat((data["static"][0], data["static"][1]), dim=1)
        # static = torch.cat((static, landcover), dim=1)

        static = data["static"][0][:, :3, ...]

        # Get the dimensions of the input data. Shape: batch size, temporal size, number of channels, height, width
        b, t, c, h, w = sentinel.shape
        _, t_w, c_w, _, _ = weather.shape

        if self.hparams.spatial_shuffle:
            perm = torch.randperm(b * h * w, device=sentinel.device)
            invperm = inverse_permutation(perm)

            if weather.shape[-1] == 1:
                weather = weather.expand(-1, -1, -1, h, w)
            else:
                weather = nn.functional.interpolate(
                    weather, size=(h, w), mode="nearest-exact"
                )

            sentinel = (
                sentinel.permute(1, 2, 0, 3, 4)
                .reshape(t, c, b * h * w)[:, :, perm]
                .reshape(t, c, b, h, w)
                .permute(2, 0, 1, 3, 4)
            )
            weather = (
                weather.permute(1, 2, 0, 3, 4)
                .reshape(t_w, c_w, b * h * w)[:, :, perm]
                .reshape(t_w, c_w, b, h, w)
                .permute(2, 0, 1, 3, 4)
                .contiguous()
            )
            static = (
                static.permute(1, 0, 2, 3)
                .reshape(3, b * h * w)[:, perm]
                .reshape(3, b, h, w)
                .permute(1, 0, 2, 3)
            )

        # Initialize hidden states for encoder ConvLSTM cells
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, height=h, width=w)
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(
            batch_size=b, height=h, width=w
        )

        output = []

        # Encoding network
        for t in range(context_length):
            # Prepare input for encoder ConvLSTM cells
            input = torch.cat((sentinel[:, t, ...], static), dim=1)
            # WARNING only for En23

            if self.hparams.use_weather:
                if self.hparams.weather_is_aggregated:
                    weather_t = weather[:, t, ...].expand(-1, -1, h, w)
                else:
                    weather_t = (
                        weather[:, t : t + 5, ...]
                        .view(weather.shape[0], 1, -1, 1, 1)
                        .squeeze(1)
                        .repeat(1, 1, 128, 128)
                    )
                input = torch.cat((input, weather_t), dim=1)

            # First ConvLSTM block
            h_t, c_t = self.encoder_1_convlstm(input_tensor=input, cur_state=[h_t, c_t])
            # Second ConvLSTM block
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )

        # First prediction
        pred = self.conv(h_t2)

        # Add the last frame of the context period if skip_connections is True
        if self.hparams.skip_connections:
            pred = pred + sentinel[:, t, 0, ...].unsqueeze(1)

        pred = self.activation_output(pred)

        # Forecasting network
        for t in range(self.hparams.target_length):
            # Copy the previous prediction for skip connections
            pred_previous = torch.clone(pred)

            # Teacher forcing scheduled sampling
            if (
                self.hparams.teacher_forcing
                and self.training
                and torch.bernoulli(teacher_forcing_decay)
            ):
                pred = target[
                    :,
                    t,
                    : (
                        self.hparams.num_outputs
                        if not self.hparams.decoder_input_subtract_s2bands
                        else 1
                    ),
                    ...,
                ]

            pred = torch.cat((pred, static), dim=1)
            if self.hparams.use_weather:
                # Prepare input for decoder ConvLSTM cells
                if self.hparams.weather_is_aggregated:
                    weather_t = weather[:, context_length + t, ...].expand(-1, -1, h, w)
                else:
                    weather_t = (
                        weather[:, context_length + t : context_length + t + 5, ...]
                        .view(weather.shape[0], 1, -1, 1, 1)
                        .squeeze(1)
                        .repeat(1, 1, 128, 128)
                    )
                pred = torch.cat((pred, weather_t), dim=1)

            # First ConvLSTM block for the decoder
            h_t, c_t = self.decoder_1_convlstm(input_tensor=pred, cur_state=[h_t, c_t])

            # Second ConvLSTM block for the decoder
            h_t2, c_t2 = self.decoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )

            pred = h_t2
            pred = self.conv(pred)

            # Add the previous prediction for skip connections
            if self.hparams.skip_connections:
                pred = pred + pred_previous

            # Output
            pred = self.activation_output(pred)
            output += [pred.unsqueeze(1)]

        output = torch.cat(output, dim=1)  # .unsqueeze(2)

        if self.hparams.spatial_shuffle:
            output = (
                output.permute(1, 2, 0, 3, 4)
                .reshape(
                    self.hparams.target_length, self.hparams.num_outputs, b * h * w
                )[:, :, invperm]
                .reshape(self.hparams.target_length, self.hparams.num_outputs, b, h, w)
                .permute(2, 0, 1, 3, 4)
            )

            output = output[:, :, :1, :, :]

        return output, {}
