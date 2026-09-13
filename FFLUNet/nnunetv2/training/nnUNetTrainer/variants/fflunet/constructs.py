import math

from monai.networks.blocks import Convolution
from torch import nn

NORM = "BATCH"
DEFAULT_ACT = "PRELU"
DROPOUT = 0.1
NO_ACTIVATION = "NO_ACTIVATION"


def get_conv(
    in_channels,
    out_channels,
    kernel=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=None,
    output_padding=None,
    is_transposed=False,
    act=None,
    bias=False,
):
    if act is None:
        act = DEFAULT_ACT
    elif act == NO_ACTIVATION:
        act = None

    if groups is not None:
        return Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            strides=stride,
            padding=padding,
            output_padding=output_padding,
            is_transposed=is_transposed,
            dropout=DROPOUT,
            dilation=dilation,
            groups=groups,
            act=(
                (DEFAULT_ACT, {"num_parameters": out_channels})
                if act == DEFAULT_ACT
                else act
            ),
            bias=bias,
            norm=NORM,
        )
    elif groups is None and math.gcd(in_channels, out_channels) == 1:
        return Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            strides=stride,
            padding=padding,
            output_padding=output_padding,
            is_transposed=is_transposed,
            dropout=DROPOUT,
            dilation=dilation,
            groups=1,
            act=(
                (DEFAULT_ACT, {"num_parameters": out_channels})
                if act == DEFAULT_ACT
                else act
            ),
            bias=bias,
            norm=NORM,
        )
    else:
        return nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                strides=stride,
                padding=padding,
                output_padding=output_padding,
                is_transposed=is_transposed,
                dropout=DROPOUT,
                dilation=dilation,
                groups=math.gcd(in_channels, out_channels),
                act=(
                    (DEFAULT_ACT, {"num_parameters": out_channels})
                    if act == DEFAULT_ACT
                    else act
                ),
                bias=bias,
                norm=NORM,
            ),
            Convolution(
                spatial_dims=3,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=DROPOUT,
                act=(
                    (DEFAULT_ACT, {"num_parameters": out_channels})
                    if act == DEFAULT_ACT
                    else act
                ),
                bias=bias,
                norm=NORM,
            ),
        )
