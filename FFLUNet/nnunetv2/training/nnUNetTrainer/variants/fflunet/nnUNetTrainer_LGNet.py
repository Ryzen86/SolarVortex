import math
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
import torch
from torch import nn
from monai.networks.blocks import Convolution
import torch.nn.functional as F


# Constants
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
        act = "PRELU"
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
            act=("PRELU", {"num_parameters": out_channels}) if act == "PRELU" else act,
            bias=bias,
            norm="BATCH",
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
            act=("PRELU", {"num_parameters": out_channels}) if act == "PRELU" else act,
            bias=bias,
            norm="BATCH",
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
                    ("PRELU", {"num_parameters": out_channels})
                    if act == "PRELU"
                    else act
                ),
                bias=bias,
                norm="BATCH",
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
                    ("PRELU", {"num_parameters": out_channels})
                    if act == "PRELU"
                    else act
                ),
                bias=bias,
                norm="BATCH",
            ),
        )


class MultiViewFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local_fe = get_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            stride=1,
            padding=1,
        )
        self.global_fe = get_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            dilation=3,
            padding=None,
            stride=1,
        )
        self.roll = (4, 4, 4)
        self.unroll = (-4, -4, -4)
        self.wts = nn.Parameter(torch.rand(2), requires_grad=True)

    def forward(self, x):
        z_g = torch.roll(x, shifts=self.roll, dims=(2, 3, 4))
        z_g = self.global_fe(z_g)
        z_g = torch.roll(z_g, shifts=self.unroll, dims=(2, 3, 4))
        z_l = self.local_fe(x)
        w = F.softmax(self.wts, dim=0)
        return w[0] * z_l + w[1] * z_g


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d_layer = get_conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        self.p_layer = get_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=1,
            stride=1,
            padding=0,
            groups=1,
        )

    def forward(self, x):
        return self.p_layer(self.d_layer(x))


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = get_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            stride=1,
            padding=None,
            dilation=2,
        )

        self.conv2 = get_conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=2,
            stride=2,
            padding=0,
            dilation=1,
            is_transposed=True,
            output_padding=0,
            groups=1,
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Input(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.model = nn.Sequential(
            get_conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel=3,
                stride=1,
                padding=1,
                groups=1,
            ),
            get_conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=2,
                stride=2,
                padding=0,
                groups=1,
            ),
        )

    def forward(self, ip_tensor):
        return self.model(ip_tensor)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_path = nn.Sequential(
            get_conv(in_channels=in_channels, out_channels=in_channels),
            get_conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=2,
                stride=2,
                padding=0,
            ),
        )

    def forward(self, x):
        return self.conv_path(x)


class Output(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_path = nn.Sequential(
            get_conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=2,
                stride=2,
                padding=0,
                output_padding=0,
                dilation=1,
                is_transposed=True,
            ),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        return self.conv_path(x)


class LocalGlobalNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input
        self.input = Input(in_channels, 32)

        # MSFF and down layer
        self.m1 = MultiViewFeatureFusion(32, 32)
        self.d1 = Downsample(32, 64)

        self.m2 = MultiViewFeatureFusion(64, 64)
        self.d2 = Downsample(64, 128)

        self.m3 = MultiViewFeatureFusion(128, 128)
        self.d3 = Downsample(128, 256)

        self.m4 = MultiViewFeatureFusion(256, 256)
        self.d4 = Downsample(256, 320)

        # Up layers and ds layers
        self.u4 = Upsample(320, 256)
        self.s4 = DepthwiseSeparableConvolution(256, 256)

        self.u3 = Upsample(256, 128)
        self.s3 = DepthwiseSeparableConvolution(128, 128)

        self.u2 = Upsample(128, 64)
        self.s2 = DepthwiseSeparableConvolution(64, 64)

        self.u1 = Upsample(64, 32)
        self.s1 = DepthwiseSeparableConvolution(32, 32)

        self.w4 = nn.Parameter(torch.rand(2), requires_grad=True)
        self.w3 = nn.Parameter(torch.rand(2), requires_grad=True)
        self.w2 = nn.Parameter(torch.rand(2), requires_grad=True)
        self.w1 = nn.Parameter(torch.rand(2), requires_grad=True)

        # Output
        self.output = Output(32, out_channels)

    def forward(self, x):
        x = self.input(x)

        x = x1 = self.m1(x)
        x = self.d1(x)

        x = x2 = self.m2(x)
        x = self.d2(x)

        x = x3 = self.m3(x)
        x = self.d3(x)

        x = x4 = self.m4(x)
        x = self.d4(x)

        w4 = F.softmax(self.w4, dim=0)
        w3 = F.softmax(self.w3, dim=0)
        w2 = F.softmax(self.w2, dim=0)
        w1 = F.softmax(self.w1, dim=0)

        x = w4[0] * self.u4(x) + w4[1] * self.s4(x4)
        x = w3[0] * self.u3(x) + w3[1] * self.s3(x3)
        x = w2[0] * self.u2(x) + w2[1] * self.s2(x2)
        x = w1[0] * self.u1(x) + w1[1] * self.s1(x1)

        x = self.output(x)
        return x


class nnUNetTrainer_LGNet(nnUNetTrainerNoDeepSupervision):

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_op_channels = label_manager.num_segmentation_heads
        return LocalGlobalNet(
            in_channels=num_input_channels,
            out_channels=num_op_channels,
        )
