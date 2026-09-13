import torch
import torch.nn as nn
import torch.nn.functional as fx

from nnunetv2.training.nnUNetTrainer.variants.fflunet.constructs import get_conv


class DynamicMultiViewFeatureFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.local_fe = get_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            stride=1,
            padding=1,
            groups=1,
        )
        self.global_fe = get_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            dilation=3,
            padding=None,
            stride=1,
            groups=1,
        )
        self.wts = nn.Parameter(torch.rand(2), requires_grad=True)

    def forward(self, x, w: int):
        z_global = self.get_global_repr(x, w)
        z_local = self.get_local_repr(x)
        return self.feature_fusion(z_global, z_local)

    def feature_fusion(self, z_global, z_local):
        w = fx.softmax(self.wts, dim=0)
        return w[0] * z_local + w[1] * z_global

    def get_local_repr(self, x):
        z_l = self.local_fe(x)
        return z_l

    def get_global_repr(self, x, w: int | tuple | list):
        if isinstance(w, int):
            roll, unroll = (w, w, w), (-w, -w, -w)
        else:
            roll, unroll = w, (-w[0], -w[1], -w[2])

        z_global = torch.roll(x, shifts=roll, dims=(2, 3, 4))
        z_global = self.global_fe(z_global)
        z_global = torch.roll(z_global, shifts=unroll, dims=(2, 3, 4))
        return z_global


class DepthWiseSeparableConvolution(nn.Module):
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


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = get_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            stride=1,
            padding=None,
            dilation=2,
            groups=1,
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


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_path = nn.Sequential(
            get_conv(in_channels=in_channels, out_channels=in_channels, groups=1),
            get_conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=2,
                stride=2,
                padding=0,
                groups=1,
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
                groups=1,
            ),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
            ),
        )

    def forward(self, x):
        return self.conv_path(x)


class OutputV2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_path = nn.Sequential(
            get_conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel=2,
                stride=2,
                padding=0,
                output_padding=0,
                dilation=1,
                is_transposed=True,
                groups=1,
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
            ),
        )

    def forward(self, x):
        return self.conv_path(x)
