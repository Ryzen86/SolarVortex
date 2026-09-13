import torch
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from torch import nn
from monai.networks.nets import VNet
from monai.networks.nets import AttentionUnet


class nnUNetTrainer_VNet(nnUNetTrainerNoDeepSupervision):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 50


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
        return VNet(
            spatial_dimensions=3,
            in_channele=num_input_channels,
            out_channels=num_op_channels,
        )


class nnUNetTrainer_AttentionUNet(nnUNetTrainerNoDeepSupervision):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 50

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

        return AttentionUnet(
            spatial_dims=3,  # 2D U-Net; set to 3 for 3D volumetric data.
            in_channels=num_input_channels,  # Number of input channels (e.g., 1 for grayscale medical images).
            out_channels=num_op_channels,  # Number of output classes; adjust based on your application.
            channels=(
                16,
                32,
                64,
                128,
                256,
            ),  # Number of channels in each level of the U-Net.
            strides=(
                2,
                2,
                2,
                2,
            ),  # Stride for each level, typically (2, 2, 2, 2) for downsampling.
            dropout=0.1,  # Dropout probability; adjust to add regularization.
        )
