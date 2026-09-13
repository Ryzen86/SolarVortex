import torch
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNet import (
    FFLUNet,
)
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShiftMoreParams import FFLUNet_MoreParams

model = FFLUNet_MoreParams(4, 3)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model):
    # Might not work
    for name, module in model.named_children():
        if not name.startswith("params"):
            new_name = name.upper()
            z = name[1:]
            if name.startswith("m"):
                new_name = f"MVFF[{z}] (in_channels={module.in_channels}, out_channels={module.out_channels})"
            elif name.startswith("d"):
                new_name = f"D2[{z}] (in_channels={module.in_channels}, out_channels={module.out_channels})"
            elif name.startswith("u"):
                new_name = f"FFU[{z}] (in_channels={module.in_channels}, out_channels={module.out_channels})"
            elif name.startswith("s"):
                new_name = f"DSC[{z}] (in_channels={module.in_channels}, out_channels={module.out_channels})"
            elif name == "input":
                new_name = f"INPUT (in_channels={module.in_channels}, out_channels={module.out_channels})"
            elif name == "output":
                new_name = f"OUTPUT (in_channels={module.in_channels}, out_channels={module.out_channels})"
            print(new_name, count_parameters(module))


try:
    count_parameters_per_layer(model)
finally:
    print(count_parameters(model))
    with torch.no_grad():
        print(model(torch.randn(1, 4, 64, 64, 64)).shape)
