from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetAttentionDynamicShift import (
    FFLUNetAttentionDynamicShift,
)
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift4Layers import (
    FFLUNetDynamicWindowShift4Layers,
)
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift import (
    FFLUNetDynamicWindowShift,
)
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift12M import (
    FFLUNetDynamicWindowShift12M,
)
import json
from calflops import calculate_flops
import torch
import time
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import numpy as np
import torch
from monai.inferers import sliding_window_inference
import gc

NNUNET_CONFIG_NAME = "3d_fullres"
INPUT_SHAPE = (1, 4, 128, 128, 128)
I_CH = 4
O_CH = 3
ROI = (64, 64, 64)
SW_BATCH_SIZE = 4
OVERLAP = 0.5

with open("data/nnUNet_preprocessed/Dataset980_BraTS2023/nnUNetPlans.json") as fp:
    plans = json.load(fp)
    pm = PlansManager(plans)

with open("data/nnUNet_preprocessed/Dataset980_BraTS2023/dataset.json") as fp:
    djson = json.load(fp)


def get_avg_std(val_list):
    average_time = np.mean(val_list)
    std_deviation = np.std(val_list)
    return average_time, std_deviation


def clear_gpu_memory():
    """Helper function to clear GPU memory between runs"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


@torch.inference_mode()
def benchmark(model, num_simulation_runs):
    print(f"Benchmarking {model.__class__.__name__}...")

    set_all_seed()

    # Flops, MACs
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=INPUT_SHAPE,
        print_detailed=False,
        print_results=False,
        output_as_string=True,
        output_precision=4,
    )
    print(
        f"[{model.__class__.__name__}]: (FLOPs:{flops}, MACs:{macs}, #Params:{params})"
    )

    benchmark_cpu(model, num_simulation_runs)
    # GPU benchmarking
    benchmark_gpu(model, num_simulation_runs)
    print("-" * 50)


def benchmark_cpu(model, num_simulation_runs):
    # Inference time
    cpu_times = []

    # Ensure model is in eval mode
    model.eval()

    # CPU benchmarking
    model = model.cpu()

    # Warmup run
    image = torch.rand(1, 4, 240, 240, 155).cpu()
    sliding_window_inference(
        image,
        roi_size=ROI,
        sw_batch_size=SW_BATCH_SIZE,
        predictor=model,
        overlap=OVERLAP,
    )

    for i in range(num_simulation_runs):
        print(f"CPU run {i+1}/{num_simulation_runs}")
        image = torch.rand(1, 4, 240, 240, 155).cpu()

        s = time.time()
        sliding_window_inference(
            image,
            roi_size=ROI,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
        )
        e = time.time()
        cpu_times.append(e - s)

    print(
        "{} CPU Inference: {:.3f} ± {:.3f} s".format(
            model.__class__.__name__, *get_avg_std(cpu_times)
        )
    )


def benchmark_gpu(model, num_simulation_runs):
    model.eval()
    set_all_seed()
    gpu_times = []

    clear_gpu_memory()

    model = model.cuda()

    # Warmup run
    image = torch.rand(1, 4, 240, 240, 155).cuda()
    sliding_window_inference(
        image,
        roi_size=ROI,
        sw_batch_size=SW_BATCH_SIZE,
        predictor=model,
        overlap=OVERLAP,
    )
    clear_gpu_memory()

    for i in range(num_simulation_runs):
        print(f"GPU run {i+1}/{num_simulation_runs}")
        image = torch.rand(1, 4, 240, 240, 155).cuda()

        torch.cuda.synchronize()
        s = time.time()
        sliding_window_inference(
            image,
            roi_size=ROI,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
        )
        torch.cuda.synchronize()
        e = time.time()
        gpu_times.append(e - s)
        clear_gpu_memory()

    print(
        "{} GPU Inference: {:.3f} ± {:.3f} s".format(
            model.__class__.__name__, *get_avg_std(gpu_times)
        )
    )


def set_all_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)


def get_model(model_name):
    model = None
    if model_name == "NNUNET":
        trainer = nnUNetTrainer(
            plans=plans,
            configuration=NNUNET_CONFIG_NAME,
            fold=1,
            dataset_json=djson,
        )
        model = trainer.build_network_architecture(
            plans_manager=pm,
            dataset_json=djson,
            configuration_manager=pm.get_configuration(NNUNET_CONFIG_NAME),
            num_input_channels=4,
            enable_deep_supervision=True,
        )
    elif model_name == "FFLUNET12M":
        model = FFLUNetDynamicWindowShift12M(4, 3)
    elif model_name == "FFLUNET":
        model = FFLUNetDynamicWindowShift(4, 3)
    elif model_name == "FFLUNET4LAYERS":
        model = FFLUNetDynamicWindowShift4Layers(4, 3)
    elif model_name == "FFLUNETATTENTION":
        model = FFLUNetAttentionDynamicShift(4, 3)
    return model


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    all_model_names = [
        "NNUNET",
        "FFLUNET",
        "FFLUNET4LAYERS",
        "FFLUNETATTENTION",
        "FFLUNET12M",
    ]

    for model_name in all_model_names:
        clear_gpu_memory()
        model = get_model(model_name)
        benchmark(model, 20)
