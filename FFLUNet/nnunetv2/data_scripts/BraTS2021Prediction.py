import argparse
import shutil
import time
import torch


from nnunetv2.dataset_conversion.Dataset080_BraTS20 import (
    convert_folder_with_preds_back_to_BraTS_labeling_convention,
)

from nnunetv2.configuration import default_num_processes
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    subdirs,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_raw, nnUNet_results


BRATS2020_TASK_ID = 90
BRATS2020_TASK_NAME = "BraTS2021"
CKPT = "checkpoint_best.pth"
DATA_ROOT = "data"
FOLDERNAME = "Dataset%03.0d_%s" % (BRATS2020_TASK_ID, BRATS2020_TASK_NAME)


def generate_brats2020_predictions(
    ip_folder: str,
    tr_root: str,
    device: str,
):
    device = torch.device(device)
    test_images_root = join(
        nnUNet_raw,
        FOLDERNAME,
        "imagesTs",
    )

    output_dir = join(
        nnUNet_results,
        FOLDERNAME,
        "predictions",
        tr_root,
    )
    print("Got ip_folder", ip_folder)
    if ip_folder:
        copy_to_test_root_dir(ip_folder)

    # Run Prediction job
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, tr_root),
        use_folds=None,
        checkpoint_name=CKPT,
    )

    predictor.predict_from_files(
        test_images_root,
        output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    final_output_folder = join(
        DATA_ROOT,
        "predictions",
        FOLDERNAME,
        tr_root,
        str(int(time.time())),
    )
    convert_folder_with_preds_back_to_BraTS_labeling_convention(
        output_dir,
        final_output_folder,
        num_processes=default_num_processes,
    )
    print("Finished")


def copy_to_test_root_dir(ip_folder):
    brats_data_dir = ip_folder

    # Set up nnunet folders for storing test data
    test_images_root = join(nnUNet_raw, FOLDERNAME, "imagesTs")
    maybe_mkdir_p(test_images_root)

    # Copy files
    case_ids = subdirs(brats_data_dir, prefix="BraTS", join=False)
    print("Copy Started")
    for c in case_ids:
        shutil.copy(
            join(brats_data_dir, c, c + "_t1.nii.gz"),
            join(test_images_root, c + "_0000.nii.gz"),
        )
        shutil.copy(
            join(brats_data_dir, c, c + "_t1ce.nii.gz"),
            join(test_images_root, c + "_0001.nii.gz"),
        )
        shutil.copy(
            join(brats_data_dir, c, c + "_t2.nii.gz"),
            join(test_images_root, c + "_0002.nii.gz"),
        )
        shutil.copy(
            join(brats_data_dir, c, c + "_flair.nii.gz"),
            join(test_images_root, c + "_0003.nii.gz"),
        )
    print("Copy Complete")

    return test_images_root


def entry_point():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        type=str,
        required=False,
        default=None,
        help="Root input folder containing validation data for Brats2020",
    )

    parser.add_argument(
        "-D",
        type=str,
        required=False,
        default="cuda:0",
        help="Device String. Default cuda:0",
    )

    parser.add_argument(
        "-tr",
        type=str,
        required=True,
        help="Trainer Path from NNUNetv2 results folder",
    )

    args = parser.parse_args()
    generate_brats2020_predictions(args.i, args.tr, args.D)
