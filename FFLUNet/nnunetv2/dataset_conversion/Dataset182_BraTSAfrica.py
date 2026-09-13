import shutil

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


BRATS_ROOT_DIR = (
    "/home/surajit/sandip/BraTS-Africa/95_Glioma"
)

BRATS23_TASK_ID = 182
BRATS23_TASK_NAME = "BraTSAfrica"

foldername = f"Dataset{BRATS23_TASK_ID:03.0f}_{BRATS23_TASK_NAME}"


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
    in_file: str,
    out_file: str,
) -> None:

    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4, 3]:
            raise RuntimeError("unexpected label")

    seg_new = np.zeros_like(img_npy)
    seg_new[((img_npy == 4) | (img_npy == 3))] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def pathnorm(p: str):
    return "_".join(p.split("-"))


if __name__ == "__main__":

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    all_brats_dirs = subdirs(
        BRATS_ROOT_DIR,
        prefix="BraTS",
        join=False,
        sort=True,
    )
    print(f"Found {len(all_brats_dirs)} cases")

    for case_id in all_brats_dirs:
        print("Working on", case_id)
        shutil.copy(
            join(BRATS_ROOT_DIR, case_id, case_id + "-t1n.nii.gz"),
            join(imagestr, pathnorm(case_id + "_0000.nii.gz")),
        )
        shutil.copy(
            join(BRATS_ROOT_DIR, case_id, case_id + "-t1c.nii.gz"),
            join(imagestr, pathnorm(case_id + "_0001.nii.gz")),
        )
        shutil.copy(
            join(BRATS_ROOT_DIR, case_id, case_id + "-t2w.nii.gz"),
            join(imagestr, pathnorm(case_id + "_0002.nii.gz")),
        )
        shutil.copy(
            join(BRATS_ROOT_DIR, case_id, case_id + "-t2f.nii.gz"),
            join(imagestr, pathnorm(case_id + "_0003.nii.gz")),
        )

        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
            join(BRATS_ROOT_DIR, case_id, case_id + "-seg.nii.gz"),
            join(labelstr, pathnorm(case_id + ".nii.gz")),
        )

    generate_dataset_json(
        out_base,
        channel_names={0: "T1", 1: "T1ce", 2: "T2", 3: "Flair"},
        labels={
            "background": 0,
            "whole tumor": (1, 2, 3),
            "tumor core": (2, 3),
            "enhancing tumor": (3,),
        },
        num_training_cases=len(all_brats_dirs),
        file_ending=".nii.gz",
        regions_class_order=(1, 2, 3),
    )
