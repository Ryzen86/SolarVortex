import shutil
import glob
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def convert_kits2023(kits_base_dir: str, nnunet_dataset_id: int = 220):
    task_name = "KiTS2023"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # cases = subdirs(kits_base_dir, prefix="case_", join=False)
    cases = glob.glob(kits_base_dir+"/*.nii.gz")
    for tr in cases:
        img_name = tr.split("/")[len(tr.split("/"))-1].split(".")[0]
        shutil.copy(
            join(tr),
            join(imagestr, f"{img_name}_0000.nii.gz"),
        )
        # shutil.copy(
        #     join(kits_base_dir, tr, "segmentation.nii.gz"),
        #     join(labelstr, f"{tr}.nii.gz"),
        # )
        print("Done with case", tr)

    generate_dataset_json(
        out_base,
        {0: "CT"},
        labels={"background": 0, "kidney": (1, 2, 3), "masses": (2, 3), "tumor": 2},
        regions_class_order=(1, 3, 2),
        num_training_cases=len(cases),
        file_ending=".nii.gz",
        dataset_name=task_name,
        reference="none",
        release="prerelease",
        overwrite_image_reader_writer="NibabelIOWithReorient",
        description="KiTS2023",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        type=str,
        help="The downloaded and extracted KiTS2023 dataset (must have case_XXXXX subfolders)",
    )
    parser.add_argument(
        "-d",
        required=False,
        type=int,
        default=220,
        help="nnU-Net Dataset ID, default: 220",
    )
    args = parser.parse_args()
    amos_base = args.i
    print("Chack path>> ", amos_base, args.d )
    convert_kits2023(amos_base, args.d)

    # /media/isensee/raw_data/raw_datasets/kits23/dataset
