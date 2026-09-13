# FFLUNet Step-by-Step Training Script for Colab

# 1. Install dependencies and clone FFLUNet
!pip install -q kagglehub nnunetv2 SimpleITK nibabel
!git clone https://github.com/Dutta-SD/FFLUNet.git
%cd FFLUNet
!pip install -e .

# # 1. Clear any prior broken clones and update numpy to satisfy numba requirements
# !rm -rf FFLUNet
# !pip install -q "numpy<2.1,>=1.22"

# # 2. Install primary prerequisite libraries
# !pip install -q kagglehub nnunetv2 SimpleITK nibabel

# # 3. Clone the repo and properly navigate inside it before installing
# !git clone https://github.com/Dutta-SD/FFLUNet.git
# %cd FFLUNet

# # 4. Install FFLUNet in editable mode
# !pip install -e .

# 2. Download BraTS Dataset
import kagglehub
path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
print("Dataset downloaded to:", path)

# 3. Setup nnUNet workspace
import os
workspace = "/content/nnunet_workspace"
os.environ["nnUNet_raw"] = os.path.join(workspace, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(workspace, "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(workspace, "nnUNet_results")

for folder in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
    os.makedirs(os.path.join(workspace, folder), exist_ok=True)

print("nnUNet workspace directories created.")

# 4. Process Dataset (Label Remap 4->3)
import glob
import shutil
import nibabel as nib
import numpy as np
import json
from tqdm import tqdm

brats_path = os.path.join(path, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
patients = sorted(glob.glob(os.path.join(brats_path, "BraTS20_*")))
patients = [p for p in patients if os.path.exists(os.path.join(p, f"{os.path.basename(p)}_seg.nii"))]
print(f"Total valid patients found: {len(patients)}")

dataset_name = "Dataset001_BraTS"
base = os.path.join(os.environ["nnUNet_raw"], dataset_name)
imagesTr = os.path.join(base, "imagesTr")
labelsTr = os.path.join(base, "labelsTr")

shutil.rmtree(base, ignore_errors=True)
os.makedirs(imagesTr, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)

valid_cases = 0
for p in tqdm(patients):
    patient_id = os.path.basename(p)
    flair = os.path.join(p, f"{patient_id}_flair.nii")
    t1 = os.path.join(p, f"{patient_id}_t1.nii")
    t1ce = os.path.join(p, f"{patient_id}_t1ce.nii")
    t2 = os.path.join(p, f"{patient_id}_t2.nii")
    seg = os.path.join(p, f"{patient_id}_seg.nii")

    files = [flair, t1, t1ce, t2, seg]
    if not all(os.path.exists(f) for f in files):
        continue

    case_id = f"BraTS_{valid_cases:03d}"
    shutil.copy(flair, os.path.join(imagesTr, f"{case_id}_0000.nii"))
    shutil.copy(t1, os.path.join(imagesTr, f"{case_id}_0001.nii"))
    shutil.copy(t1ce, os.path.join(imagesTr, f"{case_id}_0002.nii"))
    shutil.copy(t2, os.path.join(imagesTr, f"{case_id}_0003.nii"))

    img = nib.load(seg)
    mask = img.get_fdata()
    mask = np.where(mask == 4, 3, mask).astype(np.uint8)
    corrected = nib.Nifti1Image(mask, img.affine, img.header)
    nib.save(corrected, os.path.join(labelsTr, f"{case_id}.nii"))
    valid_cases += 1

dataset_json = {
    "name": "BraTS",
    "channel_names": {"0": "FLAIR", "1": "T1", "2": "T1ce", "3": "T2"},
    "labels": {"background": 0, "necrotic": 1, "edema": 2, "enhancing": 3},
    "numTraining": valid_cases,
    "file_ending": ".nii",
    "overwrite_image_reader_writer": "NibabelIO"
}
with open(os.path.join(base, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)
print("dataset.json created with NibabelIO overwrite.")

# 5. Fix PyTorch Version Issue in PolyLR
from pathlib import Path
polylr_path = Path("/content/FFLUNet/nnunetv2/training/lr_scheduler/polylr.py")
if polylr_path.exists():
    text = polylr_path.read_text()
    old = """super().__init__(
            optimizer, current_step if current_step is not None else -1, False
        )"""
    new = """super().__init__(
            optimizer,
            last_epoch=current_step if current_step is not None else -1,
        )"""
    text = text.replace(old, new)
    polylr_path.write_text(text)
    print("Patched polylr.py successfully!")

# 6. Plan and Preprocess
!nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# 7. Train Model (Standard FFLUNet)
!nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer_FFLUNet
