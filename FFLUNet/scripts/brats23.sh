#!/bin/bash
#SBATCH --job-name=brats23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=gpupart_24hour
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=23:58:00
#SBATCH --output=logs/%u_%x_%j.out

# conda activate gpu
nvidia-smi
export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

python nnunetv2/dataset_conversion/Dataset980_BraTS23.py
# nnUNetv2_plan_and_preprocess -d 980 --verify_dataset_integrity
# nnUNetv2_train 220 3d_fullres 1 -tr nnUNetTrainer_FFLUNetAttentionDynamicShift --c