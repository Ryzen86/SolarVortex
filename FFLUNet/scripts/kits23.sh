#!/bin/bash
#SBATCH --job-name=kits23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --partition=gpupart_p100
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:58:00
#SBATCH --output=logs/%u_%x_%j.out

conda activate gpu
nvidia-smi
export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

python nnunetv2/dataset_conversion/Dataset220_KiTS2023.py -i data/dataset
# nnUNetv2_plan_and_preprocess -d 220 --verify_dataset_integrity
# nnUNetv2_train 220 3d_fullres 1 -tr nnUNetTrainer_FFLUNetAttentionDynamicShift --c