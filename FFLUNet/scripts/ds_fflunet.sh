#!/bin/bash
#SBATCH --job-name=fflunetDs     
#SBATCH --nodes=1        
#SBATCH --ntasks=1               
#SBATCH --mem=64G              
#SBATCH --partition=gpupart_p100
#SBATCH --cpus-per-task=16      
#SBATCH --gpus-per-node=1           
#SBATCH --time=1-23:58:00        
#SBATCH --output=logs/%u_%x_%j.out

conda activate gpu
nvidia-smi
export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

## nnUNetv2_plan_and_preprocess -d 080 --verify_dataset_integrity
nnUNetv2_train 080 3d_fullres 1 -tr nnUNetTrainer_FFLUNetAttentionDynamicShiftDS --c