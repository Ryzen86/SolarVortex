#!/bin/bash

# Check if the dataset ID argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_id>"
  exit 1
fi

# Set the dataset ID from the first argument
DATASET_ID=$1

# Set environment variables
export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

# Run nnUNet commands with the specified dataset ID
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c 3d_fullres --clean
nnUNetv2_train $DATASET_ID 3d_fullres 0 -tr nnUNetTrainer_VNet
nnUNetv2_train $DATASET_ID 3d_fullres 0 -tr nnUNetTrainer_AttentionUNet
