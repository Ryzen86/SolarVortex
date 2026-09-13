#!/bin/bash

export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

python nnunetv2/data_scripts/FFLUNetBenchmark.py