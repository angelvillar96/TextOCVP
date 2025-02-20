#!/bin/bash

gpu run python src/03_evaluate_decomp_model.py \
    -d experiments/TextOCVP_CATER/ \
    --decomp_ckpt SAVi_CATER.pth \
    --results_name results_DecompModel \
    --batch_size 64
