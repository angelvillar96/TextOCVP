#!/bin/bash

gpu run python src/06_generate_figs_decomp_model.py \
    -d experiments/TextOCVP_CATER/ \
    --decomp_ckpt SAVi_CATER.pth \
    --num_seqs 10