#!/bin/bash

gpu run python src/06_generate_figs_predictor.py \
    -d experiments/TextOCVP_CATER/ \
    --decomp_ckpt SAVi_CATER.pth \
    --name_pred_exp TextOCVP \
    --pred_ckpt TextOCVP_CATER.pth \
    --num_preds 19 \
    --num_seqs 10
