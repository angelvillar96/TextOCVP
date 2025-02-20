#!/bin/bash

gpu run python src/06_generate_figs_predictor.py \
    -d experiments/TextOCVP_CLIPort/ \
    --decomp_ckpt ExtendedDINOSAUR_CLIPort.pth \
    --name_pred_exp TextOCVP \
    --pred_ckpt TextOCVP_CLIPort.pth \
    --num_preds 19 \
    --num_seqs 10
