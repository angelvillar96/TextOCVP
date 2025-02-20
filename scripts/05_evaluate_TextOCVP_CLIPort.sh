#!/bin/bash

gpu run python src/05_evaluate_predictor.py \
    -d experiments/TextOCVP_CLIPort/ \
    --decomp_ckpt ExtendedDINOSAUR_CLIPort.pth \
    --name_pred_exp TextOCVP \
    --pred_ckpt TextOCVP_CLIPort.pth \
    --results_name results_TextOCVP_NumSeed=1_NumPreds=9 \
    --num_seed 1 \
    --num_preds 9 \
    --batch_size 16
