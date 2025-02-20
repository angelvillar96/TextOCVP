#!/bin/bash

gpu run python src/03_evaluate_decomp_model.py \
    -d experiments/TextOCVP_CLIPort/ \
    --decomp_ckpt ExtendedDINOSAUR_CLIPort.pth \
    --results_name results_DecompModel \
    --batch_size 16
