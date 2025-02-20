#!/bin/bash

gpu run python src/06_generate_figs_decomp_model.py \
    -d experiments/TextOCVP_CLIPort/ \
    --decomp_ckpt ExtendedDINOSAUR_CLIPort.pth \
    --num_seqs 10