#!/bin/bash
# Run LaCo with SparseGPT semi-structured pruning
# Usage: bash scripts/run_sparsegpt.sh

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-2-7b-hf"}
PATTERN_N=${PATTERN_N:-4}
PATTERN_M=${PATTERN_M:-8}
NSAMPLES=${NSAMPLES:-2048}
DEVICE=${DEVICE:-"cuda:0"}

python -m laco.main ${MODEL_PATH} wikitext2 \
    --nsamples ${NSAMPLES} \
    --seqlen 128 \
    --is_prune \
    --is_sparsegpt \
    --sparsity_ratio 0.5 \
    --prune_n ${PATTERN_N} \
    --prune_m ${PATTERN_M} \
    --device ${DEVICE} \
    --is_train \
    --tune_epoch 10 \
    --tune_lr 5e-5 \
    --is_zs \
    --demo "sparsegpt-${PATTERN_N}-${PATTERN_M}" \
    --log_path ./logs
