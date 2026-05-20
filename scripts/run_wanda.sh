#!/bin/bash
# Run LaCo with Wanda unstructured pruning
# Usage: bash scripts/run_wanda.sh

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-2-7b-hf"}
SPARSITY=${SPARSITY:-0.7}
NSAMPLES=${NSAMPLES:-2048}
DEVICE=${DEVICE:-"cuda:0"}

python -m laco.main ${MODEL_PATH} wikitext2 \
    --nsamples ${NSAMPLES} \
    --seqlen 128 \
    --is_prune \
    --is_wanda \
    --sparsity_ratio ${SPARSITY} \
    --sparsity_type unstructured \
    --device ${DEVICE} \
    --is_train \
    --tune_epoch 10 \
    --tune_lr 5e-5 \
    --is_zs \
    --demo "wanda-${SPARSITY}" \
    --log_path ./logs
