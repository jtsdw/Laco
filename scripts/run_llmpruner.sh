#!/bin/bash
# Run LaCo with LLM-Pruner structured pruning
# Usage: bash scripts/run_llmpruner.sh

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-2-7b-hf"}
SPARSITY=${SPARSITY:-0.5}
NSAMPLES=${NSAMPLES:-2048}
DEVICE=${DEVICE:-"cuda:0"}

python -m laco.main ${MODEL_PATH} wikitext2 \
    --nsamples ${NSAMPLES} \
    --seqlen 128 \
    --is_prune \
    --is_lmpruner \
    --sparsity_ratio ${SPARSITY} \
    --device ${DEVICE} \
    --is_train \
    --tune_epoch 10 \
    --tune_lr 5e-5 \
    --is_zs \
    --demo "llmpruner-${SPARSITY}" \
    --log_path ./logs
