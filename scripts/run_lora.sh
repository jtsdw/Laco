#!/bin/bash
# Run LoRA baseline for comparison (requires separate LoRA training)
# Usage: bash scripts/run_lora.sh

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-2-7b-hf"}
SPARSITY=${SPARSITY:-0.7}
DEVICE=${DEVICE:-"cuda:0"}

echo "LoRA baseline: First prune with Wanda, then evaluate"
echo "For full LoRA comparison, train LoRA on the pruned model separately."

python -m laco.main ${MODEL_PATH} wikitext2 \
    --nsamples 2048 \
    --seqlen 128 \
    --is_prune \
    --is_wanda \
    --sparsity_ratio ${SPARSITY} \
    --device ${DEVICE} \
    --is_zs \
    --demo "lora-baseline-prune-only" \
    --log_path ./logs
