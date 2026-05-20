# LaCo: Layer-wise Compensation for Pruned Large Language Models

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of the paper: **LaCo: Layer-wise Compensation for Pruned Large Language Models**.

> **Abstract:** Pruning is essential for the efficient deployment of Large Language Models (LLMs); however, it causes severe performance degradation due to the structural distortion induced by sparsity. LaCo reorients the recovery paradigm from global adaptation to hierarchical representation alignment. By sequentially optimizing each layer to reconstruct the model's hidden states, LaCo effectively intercepts the error propagation chain at its source. LaCo reduces recovery-time memory usage to ~1/7 of baselines and achieves ~25x improvement in data efficiency.

## Key Features

- **Universal**: Works with unstructured (Wanda), semi-structured (SparseGPT), and structured (FLAP, LLM-Pruner) pruning
- **Memory-efficient**: ~7x lower peak memory vs. LoRA (6.2 GB vs 43.7 GB for 7B models)
- **Data-efficient**: Matches LoRA performance with only 2,048 unlabeled calibration samples (~25x fewer)
- **High fidelity**: Maintains CKA > 0.98 even at 70% sparsity

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/laco.git
cd laco

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA-capable GPU (for 7B models: 8GB+ VRAM with LaCo, 48GB+ with LoRA)

## Quick Start

### 1. Prune LLaMA-2-7B with Wanda at 70% sparsity and recover with LaCo

```bash
python -m laco.main meta-llama/Llama-2-7b-hf wikitext2 \
    --nsamples 2048 \
    --seqlen 128 \
    --is_prune \
    --is_wanda \
    --sparsity_ratio 0.7 \
    --is_train \
    --tune_epoch 10 \
    --tune_lr 5e-5 \
    --is_zs \
    --demo "wanda-0.7-laco" \
    --log_path ./logs
```

### 2. Using shell scripts

```bash
# Wanda unstructured pruning
MODEL_PATH=meta-llama/Llama-2-7b-hf SPARSITY=0.7 bash scripts/run_wanda.sh

# SparseGPT semi-structured pruning (4:8 pattern)
MODEL_PATH=meta-llama/Llama-2-7b-hf bash scripts/run_sparsegpt.sh

# FLAP structured pruning
MODEL_PATH=meta-llama/Llama-2-7b-hf SPARSITY=0.5 bash scripts/run_flap.sh

# LLM-Pruner structured pruning
MODEL_PATH=meta-llama/Llama-2-7b-hf SPARSITY=0.5 bash scripts/run_llmpruner.sh
```

## Supported Pruning Backbones

| Backbone | Type | Flag | Key Parameters |
|----------|------|------|----------------|
| **Wanda** | Unstructured | `--is_wanda` | `--sparsity_ratio` (0.2, 0.5, 0.7) |
| **SparseGPT** | Semi-structured | `--is_sparsegpt` | `--prune_n N --prune_m M` (2:4, 4:8) |
| **FLAP** | Structured | `--is_flap` | `--sparsity_ratio`, `--metrics WIFV` |
| **LLM-Pruner** | Structured | `--is_lmpruner` | `--sparsity_ratio`, `--pruner_type l2` |

## LaCo Algorithm

LaCo operates in two stages:

1. **Stage 1: Layer-wise Compensation** — Sequentially optimizes each pruned layer to minimize MSE between its output hidden states and the original dense model's hidden states, using a mask-constrained regression formulation.

2. **Stage 2: Knowledge Distillation** (optional) — End-to-end KL divergence distillation from the dense teacher to the compensated sparse student, using cached teacher logits.

```
Algorithm: Layer-wise Compensation
----------------------------------
Input: Dense params {theta_l}, Sparse params {theta'_l}, Calibration set X
For l = 1 to L:
    Phase 1: Compute dense target Y = phi_l(H_{l-1}; theta_l)
    Phase 2: Optimize theta'_l to minimize ||Y - phi_l(X; theta'_l)||²
             Apply mask constraint: theta'_l = theta'_l ⊙ M_l
    Phase 3: Propagate H'_l = phi_l(X; theta'_l), H_l = Y
Return {theta'_l}
```

## Evaluation

### Perplexity (PPL)
Evaluated automatically after training on WikiText2, PTB, and C4.

### Zero-shot Accuracy
Enable with `--is_zs`. Evaluates on 8 commonsense reasoning benchmarks:

| Task | Dataset Path Env Variable |
|------|--------------------------|
| BoolQ | `BOOLQ_PATH` |
| PIQA | `PIQA_DATA_PATH`, `PIQA_LABEL_PATH` |
| HellaSwag | `HELLASWAG_PATH` |
| WinoGrande | `WINOGRANDE_PATH` |
| OBQA | `OBQA_PATH` |
| ARC-Easy | `ARC_EASY_PATH` |
| ARC-Challenge | `ARC_CHALLENGE_PATH` |
| RTE | `RTE_PATH` |

Set dataset paths via environment variables or edit `laco/evaluation.py`.

## Project Structure

```
laco/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_wanda.sh
│   ├── run_sparsegpt.sh
│   ├── run_flap.sh
│   ├── run_llmpruner.sh
│   └── run_lora.sh
├── laco/
│   ├── main.py              # Main entry point
│   ├── compensation.py       # Core LaCo: stage1 + stage2
│   ├── pruning.py            # Pruning backends (Wanda, SparseGPT, FLAP)
│   ├── training.py           # Training loops, losses, constraints
│   ├── data.py               # Data loading (WikiText2, PTB, C4)
│   ├── evaluation.py         # PPL eval + zero-shot eval
│   ├── features.py           # Feature datasets
│   ├── quantization.py       # Quantization utils (for SparseGPT)
│   ├── model_utils.py        # Model loading utilities
│   └── utils.py              # General utilities
├── llama_pru/                # LLaMA model with pruning support
└── LLMPruner/                # LLM-Pruner framework
```

## Reproducing Paper Results

### Table 1 & 2: Main Results (LLaMA-2-7B)

```bash
# Wanda 70% sparsity (unstructured)
python -m laco.main meta-llama/Llama-2-7b-hf wikitext2 \
    --nsamples 2048 --seqlen 128 \
    --is_prune --is_wanda --sparsity_ratio 0.7 \
    --is_train --tune_epoch 10 --tune_lr 5e-5 \
    --enable_stage2 True --stage2_epochs 20 --stage2_lr 1e-5 \
    --is_zs

# SparseGPT 4:8 (semi-structured)
python -m laco.main meta-llama/Llama-2-7b-hf wikitext2 \
    --nsamples 2048 --seqlen 128 \
    --is_prune --is_sparsegpt --prune_n 4 --prune_m 8 \
    --is_train --tune_epoch 10 --tune_lr 5e-5 \
    --is_zs

# FLAP 50% sparsity (structured) - requires pre-computed FLAP masks
# First generate masks using the FLAP repository, then place them in ./flap_mask/
python -m laco.main meta-llama/Llama-2-7b-hf wikitext2 \
    --nsamples 2048 --seqlen 128 \
    --is_prune --is_flap --sparsity_ratio 0.5 \
    --is_bias \
    --is_train --tune_epoch 10 --tune_lr 5e-5 \
    --is_zs
```

### Cross-architecture (Qwen-2.5-7B, LLaMA-2-13B)

Qwen models are supported via `--model_type qwen`. LLaMA-2-13B uses the default LLaMA loader with the 13B model path.

```bash
# Qwen-2.5-7B with Wanda 70%
python -m laco.main Qwen/Qwen2.5-7B-Instruct wikitext2 \
    --model_type qwen --nsamples 2048 --seqlen 128 \
    --is_prune --is_wanda --sparsity_ratio 0.7 \
    --is_train --tune_epoch 10 --tune_lr 5e-5 --is_zs

# LLaMA-2-13B with Wanda 70%
python -m laco.main meta-llama/Llama-2-13b-hf wikitext2 \
    --nsamples 2048 --seqlen 128 \
    --is_prune --is_wanda --sparsity_ratio 0.7 \
    --is_train --tune_epoch 10 --tune_lr 5e-5 --is_zs
```

### Ablation: Module-wise Sensitivity (Table 4)

Use `--ablation_module` to control which submodules are compensated:

```bash
# Attention only
python -m laco.main MODEL_PATH wikitext2 --is_prune --is_wanda \
    --sparsity_ratio 0.7 --is_train --ablation_module attention

# MLP only
python -m laco.main MODEL_PATH wikitext2 --is_prune --is_wanda \
    --sparsity_ratio 0.7 --is_train --ablation_module ffn

# Full (default)
python -m laco.main MODEL_PATH wikitext2 --is_prune --is_wanda \
    --sparsity_ratio 0.7 --is_train --ablation_module full
```

### Layer-wise Analysis (Figure 2)

Use `--num_layers` to compensate only the first k layers:

```bash
# Compensate first 16 layers only
python -m laco.main MODEL_PATH wikitext2 --is_prune --is_wanda \
    --sparsity_ratio 0.7 --is_train --num_layers 16
```

## External Dependencies

Some experiments require external tools or manual setup:

| Feature | Dependency | Notes |
|---------|-----------|-------|
| **FLAP pruning** | [FLAP](https://github.com/CASIA-IVA-Lab/FLAP) | Pre-compute masks via FLAP repo, place in `./flap_mask/` |
| **LoRA baseline** | [PEFT](https://github.com/huggingface/peft) | Train LoRA separately on pruned model for comparison |
| **Zero-shot datasets** | Manual download | Set env vars (BOOLQ_PATH, PIQA_DATA_PATH, etc.) for each dataset |
| **Efficiency numbers** | External profiling | Memory/time were collected manually (nvidia-smi, time)

## Citation

```bibtex
@inproceedings{liu2025laco,
    title = {LaCo: Layer-wise Compensation for Pruned Large Language Models},
    author = {Liu, Yingen and Wu, Fan and Pan, Xuyan and Li, Ruihui and Tang, Zhuo and Li, Kenli},
    booktitle = {ACL},
    year = {2025}
}
```

## Acknowledgments

This project builds upon several open-source efforts:
- [Wanda](https://github.com/locuslab/wanda) — Pruning by Weights and Activations
- [SparseGPT](https://github.com/IST-DASLab/sparsegpt) — Massive Language Models Can Be Accurately Pruned in One-Shot
- [FLAP](https://github.com/CASIA-IVA-Lab/FLAP) — Fluctuation-based Adaptive Structured Pruning
- [LLM-Pruner](https://github.com/horseee/LLM-Pruner) — Structured Pruning for LLMs
- [LoRA](https://github.com/microsoft/LoRA) — Low-Rank Adaptation

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.
