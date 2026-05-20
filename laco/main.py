"""
LaCo: Layer-wise Compensation for Pruned Large Language Models.

Main entry point supporting:
  - Pruning backbones: Wanda (unstructured), SparseGPT (semi-structured),
    FLAP (structured), LLM-Pruner (structured)
  - LaCo recovery: stage1 layer-wise compensation + stage2 KD distillation
  - Evaluation: perplexity (WikiText2, PTB, C4) + zero-shot accuracy (8 tasks)

Usage:
  python -m laco.main MODEL_PATH DATASET --is_prune --is_wanda --sparsity_ratio 0.7 \\
      --is_train --tune_epoch 10 --tune_lr 5e-5 --is_zs
"""

import argparse
import logging
import os
import time

import torch
from transformers import AutoTokenizer

from . import data, evaluation, model_utils, pruning, utils
from .compensation import (
    llama_flap_stage1_compensation,
    llama_llmpruner_stage1_compensation,
    llama_sparsegpt_stage1_compensation,
    llama_stage2_compensation_cached,
    llama_wanda_stage1_compensation,
)
from .data import get_loaders
from .evaluation import evaluate_model, llama_evaluate_perplexity
from .training import precompute_teacher_logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="LaCo: Layer-wise Compensation for Pruned LLMs"
    )

    # Required
    parser.add_argument("model", type=str, help="Path to the model")
    parser.add_argument("--model_type", type=str, default="llama",
                        choices=["llama", "qwen"], help="Model architecture type")
    parser.add_argument(
        "dataset", type=str, choices=["wikitext2", "ptb", "c4"],
        help="Calibration dataset"
    )

    # Data
    parser.add_argument("--seed", type=int, default=120)
    parser.add_argument("--nsamples", type=int, default=2048)
    parser.add_argument("--seqlen", type=int, default=128)

    # Pruning
    parser.add_argument("--is_prune", action="store_true", help="Enable pruning")
    parser.add_argument("--is_wanda", action="store_true", help="Use Wanda pruning")
    parser.add_argument("--is_sparsegpt", action="store_true", help="Use SparseGPT pruning")
    parser.add_argument("--is_flap", action="store_true", help="Use FLAP pruning")
    parser.add_argument("--is_lmpruner", action="store_true", help="Use LLM-Pruner pruning")
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--sparsity_type", type=str, default="unstructured",
                        choices=["unstructured", "2:4", "4:8"])
    parser.add_argument("--prune_n", type=int, default=0, help="N for N:M pruning")
    parser.add_argument("--prune_m", type=int, default=0, help="M for N:M pruning")

    # FLAP specific
    parser.add_argument("--is_bias", action="store_true", help="Enable FLAP bias compensation")
    parser.add_argument("--metrics", type=str, default="WIFV",
                        choices=["IFV", "WIFV", "WIFN"])
    parser.add_argument("--structure", type=str, default="AL-AM",
                        choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM"])

    # LLM-Pruner specific
    parser.add_argument("--pruner_type", type=str, default="l2")
    parser.add_argument("--iterative_steps", type=int, default=1)
    parser.add_argument("--block_wise", action="store_true")
    parser.add_argument("--global_pruning", action="store_true")
    parser.add_argument("--lmpruning_ratio", type=float, default=0.5)
    parser.add_argument("--taylor", type=str, default="param_first")
    parser.add_argument("--grouping_strategy", type=str, default="sum")
    parser.add_argument("--block_attention_layer_start", type=int, default=3)
    parser.add_argument("--block_attention_layer_end", type=int, default=31)
    parser.add_argument("--block_mlp_layer_start", type=int, default=3)
    parser.add_argument("--block_mlp_layer_end", type=int, default=31)

    # LaCo training (stage 1)
    parser.add_argument("--is_train", action="store_true", help="Enable LaCo training")
    parser.add_argument("--tune_epoch", type=int, default=10)
    parser.add_argument("--tune_lr", type=float, default=5e-5)
    parser.add_argument("--num_layers", type=int, default=32,
                        help="Number of layers to compensate (for cumulative analysis)")
    parser.add_argument("--ablation_module", type=str, default="full",
                        choices=["attention", "ffn", "full"],
                        help="Module-wise ablation: train only attention, ffn, or both")

    # LaCo stage 2
    parser.add_argument("--enable_stage2", type=bool, default=False)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--stage2_lr", type=float, default=1e-5)

    # Evaluation
    parser.add_argument("--is_zs", action="store_true", help="Run zero-shot evaluation")

    # Output
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--demo", type=str, default="laco", help="Run name tag")
    parser.add_argument("--save", type=str, default="", help="Save model path")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--save_intervals", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint")
    parser.add_argument("--save_layer_ckpt", action="store_true", default=True)

    return parser.parse_args()


def setup_logging(args):
    os.makedirs(args.log_path, exist_ok=True)
    logger = logging.getLogger("LaCo")
    logger.setLevel(logging.INFO)

    save_path = os.path.join(
        args.log_path,
        f"{args.demo}_{time.strftime('%Y%m%d_%H%M%S')}.log",
    )
    fh = logging.FileHandler(save_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    args = parse_args()
    logger = setup_logging(args)
    logger.info(utils.format_args_for_logging(args))
    utils.set_seed(args.seed)

    # Load model
    logger.info(f"Loading model from {args.model} (type={args.model_type})")
    if args.is_lmpruner:
        model = model_utils.load_llama_for_llmpruner(args.model)
    elif args.model_type == "qwen":
        model = model_utils.load_qwen_model(args.model)
    else:
        model = model_utils.load_llama_model(args.model, device=args.device)

    model.seqlen = args.seqlen
    model.eval()

    if args.is_bias:
        model_utils.init_flap_bias(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataloader, testloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=args.model,
    )

    # Precompute teacher logits for stage 2
    cache_prefix = args.dataset + str(args.nsamples)
    cache_dir = precompute_teacher_logits(model, dataloader, logger, cache_prefix, "cuda")

    if args.is_prune:
        logger.info("Starting pruning + LaCo compensation...")

        if args.is_flap:
            model = llama_flap_stage1_compensation(model, dataloader, args.device, logger, args)

        elif args.is_lmpruner:
            model = llama_llmpruner_stage1_compensation(model, dataloader, tokenizer, args.device, logger, args)

        elif args.is_wanda:
            model = llama_wanda_stage1_compensation(model, dataloader, args.device, logger, args)

        elif args.is_sparsegpt:
            model = llama_sparsegpt_stage1_compensation(model, dataloader, args.device, logger, args)

        # Evaluate PPL after stage 1
        for ds in ["wikitext2", "ptb", "c4"]:
            logger.info(f"Evaluating on {ds}")
            _, testloader_ds, _ = get_loaders(
                ds, seed=args.seed, seqlen=model.seqlen, model=args.model,
            )
            model.eval()
            llama_evaluate_perplexity(model, testloader_ds, args, logger=logger, dataset=ds)

        # Stage 2: end-to-end KD distillation
        if args.enable_stage2:
            logger.info("Starting stage 2 distillation...")
            model = llama_stage2_compensation_cached(model, dataloader, logger, args, cache_dir)

    # Final PPL evaluation
    for ds in ["wikitext2", "ptb", "c4"]:
        logger.info(f"Final evaluation on {ds}")
        _, testloader_ds, _ = get_loaders(
            ds, seed=args.seed, seqlen=model.seqlen, model=args.model,
        )
        model.eval()
        llama_evaluate_perplexity(model, testloader_ds, args, logger=logger, dataset=ds)

    # Zero-shot evaluation
    if args.is_zs:
        logger.info("Running zero-shot evaluation...")
        all_tasks = [
            "rte", "boolq", "hellaswag", "winogrande", "obqa",
            "piqa", "ai2_arc_easy", "ai2_arc_challenge",
        ]
        for task in all_tasks:
            acc = evaluate_model(model, tokenizer, task=task, device=args.device, logger=logger)
            logger.info(f"{task} accuracy: {acc}")

    # Save model
    if args.save:
        model.half()
        torch.save({"model": model, "tokenizer": tokenizer}, args.save)

    logger.info("Done.")


if __name__ == "__main__":
    main()
