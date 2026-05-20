"""Evaluation utilities for LaCo: perplexity and zero-shot accuracy."""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .features import FeatureDataset


# Default zero-shot evaluation dataset paths.
# Override these via environment variables or by editing this dict.
DATASET_PATHS = {
    "rte": os.environ.get("RTE_PATH", ""),
    "boolq": os.environ.get("BOOLQ_PATH", ""),
    "hellaswag": os.environ.get("HELLASWAG_PATH", ""),
    "winogrande": os.environ.get("WINOGRANDE_PATH", ""),
    "obqa": os.environ.get("OBQA_PATH", ""),
    "piqa": {
        "data_path": os.environ.get("PIQA_DATA_PATH", ""),
        "label_path": os.environ.get("PIQA_LABEL_PATH", ""),
    },
    "ai2_arc_easy": os.environ.get("ARC_EASY_PATH", ""),
    "ai2_arc_challenge": os.environ.get("ARC_CHALLENGE_PATH", ""),
}


def load_json_or_jsonl(path: str):
    """Load a JSON or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = ""
        for ch in f.read(1000):
            if not ch.isspace():
                first_char = ch
                break
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


class LMEvalStyleEvaluator:
    """Memory-friendly log-likelihood evaluator aligned with lm-eval-harness."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def _loglikelihood_choice(self, context: str, continuation: str, mode="sum"):
        full = context + continuation
        enc = self.tokenizer(full, return_tensors="pt", truncation=False)
        ctx_enc = self.tokenizer(context, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"].to(self.device)
        ctx_len = ctx_enc["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits[:, :-1, :].detach()
            logits_cpu = logits.to("cpu", dtype=torch.float32)
            shift_labels = input_ids[:, 1:].cpu()
            log_probs = torch.nn.functional.log_softmax(logits_cpu, dim=-1)
            token_logprobs = log_probs.gather(
                2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            cont_segment = token_logprobs[0, ctx_len - 1:]
            val = cont_segment.sum().item() if mode == "sum" else cont_segment.mean().item()

        del outputs, logits, logits_cpu, log_probs, token_logprobs, shift_labels
        torch.cuda.empty_cache()
        return val


class DatasetEvaluator:
    """Evaluator for zero-shot commonsense reasoning benchmarks."""

    def __init__(self, model, tokenizer, device, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lmeval = LMEvalStyleEvaluator(model, tokenizer, device)
        self.logger = logger

    def evaluate_boolq(self, path, max_samples=None):
        data = load_json_or_jsonl(path)
        correct, total = 0, 0
        for i, s in enumerate(tqdm(data, desc="BoolQ")):
            context = f"{s['passage']}\nQuestion: {s['question']}\nAnswer:"
            choices, gold_idx = [" yes", " no"], 0 if s["answer"] else 1
            scores = [self.lmeval._loglikelihood_choice(context, c, mode="mean") for c in choices]
            pred = int(torch.tensor(scores).argmax())
            correct += (pred == gold_idx)
            total += 1
        return correct / total

    def evaluate_piqa(self, data_path, label_path, max_samples=None):
        data = load_json_or_jsonl(data_path)
        labels = [int(l.strip()) for l in open(label_path)]
        correct, total = 0, 0
        for i, s in enumerate(tqdm(data, desc="PIQA")):
            goal = s["goal"].strip()
            choices = [s["sol1"].strip(), s["sol2"].strip()]
            gold_idx = labels[i]
            scores = [self.lmeval._loglikelihood_choice(goal, " " + c, mode="sum") for c in choices]
            pred = int(torch.tensor(scores).argmax())
            correct += (pred == gold_idx)
            total += 1
        return correct / total

    def evaluate_obqa(self, path, max_samples=None):
        data = load_json_or_jsonl(path)
        correct, total = 0, 0
        for i, s in enumerate(tqdm(data, desc="OBQA")):
            question = s["question_stem"].strip()
            choices = [c.strip() for c in s["choices"]["text"]]
            gold_idx = ord(s["answerKey"]) - 65
            scores = [self.lmeval._loglikelihood_choice(question, " " + c, mode="mean") for c in choices]
            pred = int(torch.tensor(scores).argmax())
            correct += (pred == gold_idx)
            total += 1
        return correct / total

    def evaluate_hellaswag(self, path, max_samples=None):
        data = load_json_or_jsonl(path)
        correct, total = 0, 0
        for i, s in enumerate(tqdm(data, desc="HellaSwag")):
            ctx = s["ctx"].strip()
            endings = [e.strip() for e in s["endings"]]
            gold_idx = int(s["label"])
            scores = [self.lmeval._loglikelihood_choice(ctx, " " + e, mode="mean") for e in endings]
            pred = int(torch.tensor(scores).argmax())
            correct += (pred == gold_idx)
            total += 1
        return correct / total

    def evaluate_ai2_arc(self, path, max_samples=None):
        data = load_json_or_jsonl(path)
        correct, total = 0, 0
        for i, s in enumerate(tqdm(data, desc="AI2-ARC")):
            question = s["question"].strip()
            choices = [c.strip() for c in s["choices"]["text"]]
            gold_idx = ord(s["answerKey"]) - 65
            full_sentences = [f"{question}\nAnswer: {c}" for c in choices]
            scores = [self.lmeval._loglikelihood_choice("", sent, mode="sum") for sent in full_sentences]
            pred = int(torch.tensor(scores).argmax())
            correct += (pred == gold_idx)
            total += 1
        return correct / total

    def evaluate_rte(self, path, max_samples=None):
        data = load_json_or_jsonl(path)
        correct, total = 0, 0
        fewshot_examples = [
            {"premise": "The cat is sleeping on the mat.", "hypothesis": "The animal is resting.", "answer": "yes"},
            {"premise": "The man is playing the guitar.", "hypothesis": "The man is cooking dinner.", "answer": "no"},
            {"premise": "A boy is riding a bicycle.", "hypothesis": "A child is on a bike.", "answer": "yes"},
        ]
        fewshot_prompt = ""
        for ex in fewshot_examples:
            fewshot_prompt += (
                f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
                f"Question: Does the premise entail the hypothesis?\nAnswer: {ex['answer']}\n\n"
            )
        for i, s in enumerate(tqdm(data, desc="RTE")):
            premise, hypothesis = s["text1"].strip(), s["text2"].strip()
            gold_idx = int(s["label"])
            context = (
                fewshot_prompt
                + f"Premise: {premise}\nHypothesis: {hypothesis}\n"
                + "Question: Does the premise entail the hypothesis?\nAnswer:"
            )
            scores = [self.lmeval._loglikelihood_choice(context, c, mode="sum") for c in [" yes", " no"]]
            pred = int(torch.tensor(scores).argmax())
            correct += (pred == gold_idx)
            total += 1
        return correct / total

    def evaluate_winogrande(self, path, max_samples=None):
        data = load_json_or_jsonl(path)
        correct, total = 0, 0
        for i, s in enumerate(tqdm(data, desc="WinoGrande")):
            sentence = s["sentence"]
            option1, option2 = s["option1"], s["option2"]
            gold_idx = int(s["answer"]) - 1
            choices = [sentence.replace("_", option1), sentence.replace("_", option2)]
            scores = [self.lmeval._loglikelihood_choice("", sent, mode="sum") for sent in choices]
            pred = int(torch.tensor(scores).argmax())
            correct += (pred == gold_idx)
            total += 1
        return correct / total


def evaluate_model(model, tokenizer, task, max_samples=None, device="cpu", logger=None):
    """Evaluate a model on a zero-shot benchmark task.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer for the model.
        task: Task name from DATASET_PATHS.
        device: Device to run on.
        logger: Optional logger.

    Returns:
        Accuracy score (float).
    """
    if task not in DATASET_PATHS:
        raise ValueError(f"Unsupported task: {task}")

    dataset_entry = DATASET_PATHS[task]
    if isinstance(dataset_entry, dict):
        dataset_path = dataset_entry["data_path"]
        label_path = dataset_entry["label_path"]
    else:
        dataset_path = dataset_entry
        label_path = None

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}\n"
            f"Set the {task.upper()}_PATH environment variable or update DATASET_PATHS in evaluation.py."
        )

    evaluator = DatasetEvaluator(model.to(device), tokenizer, device, logger)

    task_map = {
        "boolq": evaluator.evaluate_boolq,
        "ai2_arc_easy": evaluator.evaluate_ai2_arc,
        "ai2_arc_challenge": evaluator.evaluate_ai2_arc,
        "hellaswag": evaluator.evaluate_hellaswag,
        "winogrande": evaluator.evaluate_winogrande,
        "obqa": evaluator.evaluate_obqa,
        "piqa": lambda path, _: evaluator.evaluate_piqa(path, label_path, max_samples),
        "rte": evaluator.evaluate_rte,
    }

    return task_map[task](dataset_path, max_samples)


@torch.no_grad()
def llama_evaluate_perplexity(model, testenc, args, logger, dataset: str, log_wandb: bool = False):
    """Evaluate perplexity on a test set using sequential layer processing.

    Memory-efficient: processes one layer at a time.
    """
    logger.info("Evaluating ...")
    dev = args.device

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)

    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16, device=torch.device("cpu"),
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.cpu()
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.cpu()

    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        logger.info(f"Processing layer {i}")
        layer = layers[i].to(dev)
        inps_dataset = FeatureDataset(inps, device=dev)
        dataloader = DataLoader(inps_dataset, batch_size=1, shuffle=False, drop_last=False)
        for idx, batch in dataloader:
            inps[idx] = layer(
                batch, attention_mask=attention_mask, position_ids=position_ids
            )[0].half().cpu()
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    inps_dataset = FeatureDataset(inps, device=dev)
    dataloader = DataLoader(inps_dataset, batch_size=1, shuffle=False, drop_last=False)
    for idx, inp in dataloader:
        hidden_states = inp
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (idx * model.seqlen):((idx + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    logger.info(f"*****************************************************************")
    logger.info(f"Perplexity: {ppl.item():3f}")
    logger.info(f"*****************************************************************")

    if log_wandb:
        import wandb
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache
    return ppl.item()
