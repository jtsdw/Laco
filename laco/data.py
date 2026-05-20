"""Data loading utilities for LaCo.

Provides calibration and evaluation data loaders for WikiText2, PTB, and C4 datasets.
"""

import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def get_tokenizer(model_name: str):
    """Load tokenizer with appropriate settings for the model type."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    return tokenizer


class HiddenRepairDataset(Dataset):
    """Dataset for hidden state repair training.

    Splits tokenized text into fixed-length sequences.
    """

    def __init__(self, input_ids, seqlen):
        self.seqlen = seqlen
        n_samples = input_ids.numel() // seqlen
        self.data = input_ids[:n_samples * seqlen].view(n_samples, seqlen)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """Load WikiText2 calibration data."""
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    g = torch.Generator()
    g.manual_seed(seed)

    trainloader = []
    max_idx = trainenc.input_ids.shape[1] - seqlen - 1
    random_indices = torch.randint(0, max_idx, (nsamples,), generator=g).tolist()

    for i in random_indices:
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, tokenizer):
    """Load Penn Treebank calibration data."""
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    g = torch.Generator()
    g.manual_seed(seed)

    trainloader = []
    max_idx = trainenc.input_ids.shape[1] - seqlen - 1
    random_indices = torch.randint(0, max_idx + 1, (nsamples,), generator=g).tolist()

    for i in random_indices:
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    """Load C4 calibration data."""
    traindata = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    trainloader = []
    # Process streaming data with shuffled access
    traindata = traindata.shuffle(seed=seed, buffer_size=10000)
    train_iter = iter(traindata)

    while len(trainloader) < nsamples:
        doc = next(train_iter)
        trainenc = tokenizer(doc["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] <= seqlen:
            continue
        max_idx = trainenc.input_ids.shape[1] - seqlen - 1
        i = torch.randint(0, max_idx + 1, (1,), generator=g).item()
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Validation: load a small set
    valdata = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    valdata = valdata.take(1100)
    valenc = tokenizer(
        " ".join([d["text"] for d in valdata]), return_tensors="pt"
    )
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=""):
    """Main entry point: get calibration data loaders for a dataset.

    Args:
        name: Dataset name - "wikitext2", "ptb", or "c4"
        nsamples: Number of calibration samples
        seed: Random seed
        seqlen: Sequence length
        model: Model name/path (for tokenizer loading)

    Returns:
        (trainloader, testloader, tokenizer)
    """
    tokenizer = get_tokenizer(model)

    if "wikitext2" in name:
        dataloader, testloader = get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "ptb" in name:
        dataloader, testloader = get_ptb(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        dataloader, testloader = get_c4(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return dataloader, testloader, tokenizer
