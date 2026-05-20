"""Model loading utilities for LaCo."""

import torch


def load_llama_model(model_path: str, device: str = "cuda:0", dtype=torch.float16):
    """Load a LLaMA model with pruning-compatible wrappers."""
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from llama_pru import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
    )
    return model


def load_llama_for_llmpruner(model_path: str):
    """Load a LLaMA model compatible with LLM-Pruner structured pruning."""
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    return model


def load_llama_model_cpu(model_path: str):
    """Load a LLaMA model on CPU with float32 (for memory-constrained loading)."""
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from llama_pru import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
    )
    return model


def load_qwen_model(model_path: str):
    """Load a Qwen model."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    return model


def init_flap_bias(model, num_layers: int = 32):
    """Initialize bias terms for FLAP structured pruning.

    Zero-initializes o_proj and down_proj biases if they don't exist.
    """
    for i in range(num_layers):
        o_proj = model.model.layers[i].self_attn.o_proj
        if o_proj.bias is not None:
            torch.nn.init.zeros_(o_proj.bias)
        else:
            out_features = o_proj.out_features
            o_proj.bias = torch.nn.Parameter(
                torch.zeros(out_features, device=o_proj.weight.device,
                           dtype=o_proj.weight.dtype)
            )

        down_proj = model.model.layers[i].mlp.down_proj
        if down_proj.bias is not None:
            torch.nn.init.zeros_(down_proj.bias)
        else:
            out_features = down_proj.out_features
            down_proj.bias = torch.nn.Parameter(
                torch.zeros(out_features, device=down_proj.weight.device,
                           dtype=down_proj.weight.dtype)
            )
