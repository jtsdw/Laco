import math
import time
import math
import torch
import torch.nn as nn
import transformers
import os
from .quantization import *
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
import gc
import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts
import copy
from .training import *
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class FlapMetricTracker:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, metric):
        self.layer = layer
        self.dev = self.layer.weight.device
        # print(f"FlapMetricTracker: layer {layer} on device {self.dev}")
        self.out_dim = layer.weight.data.shape[0]
        self.in_dim = layer.weight.data.shape[1]
        self.type = metric
        self.nsamples = 0

        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        if self.type == "WIFN":
            self.scaler_inp = torch.zeros((self.in_dim), device=self.dev)
        else:   
            self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (dim, seqlen)

        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.type == "WIFN":
            inp = inp.type(torch.float32)
            self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
            self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / (self.nsamples + batch_size)
        else:
            if self.nsamples == 0:
                self.fluc_inp = 0
            else:
                self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号

        self.nsamples += batch_size

        
    def free(self):
        self.baseline_inp = None
        if hasattr(self, 'fluc_inp'):
            self.fluc_inp = None
        if hasattr(self, 'scaler_inp'):
            self.scaler_inp = None
        torch.cuda.empty_cache()  

metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}

def flap_compress_layer(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=False, unstr=False, pruned_weights=None,keep_mask = False):
    """
    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    # unstr = False
    print(f"the bias is {bias} and the unstr is {unstr}")
    attn_bias = 0
    mlp_bias = 0

    # if pruned_weights is None:
    #     pruned_weights = {}
    if unstr:  # Only mask, do not really prune
        # Attention Weight Masking
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            # print(f"begin attn_mask.shape {attn_mask.shape}")
            attn_mask = attn_mask.repeat_interleave(128)
            mask_indices = torch.where(~attn_mask)[0]
            # if len(mask_indices) > 0:
            #     pruned_weights['self_attn.q_proj'] = layer.self_attn.q_proj.weight.data[mask_indices].clone()
            #     pruned_weights['self_attn.k_proj'] = layer.self_attn.k_proj.weight.data[mask_indices].clone()
            #     pruned_weights['self_attn.v_proj'] = layer.self_attn.v_proj.weight.data[mask_indices].clone()
            # Apply the mask to the query, key and value projection weights
            # print(f"the shape of attn_mask is {attn_mask.unsqueeze(-1).shape} and the shape of layer.self_attn.q_proj.weight.data is {layer.self_attn.q_proj.weight.data.shape}")
            layer.self_attn.q_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)        
            layer.self_attn.k_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            layer.self_attn.v_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.self_attn.o_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                # print(f"bias ok")
                # print(f"the device of attn_mean_inp is {attn_mean_inp.device} and the device of attn_mask is {attn_mask.device} and the device of output_weight is {output_weight.device}")
                # print(f"the attn_mean_inp is {attn_mean_inp} and the attn_mask is {attn_mask} and the output_weight is {output_weight.T}")
                output_bias = ((attn_mean_inp.float() * ~attn_mask.to(device)) @ output_weight.T.float())
                # print(f"output_bias is {output_bias}")
            # else:
            #     # print("ok attn bias")
            #     output_bias = ((attn_mean_inp.float() * ~attn_mask.to(device)) @ output_weight.T.float())
            #     # output_bias = attn_mean_inp.float() @ output_weight.T.float()
            #     attn_bias = output_bias
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.self_attn.o_proj.bias.data = output_bias
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Masking
        if mlp_mask is not None:
            mask_indices = torch.where(~mlp_mask)[0]
            if len(mask_indices) > 0:
                pruned_weights['mlp.up_proj'] = layer.mlp.up_proj.weight.data[mask_indices].clone()
                pruned_weights['mlp.gate_proj'] = layer.mlp.gate_proj.weight.data[mask_indices].clone()
            # Apply the mask to the up and gate projection weights
            layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)

            # # 保存mlp的mask  此时尺度为 11008
            # if keep_mask:
            #     layer.mlp.set_hidden_mask(mlp_mask)
            
            output_weight = layer.mlp.down_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp.float() * ~mlp_mask.to(device)) @ output_weight.T.float())
                # mlp_bias = output_bias
            # else:
            #     # print("ok mlp bias")
            #     #   11008   11008   4096,11008
            #     # print(f"the shape of mlp_mean_inp is {mlp_mean_inp.shape} and shape of mlp_mask os {mlp_mask.shape} and shape of output_weight is {output_weight.shape}")
            #     output_bias = ((mlp_mean_inp.float() * ~mlp_mask.to(device)) @ output_weight.T.float())
            #     # output_bias = mlp_mean_inp.float() @ output_weight.T.float()
            #     mlp_bias = output_bias
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.mlp.down_proj.bias.data = output_bias
            layer.mlp.down_proj.weight.data = output_weight
    
    else:
        # Real Pruning
        # Attention Weight Pruning
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            # Collect weights before pruning
            prune_indices = torch.where(~attn_mask)[0]
            if len(prune_indices) > 0:
                pruned_weights['self_attn.q_proj'] = layer.self_attn.q_proj.weight.data[prune_indices].clone()
                pruned_weights['self_attn.k_proj'] = layer.self_attn.k_proj.weight.data[prune_indices].clone()
                pruned_weights['self_attn.v_proj'] = layer.self_attn.v_proj.weight.data[prune_indices].clone()
            # Prune the query, key and value projection weights
            # We reduce the size of the weights based on the attention mask
            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]].to(device)
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]].to(device)
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]].to(device)
            
            # Update output dimensions of q, k, v projections based on remaining heads
            layer.self_attn.q_proj.out_features = attn_mask.sum().item()
            layer.self_attn.k_proj.out_features = attn_mask.sum().item()
            layer.self_attn.v_proj.out_features = attn_mask.sum().item()
            output_weight = layer.self_attn.o_proj.weight.data
            
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp.float() * ~attn_mask.to(device)) @ output_weight.T.float() )
                
            # Prune the output projection weight
            # Collect and prune o_proj weights
            prune_indices = torch.where(~attn_mask)[0]
            if len(prune_indices) > 0:
                pruned_weights['self_attn.o_proj'] = layer.self_attn.o_proj.weight.data[:, prune_indices].clone()
            output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]
            # Update layer configurations for the new output shape after pruning
            layer.self_attn.num_heads = retain_heads
            layer.self_attn.hidden_size = retain_heads * 128
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.self_attn.o_proj.in_features = attn_mask.sum().item()
                # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                if layer.self_attn.o_proj.bias is not None:
                    layer.self_attn.o_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Pruning
        if mlp_mask is not None:
            # Prune the up and gate projection weights
            # Collect weights before pruning
            prune_indices = torch.where(~mlp_mask)[0]
            if len(prune_indices) > 0:
                pruned_weights['mlp.up_proj'] = layer.mlp.up_proj.weight.data[prune_indices].clone()
                pruned_weights['mlp.gate_proj'] = layer.mlp.gate_proj.weight.data[prune_indices].clone()
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]].to(device)
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]].to(device)
            
            # Update output dimensions of up and gate projections based on the mlp mask
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            
            output_weight = layer.mlp.down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp.float() * ~mlp_mask.to(device)) @ output_weight.T.float() )
                mlp_bias = output_bias
              
            # Collect and prune down_proj weights
            prune_indices = torch.where(~mlp_mask)[0]
            if len(prune_indices) > 0:
                pruned_weights['mlp.down_proj'] = layer.mlp.down_proj.weight.data[:, prune_indices].clone()
            # Prune the down projection weight
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.mlp.down_proj.in_features = mlp_mask.sum().item()
                # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                if layer.mlp.down_proj.bias is not None:
                    layer.mlp.down_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.mlp.down_proj.weight.data = output_weight
        
    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()
    # return pruned_weights  # Return collected weights
    # print(f"the attn_bias is {attn_bias} and the mlp_bias is {mlp_bias}")
    return layer,attn_bias,mlp_bias 







def load_flap_masks_and_baseline(p,dev):
    attn_mask = torch.load(os.path.join(p, "attn_mask.pt")).to(dev)
    mlp_mask = torch.load(os.path.join(p, "mlp_mask.pt")).to(dev)    
    attn_baseline_inp_list = torch.load(os.path.join(p, "attn_baseline_inp_list.pt"))
    mlp_baseline_inp_list = torch.load(os.path.join(p, "mlp_baseline_inp_list.pt"))

    attn_baseline_inp_list = [item.to(dev) for item in attn_baseline_inp_list]
    mlp_baseline_inp_list = [item.to(dev) for item in mlp_baseline_inp_list]
    
    return attn_mask,mlp_mask,attn_baseline_inp_list,mlp_baseline_inp_list


#  flap 剪枝并保存





def llmpruner_gpu(model,tokenizer,logger,args):
    dev = "cpu"
    model.half()
    model = model.to(dev)

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(dev) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.info("Use {} pruner...".format(pruner_type))
    
    if args.block_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.lmpruning_ratio, 
            "ignored_layers":[],
            "channel_groups": {
            },
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": None, 
            "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
        }
        logger.info("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        logger.info("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))
        print(f"device of model is {model.device} and for is {forward_prompts.device}")
        
        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()

        logger.info("Start Pruning")
        for i in range(args.iterative_steps):
            # if pruner_type in ['taylor']:
            #     example_prompts = get_examples('c4', tokenizer, 128, seq_len = 64).to(dev)
            #     logger.info("Start Backwarding in iterative steps = {}...".format(i))
            #     if args.taylor in ['param_mix', 'param_second']:
            #         for j in range(128):
            #             batch_input = example_prompts[j].unsqueeze(0)
            #             loss = model(batch_input, labels=batch_input).loss
            #             logger.info("Loss = {}".format(loss))
            #             loss.backward()
            #             print("after backward")
            #             for module_param in model.parameters():
            #                 print("find module_param")
            #                 module_param.grad = module_param.grad * module_param.grad / args.num_examples
            #                 if hasattr(module_param, 'acc_grad'):
            #                     module_param.acc_grad += module_param.grad
            #                 else:
            #                     module_param.acc_grad = copy.deepcopy(module_param.grad)
            #             print("after acc")
            # -------------------- 替换后的高效逻辑 --------------------
            if pruner_type in ['taylor']:
                example_prompts = get_examples('c4', tokenizer, 128, seq_len = 64).to(dev)
                logger.info("Start Backwarding in iterative steps = {}...".format(i))
                # 确保 acc_grad 已初始化（在 i 循环外部执行一次）
                if i == 0:
                    for module_param in model.parameters():
                        if module_param.requires_grad:
                            # 预先分配累加张量，防止后续 deepcopy
                            module_param.acc_grad = torch.zeros_like(module_param.data) 
                # 确保缩放因子正确
                scale_factor = 1.0 / 128 # 128 对应于你的 j 循环次数
                for j in range(128):
                    batch_input = example_prompts[j].unsqueeze(0)
                    model.zero_grad() 
                    
                    loss = model(batch_input, labels=batch_input).loss
                    loss.backward()
                    
                    for module_param in model.parameters():
                        if module_param.grad is not None and module_param.requires_grad:
                            grad_data = module_param.grad.data
                            module_param.acc_grad.addcmul_(grad_data, grad_data, value=scale_factor)
                        
                # 清理本 Batch 的梯度
                        model.zero_grad()
                        del loss.grad
                    
                loss = model(example_prompts, labels=example_prompts).loss
                logger.info("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
        
            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        del pruner

    logger.info("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    # if args.save_model:
    #     model.half()
    #     torch.save({
    #         'model': model, 
    #         'tokenizer': tokenizer,
    #     }, logger.best_checkpoint_path)

    return model


# Define WandaWeightTracker class
class WandaWeightTracker:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


#
# sparsegpt所使用到的剪枝函数
#
##################################################################################################

class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def free(self):
        self.H = None
        torch.cuda.empty_cache()

@torch.no_grad()

def compress_wanda_with_constraint(layer, dev, args, inps, outs, attention_mask, position_ids):
    """
    改进版compress_wanda，返回稀疏性约束对象
    """
    prune_m = args.prune_m
    prune_n = args.prune_n
    subset = find_layers(layer)
    
        # 修改开始：先检查是否为 None 再移动设备
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    if position_ids is not None:
        position_ids = position_ids.to(dev)

    wrapped_layers = {}
    for name in subset:
        wrapped_layers[name] = WandaWeightTracker(subset[name])

    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in wrapped_layers:
        handles.append(subset[name].register_forward_hook(add_batch(name)))

    # 第一次前向传播：收集统计信息
    for j in range(128):
        with torch.no_grad():
            current_inp = inps[j].unsqueeze(0).to(dev)
            output_tensor = layer(current_inp, attention_mask=attention_mask, position_ids=position_ids)[0]
            outs[j] = output_tensor.cpu() 
            del current_inp, output_tensor
            torch.cuda.empty_cache()
    
    for h in handles:
        h.remove()

    # 执行剪枝
    for name in subset:
        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
        W_mask = (torch.zeros_like(W_metric) == 1)
        
        if prune_n != 0:
            print("structured n:m sparsity")
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            print("unstructured pruning")
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

        subset[name].weight.data[W_mask] = 0

    # 第二次前向传播：验证剪枝效果
    for j in range(128):
        with torch.no_grad():
            current_inp = inps[j].unsqueeze(0).to(dev)
            output_tensor = layer(current_inp, attention_mask=attention_mask, position_ids=position_ids)[0]
            outs[j] = output_tensor.cpu() 
            del current_inp, output_tensor
            torch.cuda.empty_cache()
    
    inps, outs = outs, inps
    del attention_mask, position_ids
    
    # 创建稀疏性约束对象
    sparsity_constraint = SparsityConstraint(layer)
    
    return layer, inps, outs, sparsity_constraint




# ============================================================================
# 修改后的compress_wanda函数 - 返回SparsityConstraint
# ============================================================================

def compress_wanda_with_constraint_qwen(layer,rotary_emb, dev, args, inps, outs, attention_mask, position_ids):
    """
    改进版compress_wanda，返回稀疏性约束对象
    """
    prune_m = args.prune_m
    prune_n = args.prune_n
    subset = find_layers(layer)
    
    attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)

    wrapped_layers = {}
    for name in subset:
        wrapped_layers[name] = WandaWeightTracker(subset[name])

    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in wrapped_layers:
        handles.append(subset[name].register_forward_hook(add_batch(name)))

    # === 【修改点 1】: 手动计算 position_embeddings ===
    # Qwen2 需要传入 (cos, sin) 元组
    device = dev
    position_embeddings = None
    if position_ids is not None:
        # 准备一个临时的 x 用于推断 device 和 dtype
        # inps 的形状通常是 [nsamples, seqlen, hidden]
        # 我们取第一个样本,增加 batch 维度,并移到 device
        temp_x = inps[0].unsqueeze(0).to(device)
        temp_pos = position_ids.to(device)
        
        # 调用 rotary_emb 计算 cos, sin
        with torch.no_grad():
            cos, sin = rotary_emb(temp_x, temp_pos)
            position_embeddings = (cos, sin)
    # ===============================================
    
    # 第一次前向传播：收集统计信息
    for j in range(128):
        with torch.no_grad():
            current_inp = inps[j].unsqueeze(0).to(dev)
            output_tensor = layer(current_inp, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                position_embeddings=position_embeddings, # 新增参数
                )[0]
            outs[j] = output_tensor.cpu() 
            del current_inp, output_tensor
            torch.cuda.empty_cache()
    
    for h in handles:
        h.remove()

    # 执行剪枝
    for name in subset:
        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
        W_mask = (torch.zeros_like(W_metric) == 1)
        
        if prune_n != 0:
            print("structured n:m sparsity")
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            print("unstructured pruning")
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

        subset[name].weight.data[W_mask] = 0

    # 第二次前向传播：验证剪枝效果
    for j in range(128):
        with torch.no_grad():
            current_inp = inps[j].unsqueeze(0).to(dev)
            output_tensor = layer(current_inp, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                position_embeddings=position_embeddings, # 新增参数
                )[0]
            outs[j] = output_tensor.cpu() 
            del current_inp, output_tensor
            torch.cuda.empty_cache()
    
    inps, outs = outs, inps
    del attention_mask, position_ids
    
    # 创建稀疏性约束对象
    sparsity_constraint = SparsityConstraint(layer)
    
    return layer, inps, outs, sparsity_constraint

@torch.no_grad()
def compress_sparsegpt_with_constraint_qwen(layer, rotary_emb,dev, args, inps, outs, attention_mask, position_ids):
    print("compress now !")
    # if f"model.layers.{i}" in model.hf_device_map:
    #     dev = model.hf_device_map[f"model.layers.{i}"]
    #     print(f"layer {i} device {dev}")
    prune_m = args.prune_m
    prune_n = args.prune_n
    print(f"prune_m is {prune_m} and prune_n is {prune_n}")
    # subset = find_layers(layer)
    attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)
    subset = find_layers(layer)
    # # 确保拿到 rotary_emb 模块
    # rotary_emb = model.model.rotary_emb
    gpts = {}
    for name in subset:
        gpts[name] = SparseGPT(subset[name])

    def add_batch(name):
        def tmp(_, inp, out):
            gpts[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in gpts:
        handles.append(subset[name].register_forward_hook(add_batch(name)))

    # === 【修改点 1】: 手动计算 position_embeddings ===
    # Qwen2 需要传入 (cos, sin) 元组
    position_embeddings = None
    device = dev
    if position_ids is not None:
        # 准备一个临时的 x 用于推断 device 和 dtype
        # inps 的形状通常是 [nsamples, seqlen, hidden]
        # 我们取第一个样本,增加 batch 维度,并移到 device
        temp_x = inps[0].unsqueeze(0).to(device)
        temp_pos = position_ids.to(device)
        
        # 调用 rotary_emb 计算 cos, sin
        with torch.no_grad():
            cos, sin = rotary_emb(temp_x, temp_pos)
            position_embeddings = (cos, sin)

    for j in range(128):
        with torch.no_grad():
            current_inp = inps[j].unsqueeze(0).to(dev)

            output_tensor = layer(current_inp, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    position_embeddings=position_embeddings, # 新增参数
                )[0]

            outs[j] = output_tensor.cpu() 

            del current_inp, output_tensor
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    for name in gpts:
        # print(i, name)
        print('Pruning ...')

        gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
        gpts[name].free()
    for j in range(128):
        with torch.no_grad():
            current_inp = inps[j].unsqueeze(0).to(dev)

            output_tensor = layer(current_inp, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    position_embeddings=position_embeddings, # 新增参数
                )[0]
            outs[j] = output_tensor.cpu() 

            del current_inp, output_tensor
            torch.cuda.empty_cache()
    # layers[i] = layer 
   
    # 创建稀疏性约束对象
    sparsity_constraint = SparsityConstraint(layer)
    inps, outs = outs, inps
    del attention_mask,position_ids,gpts
    gc.collect
    torch.cuda.empty_cache()
    return layer,inps,outs, sparsity_constraint



@torch.no_grad()
def compress_sparsegpt_with_constraint(layer, dev, args, inps, outs, attention_mask, position_ids):
    print("compress now !")
    # if f"model.layers.{i}" in model.hf_device_map:
    #     dev = model.hf_device_map[f"model.layers.{i}"]
    #     print(f"layer {i} device {dev}")
    prune_m = args.prune_m
    prune_n = args.prune_n
    print(f"prune_m is {prune_m} and prune_n is {prune_n}")
    # subset = find_layers(layer)
    attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)
    subset = find_layers(layer)

    gpts = {}
    for name in subset:
        gpts[name] = SparseGPT(subset[name])

    def add_batch(name):
        def tmp(_, inp, out):
            gpts[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in gpts:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    for j in range(128):
        with torch.no_grad():
            current_inp = inps[j].unsqueeze(0).to(dev)

            output_tensor = layer(current_inp, attention_mask=attention_mask, position_ids=position_ids)[0]

            outs[j] = output_tensor.cpu() 

            del current_inp, output_tensor
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    for name in gpts:
        # print(i, name)
        print('Pruning ...')

        gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
        gpts[name].free()
    for j in range(128):
        with torch.no_grad():
            current_inp = inps[j].unsqueeze(0).to(dev)

            output_tensor = layer(current_inp, attention_mask=attention_mask, position_ids=position_ids)[0]

            outs[j] = output_tensor.cpu() 

            del current_inp, output_tensor
            torch.cuda.empty_cache()
    # layers[i] = layer 
   
    # 创建稀疏性约束对象
    sparsity_constraint = SparsityConstraint(layer)
    inps, outs = outs, inps
    del attention_mask,position_ids,gpts
    gc.collect
    torch.cuda.empty_cache()
    return layer,inps,outs, sparsity_constraint



