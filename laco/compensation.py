########################
# 标准库
########################
import os
import time
import random
import copy
import logging
import gc
import glob
########################
# 第三方库（通用）
########################
import numpy as np
from tqdm import tqdm

########################
# PyTorch 核心依赖
########################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

########################
# CUDA AMP
########################
from torch.cuda import amp
from torch.cuda.amp import GradScaler, autocast

########################
# Transformers / HuggingFace
########################
from transformers import AutoTokenizer
from accelerate import Accelerator

########################
# LLMPruner 库
########################
import LLMPruner.torch_pruning as tp
from LLMPruner.models.hf_llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaAttention,
    LlamaMLP
)
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

########################
# 本项目内部模块
########################
from .pruning import *
from .quantization import *
from .data import get_loaders
from .utils import find_layers
from .features import DoubleFeatureDataset, FeatureDataset
from .training import *
from .evaluation import evaluate_model

########################
# wandb（可选）
########################
try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


def llama_flap_stage1_compensation(
    model, 
    dataloader, 
    dev, 
    logger, 
    args
):
    """
    两阶段补偿训练的主函数
    
    阶段1：逐层独立训练（Layer-wise Alignment）
    阶段2：端到端联合训练（End-to-End Alignment）
    """
    
    logger.info('='*80)
    logger.info('Starting Two-Stage Compensation Training')
    logger.info('='*80)
    
    # 创建配置对象
    config = CompensationConfig(
        # 阶段1
        stage1_epochs=getattr(args, 'tune_epoch', 0),
        stage1_lr=getattr(args, 'tune_lr', 1e-5),
        
        # 阶段2
        enable_stage2=getattr(args, 'enable_stage2', True),
        stage2_epochs=getattr(args, 'stage2_epochs', 50),
        stage2_lr=getattr(args, 'stage2_lr', 1e-5),
        
        # 损失权重
        alpha_hidden=getattr(args, 'alpha_hidden', 0.4),
        beta_logits=getattr(args, 'beta_logits', 0.5),
        gamma_hard=getattr(args, 'gamma_hard', 0.1),
        
        # 其他
        layer_weight_strategy=getattr(args, 'layer_weight_strategy', 'uniform'),
        kl_temperature=getattr(args, 'kl_temperature', 2.0),
        use_amp=getattr(args, 'use_amp', True),
        cache_hidden_states=getattr(args, 'cache_hiddens', True),
        save_checkpoint=getattr(args, 'save_checkpoint', True),
        checkpoint_dir=getattr(args, 'checkpoint_dir', './checkpoints'),
    )
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建checkpoint目录
    import os
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # 移动组件到设备
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)  # ← 添加这一行！
    for layer in layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(dev)
    
    # 初始化数据流状态
    dtype = next(iter(model.parameters())).dtype
    err_stream_state = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16,
        device=dev
    )
    
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
    # 数据捕获（你的代码）
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            err_stream_state[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])
    
    logger.info('Capturing initial embeddings...')
    for batch in dataloader:
        try:
            input_ids = batch[0].to(dev)
            model(input_ids)
        except ValueError:
            pass

    layers[0] = layers[0].module

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    std_stream_state = err_stream_state.clone()

    logger.info('Data capture completed.')

    # 加载剪枝mask（你的代码）
    p = "./flap_mask"
    if args.sparsity_ratio == 0.2:
        p = "./flap_mask/0.2"
    elif args.sparsity_ratio == 0.7:
        p = "./flap_mask/0.7"
    attn_mask, mlp_mask, attn_baseline_inp_list, mlp_baseline_inp_list = load_flap_masks_and_baseline(p, dev)
    logger.info("Loaded pruning masks and baseline inputs.")

    # === 阶段1：逐层训练 ===
    logger.info('\n' + '='*80)
    logger.info('STAGE 1: Layer-wise Compensation Training')
    logger.info('='*80 + '\n')

    stage1_losses = []  # 记录每层的最终损失



    # =======================================================
    # 🆕 1. 断点目录设置
    # =======================================================
    layer_ckpt_dir = os.path.join(args.checkpoint_dir, "layer_checkpoints")
    os.makedirs(layer_ckpt_dir, exist_ok=True)
    
    start_layer_idx = 0
    
    # =======================================================
    # 🆕 2. 尝试加载最新的断点 (Resume Logic)
    # =======================================================
    # 查找是否有类似 layer_2.pt 的文件
    existing_ckpts = glob.glob(os.path.join(layer_ckpt_dir, "layer_*.pt"))
    
    if existing_ckpts and getattr(args, 'resume', True): # 默认开启 resume
        # 找到 layer_idx 最大的文件
        latest_ckpt_path = max(existing_ckpts, key=os.path.getctime)
        logger.info(f"🔄 Found checkpoint: {latest_ckpt_path}. Loading...")
        
        try:
            # 加载 checkpoint 到 CPU 以节省显存
            checkpoint = torch.load(latest_ckpt_path, map_location='cpu')

            # 3. 恢复进度
            # 既然保存的是“已完成 Layer N”，那么下次开始应该是 N + 1
            start_layer_idx = checkpoint['finished_layer_idx'] + 1
            #对之前的层进行剪枝，但是可以跳过训练
            for idx in range(start_layer_idx):
                layer = layers[idx].to(dev)
                layer, a, m = flap_compress_layer(
                layer, 
                attn_mask[idx], 
                mlp_mask[idx],
                attn_baseline_inp_list[idx],
                mlp_baseline_inp_list[idx],
                dev,
                unstr=False,
                pruned_weights={},
                bias=args.is_bias
            )
            
            # 1. 恢复模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 2. 恢复隐藏状态 (移动到对应的 dev)
            # 注意：如果 checkpoint 保存的是 CPU tensor，这里需要 .to(dev)
            err_stream_state = checkpoint['err_stream_state'].to(dev)
            std_stream_state = checkpoint['std_stream_state'].to(dev)

            
            logger.info(f"✅ Successfully resumed from Layer {start_layer_idx}!")
            
            # 显存清理
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            logger.warning("Starting from Layer 0.")
            start_layer_idx = 0
    else:
        logger.info("Starting training from scratch (Layer 0).")

    # 如果所有层都完成了，直接返回
    if start_layer_idx >= len(layers):
        logger.info("All layers already processed. Skipping Stage 1.")
        return model


    for idx in range(start_layer_idx,len(layers)):
        layer = layers[idx].to(dev)
        
        # Step 1: 获取未剪枝的输出
        logger.info(f'\n--- Layer {idx}: Computing unpruned outputs ---')
        dataset_std = FeatureDataset(std_stream_state, device=dev)
        dataloader_std = DataLoader(dataset_std, batch_size=1, shuffle=False, drop_last=False)
        
        layer.eval()
        
        if args.is_train:
            with torch.no_grad():
                for idxz, batch in dataloader_std:
                    outputs = layer(
                        batch,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False
                    )
                    std_stream_state[idxz] = outputs[0].half()
        
        # Step 2: 剪枝
        logger.info(f'--- Layer {idx}: Pruning ---')
        layer, a, m = flap_compress_layer(
            layer, 
            attn_mask[idx], 
            mlp_mask[idx],
            attn_baseline_inp_list[idx],
            mlp_baseline_inp_list[idx],
            dev,
            unstr=False,
            pruned_weights={},
            bias=args.is_bias
        )
        
        # Step 4: 逐层补偿训练（改进版）
        # config.stage1_epochs = 0
        if args.is_train and config.stage1_epochs > 0 :
            print(f"the epoch is {config.stage1_epochs}")
            best_loss, history = layer_compensation_training_v2(
                layer_idx=idx,
                adapted_layer=layer,
                pruned_hidden_states=err_stream_state.detach(),
                unpruned_hidden_states=std_stream_state.detach(),
                attention_mask=attention_mask,
                position_ids=position_ids,
                config=config,
                logger=logger,
                device=dev
            )
            stage1_losses.append(best_loss)
        else:
            stage1_losses.append(0.0)
        
        # Step 5: 获取补偿后的输出
        logger.info(f'--- Layer {idx}: Computing compensated outputs ---')
        dataset_err = FeatureDataset(err_stream_state, device=dev)
        dataloader_err = DataLoader(dataset_err, batch_size=1, shuffle=False, drop_last=False)
        
        if args.is_train:
            layer.eval()
            with torch.no_grad():
                for idxz, batch in dataloader_err:
                    outputs = layer(
                        batch,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False
                    )
                    err_stream_state[idxz] = outputs[0].half()
        
        # 更新模型
        layer = layer.to("cpu")
        model.model.layers[idx] = layer
        del layer
        
        torch.cuda.empty_cache()
        logger.info(f'Layer {idx} processing completed.\n')
        # =======================================================
        # 🆕 3. 保存当前层断点 (Save Logic)
        # =======================================================
        # 仅主进程保存 (如果是多卡 DDP)

        # 这里的 save_checkpoint 参数控制是否保存
        save_interval = 10
        #  save_interval 控制保存间隙
        if getattr(args, 'save_layer_ckpt', True) and idx % args.save_intervals == 0 and idx > 0 or idx == (len(layers)-1): 
            ckpt_path = os.path.join(layer_ckpt_dir, f"layer_{idx}.pt")
            
            logger.info(f"💾 Saving checkpoint for Layer {idx}...")
            
            # 建议先保存到临时文件再重命名，防止保存一半中断导致文件损坏
            tmp_path = ckpt_path + ".tmp"
            
            # 构建保存字典
            save_dict = {
                'finished_layer_idx': idx,
                'model_state_dict': model.state_dict(), # 保存整个模型权重
                'err_stream_state': err_stream_state.cpu(), # 移回 CPU 保存节省显存
                'std_stream_state': std_stream_state.cpu(),
            }
            
            torch.save(save_dict, tmp_path)
            os.rename(tmp_path, ckpt_path) # 原子操作
            
            # 4. 删除旧的断点以节省磁盘空间 (可选)
            # 我们只保留最近的一个断点
            if idx > 0:
                prev_ckpt = os.path.join(layer_ckpt_dir, f"layer_{idx-args.save_intervals}.pt")
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
                    
            logger.info(f"✅ Checkpoint saved: {ckpt_path}")

    # # === 阶段2：端到端训练 ===
    # if config.enable_stage2 and args.is_train:
    #     llama_stage2_compensation(model, logger, args)


    return model


##################################################
#
#  用于对比的特殊lora版本
#
##################################################

def llama_llmpruner_stage1_compensation(
    model, 
    dataloader, 
    tokenizer,
    dev, 
    logger, 
    args
):
    """
    基于LLM-Pruner的两阶段剪枝补偿训练
    
    阶段1：逐层剪枝与补偿（Layer-wise Pruning and Repair）
    阶段2：端到端知识蒸馏（End-to-End Logits Distillation）
    """
    
    logger.info('='*80)
    logger.info('Two-Stage Training with LLM-Pruner')
    logger.info('='*80)
    
    # ========== 配置设置 ==========
    config = CompensationConfig(
        # 阶段1
        stage1_epochs=getattr(args, 'tune_epoch', 0),
        stage1_lr=getattr(args, 'tune_lr', 1e-5),
        
        # 阶段2
        enable_stage2=getattr(args, 'enable_stage2', True),
        stage2_epochs=getattr(args, 'stage2_epochs', 50),
        stage2_lr=getattr(args, 'stage2_lr', 1e-5),
        
        # 损失权重
        alpha_hidden=getattr(args, 'alpha_hidden', 0.4),
        beta_logits=getattr(args, 'beta_logits', 0.5),
        gamma_hard=getattr(args, 'gamma_hard', 0.1),
        
        # 其他
        layer_weight_strategy=getattr(args, 'layer_weight_strategy', 'uniform'),
        kl_temperature=getattr(args, 'kl_temperature', 2.0),
        use_amp=getattr(args, 'use_amp', True),
        cache_hidden_states=getattr(args, 'cache_hiddens', True),
        save_checkpoint=getattr(args, 'save_checkpoint', True),
        checkpoint_dir=getattr(args, 'checkpoint_dir', './checkpoints'),
    )
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建输出目录
    import os
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # ========== 阶段1：逐层剪枝与补偿 ==========
    logger.info('\n' + '='*80)
    logger.info('STAGE 1: Layer-wise Pruning and Compensation')
    logger.info('='*80 + '\n')
    
    # 基本设置
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # 移动必要组件到设备
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)  # 确保lm_head也在GPU上
    
    for layer in layers:
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(dev)
    
    # 初始化数据流状态
    dtype = next(iter(model.parameters())).dtype
    err_stream_state = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16,
        device=dev
    )
    
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
    # 数据捕获器
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            err_stream_state[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    # 捕获初始embeddings
    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])
    
    logger.info('Capturing initial embeddings...')
    for batch in dataloader:
        try:
            input_ids = batch[0].to(dev)
            model(input_ids)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    std_stream_state = err_stream_state.clone()
    
    logger.info('Data capture completed.')
    
    # 执行LLM-Pruner剪枝
    logger.info('Performing LLM-Pruner pruning...')
    original_model = copy.deepcopy(model).to(dev)
    pruned_model = llmpruner_gpu(model, tokenizer=tokenizer, logger=logger, args=args).to("cpu")
    logger.info('LLM-Pruner pruning completed.')
    
    original_layers = original_model.model.layers
    pruned_layers = pruned_model.model.layers
    

    stage1_losses = []

    
    # =======================================================
    # 🆕 1. 断点目录设置
    # =======================================================
    layer_ckpt_dir = os.path.join(args.checkpoint_dir, "layer_checkpoints")
    os.makedirs(layer_ckpt_dir, exist_ok=True)
    
    start_layer_idx = 0
    
    # =======================================================
    # 🆕 2. 尝试加载最新的断点 (Resume Logic)
    # =======================================================
    # 查找是否有类似 layer_2.pt 的文件
    existing_ckpts = glob.glob(os.path.join(layer_ckpt_dir, "layer_*.pt"))
    
    if existing_ckpts and getattr(args, 'resume', True): # 默认开启 resume
        # 找到 layer_idx 最大的文件
        latest_ckpt_path = max(existing_ckpts, key=os.path.getctime)
        logger.info(f"🔄 Found checkpoint: {latest_ckpt_path}. Loading...")
        
        try:
            # 加载 checkpoint 到 CPU 以节省显存
            checkpoint = torch.load(latest_ckpt_path, map_location='cpu')
            
            # 1. 恢复模型权重
            pruned_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 2. 恢复隐藏状态 (移动到对应的 dev)
            # 注意：如果 checkpoint 保存的是 CPU tensor，这里需要 .to(dev)
            err_stream_state = checkpoint['err_stream_state'].to(dev)
            std_stream_state = checkpoint['std_stream_state'].to(dev)

            
            # 3. 恢复进度
            # 既然保存的是“已完成 Layer N”，那么下次开始应该是 N + 1
            start_layer_idx = checkpoint['finished_layer_idx'] + 1
            
            logger.info(f"✅ Successfully resumed from Layer {start_layer_idx}!")
            
            # 显存清理
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            logger.warning("Starting from Layer 0.")
            start_layer_idx = 0
    else:
        logger.info("Starting training from scratch (Layer 0).")

    # 如果所有层都完成了，直接返回
    if start_layer_idx >= len(layers):
        logger.info("All layers already processed. Skipping Stage 1.")
        return pruned_model
    
    # 逐层处理
    for layer_idx in range(start_layer_idx,len(layers)):
        logger.info(f'\n{"="*60}')
        logger.info(f'Processing Layer {layer_idx}/{len(layers)-1}')
        logger.info(f'{"="*60}')
        
        # Step 1: 获取未剪枝的输出（作为训练目标）
        logger.info(f'[Step 1] Computing unpruned outputs for layer {layer_idx}...')
        ori_layer = original_layers[layer_idx].to(dev)
        ori_layer.eval()
        
        dataset_std = FeatureDataset(std_stream_state, device=dev)
        dataloader_std = DataLoader(dataset_std, batch_size=1, shuffle=False, drop_last=False)
        
        if args.is_train:
            with torch.no_grad():
                for idx, batch in dataloader_std:
                    outputs = ori_layer(
                        batch,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False
                    )
                    std_stream_state[idx] = outputs[0].half()
        
        # 释放原始层
        original_layers[layer_idx] = ori_layer.cpu()
        del ori_layer
        torch.cuda.empty_cache()
        
        # Step 2: 获取剪枝后的层
        logger.info(f'[Step 2] Loading pruned layer {layer_idx}...')
        layer = pruned_layers[layer_idx].to(dev)
        
        
        # Step 4: 逐层补偿训练
        if args.is_train and config.stage1_epochs > 0:
            logger.info(f'[Step 4] Training compensation for layer {layer_idx}...')
            best_loss, history = layer_compensation_training_v2(
                layer_idx=layer_idx,
                adapted_layer=layer,
                pruned_hidden_states=err_stream_state.detach(),
                unpruned_hidden_states=std_stream_state.detach(),
                attention_mask=attention_mask,
                position_ids=position_ids,
                config=config,
                logger=logger,
                device=dev
            )
            stage1_losses.append(best_loss)
            logger.info(f'Layer {layer_idx} training completed. Best loss: {best_loss:.6f}')
        else:
            stage1_losses.append(0.0)
            logger.info(f'[Step 4] Skipped training for layer {layer_idx}')
        
        # Step 5: 获取补偿后的输出
        logger.info(f'[Step 5] Computing compensated outputs for layer {layer_idx}...')
        layer.eval()
        
        dataset_err = FeatureDataset(err_stream_state, device=dev)
        dataloader_err = DataLoader(dataset_err, batch_size=1, shuffle=False, drop_last=False)
        
        if args.is_train:
            with torch.no_grad():
                for idxz, batch in dataloader_err:
                    outputs = layer(
                        batch,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False
                    )
                    err_stream_state[idxz] = outputs[0].half()
        
        # 更新剪枝模型（保持在GPU上）
        pruned_layers[layer_idx] = layer.cpu()
        
        torch.cuda.empty_cache()
        logger.info(f'Layer {layer_idx} processing completed.\n')
    
    
    # 清理阶段1资源
    del err_stream_state, std_stream_state, original_model,layer
    torch.cuda.empty_cache()
    
    # =======================================================
    # 🆕 3. 保存当前层断点 (Save Logic)
    # =======================================================
    # 仅主进程保存 (如果是多卡 DDP)
    # 这里的 save_checkpoint 参数控制是否保存
    if getattr(args, 'save_layer_ckpt', True) and layer_idx % args.save_intervals == 0 and layer_idx > 0 or layer_idx == (len(layers) - 1):
        ckpt_path = os.path.join(layer_ckpt_dir, f"layer_{layer_idx}.pt")
        logger.info(f"Saving checkpoint for Layer {layer_idx}...")
        tmp_path = ckpt_path + ".tmp"
        save_dict = {
            'finished_layer_idx': layer_idx,
            'model_state_dict': pruned_model.state_dict(),
            'err_stream_state': err_stream_state.cpu(),
            'std_stream_state': std_stream_state.cpu(),
        }
        
        torch.save(save_dict, tmp_path)
        os.rename(tmp_path, ckpt_path) # 原子操作
        
        # 4. 删除旧的断点以节省磁盘空间 (可选)
        # 我们只保留最近的一个断点
        if idx > 0:
            prev_ckpt = os.path.join(layer_ckpt_dir, f"layer_{idx-1}.pt")
            if os.path.exists(prev_ckpt):
                os.remove(prev_ckpt)
                
        logger.info(f"✅ Checkpoint saved: {ckpt_path}")
    logger.info(f'Stage 1 cleanup completed. GPU memory: {torch.cuda.memory_allocated(dev)/1e9:.2f}GB\n')
    
    return pruned_model


#在外面处理剪枝模型
def llama_wanda_stage1_compensation(
    model, 
    dataloader, 
    dev, 
    logger, 
    args
):
    """
    两阶段补偿训练的主函数
    
    阶段1：逐层独立训练（Layer-wise Alignment）
    阶段2：端到端联合训练（End-to-End Alignment）
    """
    
    logger.info('='*80)
    logger.info('Starting Two-Stage Compensation Training')
    logger.info('='*80)
    
    # 创建配置对象
    config = CompensationConfig(
        # 阶段1
        stage1_epochs=getattr(args, 'tune_epoch', 10),
        stage1_lr=getattr(args, 'tune_lr', 1e-5),
        
        # 阶段2
        enable_stage2=getattr(args, 'enable_stage2', True),
        stage2_epochs=getattr(args, 'stage2_epochs', 10),
        stage2_lr=getattr(args, 'stage2_lr', 1e-5),
        
        # 损失权重
        alpha_hidden=getattr(args, 'alpha_hidden', 0.4),
        beta_logits=getattr(args, 'beta_logits', 0.5),
        gamma_hard=getattr(args, 'gamma_hard', 0.1),
        
        # 其他
        layer_weight_strategy=getattr(args, 'layer_weight_strategy', 'uniform'),
        kl_temperature=getattr(args, 'kl_temperature', 2.0),
        use_amp=getattr(args, 'use_amp', True),
        cache_hidden_states=getattr(args, 'cache_hiddens', True),
        save_checkpoint=getattr(args, 'save_checkpoint', True),
        checkpoint_dir=getattr(args, 'checkpoint_dir', './checkpoints'),
    )
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建checkpoint目录
    import os
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # 移动组件到设备
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.norm = model.model.norm.to(dev)
    # model.lm_head = model.lm_head.to(dev)  # ← 添加这一行！
    for layer in layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(dev)
    
    # 初始化数据流状态
    dtype = next(iter(model.parameters())).dtype
    err_stream_state = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16,
        device=dev
    )
    
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
    # 数据捕获（你的代码）
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            err_stream_state[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])
    
    logger.info('Capturing initial embeddings...')
    for batch in dataloader:
        try:
            input_ids = batch[0].to(dev)
            model(input_ids)
        except ValueError:
            pass

    layers[0] = layers[0].module

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    std_stream_state = err_stream_state.clone()

    logger.info('Data capture completed.')

    # wanda提前准备

    # wanda原始代码传入c4作为基准数据，而我们之前测量以wiki为准，因此遵守wanda的测试基准
    print("loading calibdation data")
    dataloader, _,_= get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,model = args.model)
    print("dataset loading complete")

    # with torch.no_grad():
    #     inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)


    # === 阶段1：逐层训练 ===
    logger.info('\n' + '='*80)
    logger.info('STAGE 1: Layer-wise Compensation Training')
    logger.info('='*80 + '\n')

    # 第一层embedding
    inps,outs = std_stream_state,err_stream_state

    stage1_losses = []  # 记录每层的最终损失


    # =======================================================
    # 🆕 1. 断点目录设置
    # =======================================================
    layer_ckpt_dir = os.path.join(args.checkpoint_dir, "layer_checkpoints")
    os.makedirs(layer_ckpt_dir, exist_ok=True)
    
    start_layer_idx = 0
    
    # =======================================================
    # 🆕 2. 尝试加载最新的断点 (Resume Logic)
    # =======================================================
    # 查找是否有类似 layer_2.pt 的文件
    existing_ckpts = glob.glob(os.path.join(layer_ckpt_dir, "layer_*.pt"))
    
    if existing_ckpts and getattr(args, 'resume', True): # 默认开启 resume
        # 找到 layer_idx 最大的文件
        latest_ckpt_path = max(existing_ckpts, key=os.path.getctime)
        logger.info(f"🔄 Found checkpoint: {latest_ckpt_path}. Loading...")
        
        try:
            # 加载 checkpoint 到 CPU 以节省显存
            checkpoint = torch.load(latest_ckpt_path, map_location='cpu')
            
            # 1. 恢复模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 2. 恢复隐藏状态 (移动到对应的 dev)
            # 注意：如果 checkpoint 保存的是 CPU tensor，这里需要 .to(dev)
            err_stream_state = checkpoint['err_stream_state'].to(dev)
            std_stream_state = checkpoint['std_stream_state'].to(dev)

            inps = checkpoint['inps']
            outs = checkpoint['outs']
            
            # 3. 恢复进度
            # 既然保存的是“已完成 Layer N”，那么下次开始应该是 N + 1
            start_layer_idx = checkpoint['finished_layer_idx'] + 1
            
            logger.info(f"✅ Successfully resumed from Layer {start_layer_idx}!")
            
            # 显存清理
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            logger.warning("Starting from Layer 0.")
            start_layer_idx = 0
    else:
        logger.info("Starting training from scratch (Layer 0).")

    # 如果所有层都完成了，直接返回
    if start_layer_idx >= len(layers):
        logger.info("All layers already processed. Skipping Stage 1.")
        return model

    for idx in range(start_layer_idx,len(layers)):
        layer = layers[idx].to(dev)
        # subset = find_layers(layer)
        
        # Step 1: 获取未剪枝的输出
        logger.info(f'\n--- Layer {idx}: Computing unpruned outputs ---')
        dataset_std = FeatureDataset(std_stream_state, device=dev)
        dataloader_std = DataLoader(dataset_std, batch_size=1, shuffle=False, drop_last=False)
        
        layer.eval()
        
        if args.is_train:
            with torch.no_grad():
                for idxz, batch in dataloader_std:
                    outputs = layer(
                        batch,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False
                    )
                    std_stream_state[idxz] = outputs[0].half()
        
        # Step 2: 剪枝
        logger.info(f'--- Layer {idx}: Wanda Pruning ---')

        layer, inps, outs, sparsity_constraint = compress_wanda_with_constraint(
            layer, dev, args, inps, outs, attention_mask, position_ids
        )
        inps = inps.to("cpu")
        outs = outs.to("cpu")
        # 释放 compress_wanda 中间计算产生的 GPU 显存
        torch.cuda.empty_cache()

        # Step 4: 逐层补偿训练（改进版）
        # config.stage1_epochs = 0
        if args.is_train and config.stage1_epochs > 0:
            print(f"the epoch is {config.stage1_epochs}")
            best_loss, history = layer_compensation_training_sparse(
                layer_idx=idx,
                adapted_layer=layer,
                pruned_hidden_states=err_stream_state.detach(),
                unpruned_hidden_states=std_stream_state.detach(),
                attention_mask=attention_mask,
                position_ids=position_ids,
                sparsity_constraint=sparsity_constraint,  # 传入约束对象
                config=config,
                logger=logger,
                device=dev
            )
            stage1_losses.append(best_loss)
        else:
            stage1_losses.append(0.0)
        
        # Step 5: 获取补偿后的输出
        logger.info(f'--- Layer {idx}: Computing compensated outputs ---')
        dataset_err = FeatureDataset(err_stream_state, device=dev)
        dataloader_err = DataLoader(dataset_err, batch_size=1, shuffle=False, drop_last=False)
        
        if args.is_train:
            layer.eval()
            with torch.no_grad():
                for idxz, batch in dataloader_err:
                    outputs = layer(
                        batch,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False
                    )
                    err_stream_state[idxz] = outputs[0].half()
        
        # 更新模型
        model.model.layers[idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        logger.info(f'Layer {idx} processing completed.\n')

         # =======================================================
        # 🆕 3. 保存当前层断点 (Save Logic)
        # =======================================================
        # 仅主进程保存 (如果是多卡 DDP)
        # 这里的 save_checkpoint 参数控制是否保存
        if getattr(args, 'save_layer_ckpt', True) and idx % args.save_intervals == 0  and idx>0 or idx == (len(layers)-1): 
            ckpt_path = os.path.join(layer_ckpt_dir, f"layer_{idx}.pt")
            
            logger.info(f"💾 Saving checkpoint for Layer {idx}...")
            
            # 建议先保存到临时文件再重命名，防止保存一半中断导致文件损坏
            tmp_path = ckpt_path + ".tmp"
            
            # 构建保存字典
            save_dict = {
                'finished_layer_idx': idx,
                'model_state_dict': model.state_dict(), # 保存整个模型权重
                'err_stream_state': err_stream_state.cpu(), # 移回 CPU 保存节省显存
                'std_stream_state': std_stream_state.cpu(),
                'inps':inps.cpu(),
                'outs':outs.cpu()
            }
            
            torch.save(save_dict, tmp_path)
            os.rename(tmp_path, ckpt_path) # 原子操作
            
            # 4. 删除旧的断点以节省磁盘空间 (可选)
            # 我们只保留最近的一个断点
            if idx > 0:
                prev_ckpt = os.path.join(layer_ckpt_dir, f"layer_{idx-1}.pt")
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
                    
            logger.info(f"✅ Checkpoint saved: {ckpt_path}")

    # # === 阶段2：端到端训练 ===
    # if config.enable_stage2 and args.is_train:
    #     llama_stage2_compensation(model, logger, args)

    del std_stream_state, err_stream_state, inps, outs

    return model



######################################
#    基于wanda-plus版本的改写




def llama_sparsegpt_stage1_compensation(
    model, 
    dataloader, 
    dev, 
    logger, 
    args
):
    """
    两阶段补偿训练的主函数
    
    阶段1：逐层独立训练（Layer-wise Alignment）
    阶段2：端到端联合训练（End-to-End Alignment）
    """
    
    logger.info('='*80)
    logger.info('Starting Two-Stage Compensation Training')
    logger.info('='*80)
    
    # 创建配置对象
    config = CompensationConfig(
        # 阶段1
        stage1_epochs=getattr(args, 'tune_epoch', 0),
        stage1_lr=getattr(args, 'tune_lr', 1e-5),
        
        # 阶段2
        enable_stage2=getattr(args, 'enable_stage2', True),
        stage2_epochs=getattr(args, 'stage2_epochs', 50),
        stage2_lr=getattr(args, 'stage2_lr', 1e-5),
        
        # 损失权重
        alpha_hidden=getattr(args, 'alpha_hidden', 0.4),
        beta_logits=getattr(args, 'beta_logits', 0.5),
        gamma_hard=getattr(args, 'gamma_hard', 0.1),
        
        # 其他
        layer_weight_strategy=getattr(args, 'layer_weight_strategy', 'uniform'),
        kl_temperature=getattr(args, 'kl_temperature', 2.0),
        use_amp=getattr(args, 'use_amp', True),
        cache_hidden_states=getattr(args, 'cache_hiddens', True),
        save_checkpoint=getattr(args, 'save_checkpoint', False),
        checkpoint_dir=getattr(args, 'checkpoint_dir', './checkpoints'),
    )
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建checkpoint目录
    import os
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # 移动组件到设备
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.norm = model.model.norm.to(dev)
    # model.lm_head = model.lm_head.to(dev)  # ← 添加这一行！
    for layer in layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(dev)
    
    # 初始化数据流状态
    dtype = next(iter(model.parameters())).dtype
    err_stream_state = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16,
        device=dev
    )
    
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
    # 数据捕获（你的代码）
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            err_stream_state[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])
    
    logger.info('Capturing initial embeddings...')
    for batch in dataloader:
        try:
            input_ids = batch[0].to(dev)
            model(input_ids)
        except ValueError:
            pass

    layers[0] = layers[0].module

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    std_stream_state = err_stream_state.clone()

    logger.info('Data capture completed.')

    # wanda提前准备

    # wanda原始代码传入c4作为基准数据，而我们之前测量以wiki为准，因此遵守wanda的测试基准
    print("loading calibdation data")
    dataloader, _,_= get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,model = args.model)
    print("dataset loading complete")

    # with torch.no_grad():
    #     inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)


    # === 阶段1：逐层训练 ===
    logger.info('\n' + '='*80)
    logger.info('STAGE 1: Layer-wise Compensation Training')
    logger.info('='*80 + '\n')

    # 第一层embedding
    inps,outs = std_stream_state,err_stream_state

    stage1_losses = []  # 记录每层的最终损失



    # =======================================================
    # 🆕 1. 断点目录设置
    # =======================================================
    layer_ckpt_dir = os.path.join(args.checkpoint_dir, "layer_checkpoints")
    os.makedirs(layer_ckpt_dir, exist_ok=True)
    
    start_layer_idx = 0
    
    # =======================================================
    # 🆕 2. 尝试加载最新的断点 (Resume Logic)
    # =======================================================
    # 查找是否有类似 layer_2.pt 的文件
    existing_ckpts = glob.glob(os.path.join(layer_ckpt_dir, "layer_*.pt"))
    
    if existing_ckpts and getattr(args, 'resume', True): # 默认开启 resume
        # 找到 layer_idx 最大的文件
        latest_ckpt_path = max(existing_ckpts, key=os.path.getctime)
        logger.info(f"🔄 Found checkpoint: {latest_ckpt_path}. Loading...")
        
        try:
            # 加载 checkpoint 到 CPU 以节省显存
            checkpoint = torch.load(latest_ckpt_path, map_location='cpu')
            
            # 1. 恢复模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 2. 恢复隐藏状态 (移动到对应的 dev)
            # 注意：如果 checkpoint 保存的是 CPU tensor，这里需要 .to(dev)
            err_stream_state = checkpoint['err_stream_state'].to(dev)
            std_stream_state = checkpoint['std_stream_state'].to(dev)

            inps = checkpoint['inps']
            outs = checkpoint['outs']
            
            # 3. 恢复进度
            # 既然保存的是“已完成 Layer N”，那么下次开始应该是 N + 1
            start_layer_idx = checkpoint['finished_layer_idx'] + 1
            
            logger.info(f"✅ Successfully resumed from Layer {start_layer_idx}!")
            
            # 显存清理
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            logger.warning("Starting from Layer 0.")
            start_layer_idx = 0
    else:
        logger.info("Starting training from scratch (Layer 0).")

    # 如果所有层都完成了，直接返回
    if start_layer_idx >= len(layers):
        logger.info("All layers already processed. Skipping Stage 1.")
        return model



    for idx in range(start_layer_idx, len(layers)):
        layer = layers[idx].to(dev)
        # subset = find_layers(layer)
        
        # Step 1: 获取未剪枝的输出
        logger.info(f'\n--- Layer {idx}: Computing unpruned outputs ---')
        dataset_std = FeatureDataset(std_stream_state, device=dev)
        dataloader_std = DataLoader(dataset_std, batch_size=1, shuffle=False, drop_last=False)
        
        layer.eval()
        
        if args.is_train:
            with torch.no_grad():
                for idxz, batch in dataloader_std:
                    outputs = layer(
                        batch,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False
                    )
                    std_stream_state[idxz] = outputs[0].half()
        
        # Step 2: 剪枝
        logger.info(f'--- Layer {idx}: sparsegpt Pruning ---')

        layer, inps, outs, sparsity_constraint = compress_sparsegpt_with_constraint(
            layer, dev, args, inps, outs, attention_mask, position_ids
        )
        inps = inps.to("cpu")
        outs = outs.to("cpu")
        # 释放 compress_wanda 中间计算产生的 GPU 显存
        torch.cuda.empty_cache()
        # Step 4: 逐层补偿训练（改进版）
        # config.stage1_epochs = 0
        if args.is_train and config.stage1_epochs > 0:
            print(f"the epoch is {config.stage1_epochs}")
            best_loss, history = layer_compensation_training_sparse(
                layer_idx=idx,
                adapted_layer=layer,
                pruned_hidden_states=err_stream_state.detach(),
                unpruned_hidden_states=std_stream_state.detach(),
                attention_mask=attention_mask,
                position_ids=position_ids,
                sparsity_constraint=sparsity_constraint,  # 传入约束对象
                config=config,
                logger=logger,
                device=dev
            )
            stage1_losses.append(best_loss)
        else:
            stage1_losses.append(0.0)
        
        # Step 5: 获取补偿后的输出
        logger.info(f'--- Layer {idx}: Computing compensated outputs ---')
        dataset_err = FeatureDataset(err_stream_state, device=dev)
        dataloader_err = DataLoader(dataset_err, batch_size=1, shuffle=False, drop_last=False)
        
        if args.is_train:
            layer.eval()
            with torch.no_grad():
                for idxz, batch in dataloader_err:
                    outputs = layer(
                        batch,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False
                    )
                    err_stream_state[idxz] = outputs[0].half()
        
        # 更新模型
        layer = layer.to("cpu")
        model.model.layers[idx] = layer
        del layer
        torch.cuda.empty_cache()
        logger.info(f'Layer {idx} processing completed.\n')
        # =======================================================
        # 🆕 3. 保存当前层断点 (Save Logic)
        # =======================================================
        # 仅主进程保存 (如果是多卡 DDP)
        # 这里的 save_checkpoint 参数控制是否保存
        if getattr(args, 'save_layer_ckpt', True) and idx % args.save_intervals == 0 and idx > 0 or idx == (len(layers)-1): 
            ckpt_path = os.path.join(layer_ckpt_dir, f"layer_{idx}.pt")
            
            logger.info(f"💾 Saving checkpoint for Layer {idx}...")
            
            # 建议先保存到临时文件再重命名，防止保存一半中断导致文件损坏
            tmp_path = ckpt_path + ".tmp"
            
            # 构建保存字典
            save_dict = {
                'finished_layer_idx': idx,
                'model_state_dict': model.state_dict(), # 保存整个模型权重
                'err_stream_state': err_stream_state.cpu(), # 移回 CPU 保存节省显存
                'std_stream_state': std_stream_state.cpu(),
                'inps':inps.cpu(),
                'outs':outs.cpu()
            }
            
            torch.save(save_dict, tmp_path)
            os.rename(tmp_path, ckpt_path) # 原子操作
            
            # 4. 删除旧的断点以节省磁盘空间 (可选)
            # 我们只保留最近的一个断点
            if idx > 0:
                prev_ckpt = os.path.join(layer_ckpt_dir, f"layer_{idx-1}.pt")
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
                    
            logger.info(f"✅ Checkpoint saved: {ckpt_path}")

        

    del std_stream_state, err_stream_state, inps, outs
    return model


def llama_stage2_compensation_cached(model,dataloader, logger, args,cache_dir):
    logger.info('\n' + '='*80)
    logger.info('STAGE 2: End-to-End Compensation Training')
    logger.info('='*80 + '\n')



    # 创建配置对象
    config = CompensationConfig(
        # 阶段1
        stage1_epochs=getattr(args, 'tune_epoch', 0),
        stage1_lr=getattr(args, 'tune_lr', 1e-5),
        
        # 阶段2
        enable_stage2=getattr(args, 'enable_stage2', True),
        stage2_epochs=getattr(args, 'stage2_epochs', 10),
        stage2_lr=getattr(args, 'stage2_lr', 1e-5),
        
        # 损失权重
        alpha_hidden=getattr(args, 'alpha_hidden', 0.4),
        beta_logits=getattr(args, 'beta_logits', 0.5),
        gamma_hard=getattr(args, 'gamma_hard', 0.1),
        
        # 其他
        layer_weight_strategy=getattr(args, 'layer_weight_strategy', 'uniform'),
        kl_temperature=getattr(args, 'kl_temperature', 2.0),
        use_amp=getattr(args, 'use_amp', True),
        cache_hidden_states=getattr(args, 'cache_hiddens', True),
        save_checkpoint=getattr(args, 'save_checkpoint', True),
        checkpoint_dir=getattr(args, 'checkpoint_dir', './checkpoints'),
    )
    dev = args.device

    # #两个模型暂时放到cpu上
    model = model.to("cpu")

    model, stage2_history = end_to_end_logits_distillation_cached(
        model=model,
        cache_dir = cache_dir,
        dataloader=dataloader,
        config=config,
        logger=logger,
        device = args.device
    )



    logger.info('\n' + '='*80)
    logger.info('Two-Stage Compensation Training Completed!')
    logger.info('='*80)

    return model


