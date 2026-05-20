import gc
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .features import DoubleFeatureDataset, ThreeFeatureDataset
from .pruning import *
from .quantization import *
from .utils import find_layers

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


@dataclass
class CompensationConfig:
    """Configuration for LaCo compensation training.

    Stage 1: Layer-wise hidden state alignment.
    Stage 2: End-to-end knowledge distillation.
    """
    stage1_epochs: int = 10
    stage1_lr: float = 1e-5
    stage1_batch_size: int = 1
    stage2_epochs: int = 20
    stage2_lr: float = 5e-5
    stage2_batch_size: int = 1
    enable_stage2: bool = True
    alpha_hidden: float = 0.4
    beta_logits: float = 0.5
    gamma_hard: float = 0.1
    layer_weight_strategy: str = "uniform"
    kl_temperature: float = 2.0
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    early_stop_patience: int = 10
    use_amp: bool = True
    log_interval: int = 100
    save_checkpoint: bool = True
    checkpoint_dir: str = "./checkpoints"
    cache_hidden_states: bool = True
    l2_reg: float = 0.0
    seed: int = 42
    use_8bit_optimizer: bool = False
    gradient_accumulation_steps: int = 1
    resume_stage2: bool = False


def compute_kl_divergence_loss(student_logits, teacher_logits, temperature=2.0, attention_mask=None):
    """
    计算KL散度损失（标准知识蒸馏）
    
    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        attention_mask: [batch, seq_len]
        temperature: 温度系数，控制分布平滑度
    
    Returns:
        loss: KL散度损失
    """
    # 应用temperature softmax
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # 计算KL散度: KL(P||Q) = sum(P * log(P/Q))
    kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='none',
        log_target=False
    )
    
    # 对vocab维度求和
    kl_loss = kl_loss.sum(dim=-1)  # [batch, seq_len]
    
    # # 应用attention mask（忽略padding位置）
    # if attention_mask is not None:
    #     mask = attention_mask.float()
    #     kl_loss = (kl_loss * mask).sum() / mask.sum()
    # else:
    #     kl_loss = kl_loss.mean()
    
    # Temperature平方缩放（标准做法，保持梯度尺度）
    kl_loss = kl_loss * (temperature ** 2)
    
    return kl_loss


##################################################
#
#    拆分版本，将logits和具体训练函数进行拆分
#
##################################################
import os
import torch
import tempfile
import shutil
import gc

def precompute_teacher_logits(
    teacher_model,
    dataloader,
    logger,
    cache_prefix,
    device,
    cache_base_dir="./cache"
):
    """
    预计算并缓存 teacher logits。
    使用 cache_prefix 创建稳定的缓存目录，并检查是否存在以实现重用。
    返回缓存目录路径。
    """
    logger.info('\n[Step 1] Pre-computing teacher logits...')
    
    # =======================================================
    # 🆕 1. 创建稳定的缓存目录路径
    # =======================================================
    # 使用 hashlib 创建一个稳定的、唯一的目录名，以避免路径过长且保证唯一性。
    # 路径通常应该基于模型名称、数据集名称和序列长度等，但这里我们只用 prefix。
    # 假设 cache_prefix 已经包含了足够的信息（例如：llama-7b_wikitext2_1024）
    
    # 确保前缀是安全的文件名
    safe_prefix = cache_prefix.replace('/', '_').replace('-', '_')
    
    # 完整的缓存目录路径
    cache_dir = os.path.join(cache_base_dir, f'logits_cache_{safe_prefix}')
    
    # =======================================================
    # 🆕 2. 检查缓存是否存在 (存在则直接返回)
    # =======================================================
    # 我们假设如果目录存在，并且其中有至少一个缓存文件（例如 batch_0.pt），则缓存完整。
    if os.path.exists(cache_dir):
        # 简单检查 batch_0.pt 是否存在
        test_file_path = os.path.join(cache_dir, 'batch_100.pt')
        if os.path.exists(test_file_path):
            logger.info(f'✅ Found existing cache at: {cache_dir}. Skipping pre-computation.')
            return cache_dir
        else:
            logger.warning(f'⚠ Cache directory {cache_dir} found, but test file {test_file_path} is missing. Recomputing...')
            # 如果目录不完整，尝试删除并重新计算
            shutil.rmtree(cache_dir, ignore_errors=True)
            
    # =======================================================
    # 3. 缓存不存在或不完整，开始计算
    # =======================================================
    logger.info(f'Cache not found. Starting pre-computation to: {cache_dir}')
    os.makedirs(cache_dir, exist_ok=False) # 确保目录不存在，如果存在则会报错
    
    # 准备 Teacher 模型
    teacher_model.eval()
    teacher_model = teacher_model.half()  # fp16
    teacher_model.to(device)
    
    # 创建缓存目录
    os.makedirs(cache_base_dir, exist_ok=True)
    # cache_dir = tempfile.mkdtemp(dir=cache_base_dir, prefix='logits_cache_')
    logger.info(f'Cache directory created at: {cache_dir}')
    
    num_cached = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 将输入移到 GPU
            input_ids = batch[0].to(device)
            batch_size, seq_len = input_ids.shape
            
            # 构建 mask 和 pos_ids
            batch_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            batch_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # 前向传播 (仅需 logits)
            outputs = teacher_model(
                input_ids,
                attention_mask=batch_attention_mask,
                position_ids=batch_position_ids,
                use_cache=False
            )
            
            # 保存数据 (fp16 on CPU)
            cache_data = {
                'logits': outputs.logits.cpu().half(),
                'attention_mask': batch_attention_mask.cpu()
            }
            
            cache_path = os.path.join(cache_dir, f'batch_{batch_idx}.pt')
            torch.save(cache_data, cache_path)
            
            num_cached += 1
            if batch_idx % 50 == 0:
                logger.info(f'  Cached {batch_idx + 1}/{len(dataloader)} batches')
            
            # 及时清理
            del input_ids, outputs, batch_attention_mask
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                
    logger.info(f'✓ Cached {num_cached} batches of teacher logits')
    
    # 彻底释放 Teacher
    del teacher_model
    torch.cuda.empty_cache()
    gc.collect()
    
    gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f'✓ Teacher model released. Current GPU memory: {gpu_mem:.2f} GB')
    
    return cache_dir

def end_to_end_logits_distillation_cached(
    model: nn.Module,
    cache_dir: str, 
    dataloader: torch.utils.data.DataLoader,
    config: 'CompensationConfig', # 假设 CompensationConfig 是可用的
    logger,
    device: torch.device
):
    """
    阶段2：使用预计算的 Logits 训练 Student，具备断点续训能力。
    
    主要修复：
    1. 引入断点续训 (Epoch 结束时保存一次)。
    2. 修复 Loss 聚合 (batch_loss 无法转换为标量的问题)。
    3. 修正梯度清零和学习率调度逻辑。
    4. 简化 Batch 循环恢复逻辑。
    """
    
    if not config.enable_stage2:
        logger.info('Stage 2 is disabled, skipping...')
        return model, []

    logger.info('='*80)
    logger.info('STAGE 2: Logits-Only Knowledge Distillation (Cached)')
    logger.info('='*80)
    
    logger.info('\n[Step 1] Preparing student model for training...')
    
    # 确保 Student 在 GPU
    model = model.to(device).float()
    model.train()

    # 梯度检查点
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info('✓ Gradient checkpointing enabled')
        
    # 优化器配置
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    use_8bit = getattr(config, 'use_8bit_optimizer', True)
    
    # --- 优化器定义 (保持原样) ---
    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params, lr=config.stage2_lr, weight_decay=config.weight_decay,
                betas=(config.adam_beta1, config.adam_beta2)
            )
            logger.info('✓ Using 8-bit AdamW optimizer')
        except ImportError:
            logger.warning('⚠ bitsandbytes not found, using standard AdamW')
            optimizer = torch.optim.AdamW(
                trainable_params, lr=config.stage2_lr, weight_decay=config.weight_decay,
                betas=(config.adam_beta1, config.adam_beta2)
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=config.stage2_lr, weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2)
        )
        
    # 调度器
    total_steps = config.stage2_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

    # =======================================================
    # 1. 断点和恢复逻辑 (Epoch 粒度)
    # =======================================================
    latest_ckpt_path = os.path.join(config.checkpoint_dir, 'stage2_latest.pt')
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(latest_ckpt_path) and getattr(config, 'resume_stage2', True):
        logger.info(f"🔄 Found checkpoint: {latest_ckpt_path}. Loading Stage 2 progress...")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location='cpu')
            
            # 恢复模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # 恢复优化器和调度器
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复进度
            start_epoch = checkpoint['epoch'] # 恢复的是下一轮的起始 epoch
            best_loss = checkpoint.get('best_loss', float('inf'))
            
            # 恢复 Scaler
            if scaler and 'scaler_state_dict' in checkpoint:
                 scaler.load_state_dict(checkpoint['scaler_state_dict'])
                 
            logger.info(f"✅ Resumed from Epoch {start_epoch}. Best Loss: {best_loss:.6f}")
        
        except Exception as e:
            logger.error(f"❌ Failed to resume Stage 2: {e}. Starting from scratch.")
            start_epoch = 0 # 恢复失败，重置起始 epoch
    print(f"Can not find. Now the path is {latest_ckpt_path}")
            
    # ========== 训练循环 ==========
    logger.info('\n[Step 2] Starting training...')
    training_history = []
    
    # 🚩 由于现在是 Epoch 粒度保存，Batch 循环总是从 0 开始
    for epoch in range(start_epoch, config.stage2_epochs):
        
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad() # 每个 Epoch 开始时清零梯度
        
        # ⚠️ 注意：如果dataloader是迭代器，它会在每次epoch循环开始时自动重置
        for batch_idx, batch in enumerate(dataloader):
            
            input_ids = batch[0].to(device)
            batch_size, seq_len = input_ids.shape
            
            # 加载 Cache
            cache_path = os.path.join(cache_dir, f'batch_{batch_idx}.pt')
            teacher_cache = torch.load(cache_path, map_location='cpu')
            teacher_logits = teacher_cache['logits'].to(device)
            
            # 构造辅助变量
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Student 前向
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                student_outputs = model(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                
                # 计算 Loss 
                loss_tensor = compute_kl_divergence_loss(
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_logits,
                    attention_mask=attention_mask,
                    temperature=config.kl_temperature
                )
                
                # 🔥 修复 Loss 聚合：确保 Loss 在反向传播前是标量，并处理梯度累积
                loss = loss_tensor.mean() 
                loss_scaled = loss / accumulation_steps
                
            # 反向传播
            if config.use_amp and scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            
            is_update_step = (batch_idx + 1) % accumulation_steps == 0
            
            # 梯度更新
            if is_update_step:
                if config.use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad() # 始终在 step() 之后清零
                scheduler.step()
                
            # 统计
            # 🔥 修复 batch_loss 统计：使用聚合后的 loss (已经是标量)
            # 使用 loss.item() 乘以 accumulation_steps 得到未平均的当前批次总 loss
            batch_loss = loss.item() * accumulation_steps
            epoch_loss += batch_loss
            num_batches += 1
            
            if batch_idx % config.log_interval == 0:
                gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
                logger.info(
                    f'Epoch {epoch+1}/{config.stage2_epochs}, Batch {batch_idx}: '
                    f'Loss={batch_loss:.6f}, LR={scheduler.get_last_lr()[0]:.2e}, GPU={gpu_mem:.2f}GB'
                )
                
            # 及时释放
            del teacher_logits, teacher_cache, student_outputs
            gc.collect() # 显式垃圾回收
        
        # Epoch 结束统计
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        training_history.append({'epoch': epoch, 'loss': avg_loss})
        logger.info(f'Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.6f}')
        
        # =======================================================
        # 🆕 3. Epoch 结束时保存断点
        # =======================================================
        if epoch % 3 == 0 and epoch != 0: 
            try:
                # 保存下一轮的起始状态：下一个 Epoch，Batch 从 0 开始
                ckpt_dict = {
                    'epoch': epoch + 1,          # 下次从下一个 Epoch 开始
                    'batch_idx': 0,              # 下次从 Batch 0 开始
                    'best_loss': best_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                if scaler:
                    ckpt_dict['scaler_state_dict'] = scaler.state_dict()
                    
                tmp_path = latest_ckpt_path + ".tmp"
                torch.save(ckpt_dict, tmp_path)
                os.rename(tmp_path, latest_ckpt_path)
                logger.info(f'💾 Saved epoch-end checkpoint for Epoch {epoch+1} to {latest_ckpt_path}')
                
            except Exception as e:
                logger.error(f"❌ Failed to save epoch-end checkpoint for Epoch {epoch+1}: {e}")

        # 最佳模型保存 (在断点保存之后执行)
        if avg_loss < best_loss:
            best_loss = avg_loss
            if config.save_checkpoint:
                ckpt_path = os.path.join(config.checkpoint_dir, 'stage2_best.pt')
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f'✓ Saved best model to {ckpt_path}')
                
    # ========== 清理 ==========
    # logger.info(f'\n[Cleanup] Removing latest checkpoint: {latest_ckpt_path}')
    # if os.path.exists(latest_ckpt_path):
    #     os.remove(latest_ckpt_path)

    # logger.info('\n[Cleanup] Removing cache directory...')
    # shutil.rmtree(cache_dir, ignore_errors=True)
    
    model.eval()
    return model, training_history




def layer_compensation_training_v2(
    layer_idx,
    adapted_layer,
    pruned_hidden_states,
    unpruned_hidden_states,
    attention_mask,
    position_ids,
    config: CompensationConfig,
    logger,
    device
):
    """
    全层训练函数
    
    Args:
        layer_idx: 层索引
        adapted_layer: 带适配器的层
        pruned_hidden_states: 剪枝后的hidden states
        unpruned_hidden_states: 未剪枝的hidden states (作为目标)
        attention_mask: 注意力掩码
        position_ids: 位置编码
        config: 补偿配置
        logger: 日志记录器
        device: 设备
    """
    adapted_layer = adapted_layer.float()
    # 设置训练模式
    params = list(adapted_layer.parameters())
    for param in params:
        param.requires_grad = True
    adapted_layer.train()
    
    
    # 优化器
    optimizer = optim.AdamW(
        params,
        lr=config.stage1_lr,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.stage1_epochs,
        eta_min=1e-7
    )
    
    # 混合精度
    scaler = GradScaler() if config.use_amp else None
    
    # 损失函数
    loss_module = CompensationLosses()
    
    # 数据加载器
    dataset = DoubleFeatureDataset(
        pruned_hidden_states, 
        unpruned_hidden_states, 
        device=device
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config.stage1_batch_size, 
        shuffle=True, 
        drop_last=False
    )
    
    logger.info(f'=== Stage 1: Layer {layer_idx} Compensation Training ===')
    logger.info(f'Trainable parameters: {sum(p.numel() for p in params):,}')
    logger.info(f'Training samples: {len(dataset)}')
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    for epoch in range(config.stage1_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_cosine = 0.0
        num_batches = 0
        
        for batch_idx, (idx, pruned_batch, unpruned_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                outputs = adapted_layer(
                    pruned_batch,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                    use_cache=False
                )
                
                compensated_hidden = outputs[0]
                
                # 计算多种损失
                loss_mse = loss_module.mse_loss(
                    compensated_hidden, 
                    unpruned_batch
                )
                loss_cosine = loss_module.cosine_loss(
                    compensated_hidden,
                    unpruned_batch
                )
                
                # 主损失：MSE（你可以根据需要调整）
                loss = loss_mse
                
                # 可选：L2正则化
                if config.l2_reg > 0:
                    l2_penalty = sum(torch.norm(p, 2)**2 for p in params)
                    loss += config.l2_reg * l2_penalty
            
            # 反向传播
            if config.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)
                optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_cosine += loss_cosine.item()
            num_batches += 1
            
            # 定期日志
            if batch_idx % config.log_interval == 0:
                logger.info(
                    f'Layer {layer_idx}, Epoch {epoch}, Batch {batch_idx}: '
                    f'Loss={loss.item():.6f}, MSE={loss_mse.item():.6f}, '
                    f'Cosine={loss_cosine.item():.6f}'
                )
        
        # Epoch统计
        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_cosine = epoch_cosine / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        training_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'mse': avg_mse,
            'cosine': avg_cosine,
            'lr': current_lr
        })
        
        logger.info(
            f'Layer {layer_idx}, Epoch {epoch} Summary: '
            f'Loss={avg_loss:.6f}, MSE={avg_mse:.6f}, '
            f'Cosine={avg_cosine:.6f}, LR={current_lr:.2e}'
        )
        
        # 学习率调度
        scheduler.step()
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # 保存最佳模型（可选）
            if config.save_checkpoint:
                checkpoint_path = f"{config.checkpoint_dir}/layer_{layer_idx}_best.pt"
                # torch.save({
                #     'layer_idx': layer_idx,
                #     'epoch': epoch,
                #     'state_dict': adapted_layer.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'loss': best_loss,
                # }, checkpoint_path)
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stop_patience:
            logger.info(f'Early stopping at epoch {epoch} for layer {layer_idx}')
            break
    
    # 训练完成
    adapted_layer.eval()
    for param in params:
        param.requires_grad = False
    
    logger.info(
        f'=== Layer {layer_idx} Stage 1 Completed ===\n'
        f'Final Loss: {best_loss:.6f}, Epochs: {epoch+1}/{config.stage1_epochs}'
    )
    
    return best_loss, training_history


class CompensationLosses:
    """Loss functions for compensation training."""

    @staticmethod
    def mse_loss(pred, target):
        """MSE loss."""
        return torch.nn.functional.mse_loss(pred, target)

    @staticmethod
    def cosine_loss(pred, target):
        """Cosine similarity loss."""
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        cosine_sim = torch.nn.functional.cosine_similarity(
            pred_flat, target_flat, dim=-1
        )
        return 1 - cosine_sim.mean()

    @staticmethod
    def l1_loss(pred, target):
        """L1 loss."""
        return torch.nn.functional.l1_loss(pred, target)

    def compute_layer_weights(self, num_layers, strategy='uniform', converged_losses=None):
        """Compute per-layer weights."""
        if strategy == 'uniform':
            return torch.ones(num_layers) / num_layers
        elif strategy == 'inverse_loss' and converged_losses is not None:
            losses = torch.tensor(converged_losses)
            weights = 1.0 / (losses + 1e-8)
            return weights / weights.sum()
        else:
            return torch.ones(num_layers) / num_layers


class SparsityConstraint:
    """
    管理和维护模型的稀疏性约束
    
    功能：
    1. 记录剪枝后的mask（哪些权重应该保持为0）
    2. 在训练过程中应用mask，防止0权重被更新
    3. 提供稀疏性验证和统计功能
    """
    
    def __init__(self, model_or_layer):
        """
        初始化稀疏性约束管理器
        
        Args:
            model_or_layer: 整个模型或单个layer
        """
        self.masks = {}  # 存储每个参数的mask
        self.hooks = []  # 存储注册的hooks
        self._extract_masks(model_or_layer)
    
    def _extract_masks(self, module):
        """
        从模型/层中提取当前的稀疏性mask
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                # 创建mask: True表示该位置应该保持为0
                mask = (param.data == 0).clone()
                self.masks[name] = mask
                
                # 统计稀疏性
                total = mask.numel()
                zeros = mask.sum().item()
                sparsity = zeros / total * 100
                print(f"Parameter {name}: {sparsity:.2f}% sparse ({zeros}/{total} zeros)")
    
    def apply_mask_to_gradients(self, module):
        """
        方法1: 将mask应用到梯度上（推荐）
        在反向传播后调用，将应该为0的位置的梯度清零
        """
        for name, param in module.named_parameters():
            if param.grad is not None and name in self.masks:
                # 将被剪枝位置的梯度置零
                param.grad.data[self.masks[name]] = 0
    
    def apply_mask_to_weights(self, module):
        """
        方法2: 将mask应用到权重上（投影法）
        在优化器更新后调用，强制将权重投影回稀疏约束
        """
        for name, param in module.named_parameters():
            if name in self.masks:
                # 强制将被剪枝位置设为0
                param.data[self.masks[name]] = 0
    
    def register_gradient_hooks(self, module):
        """
        方法3: 注册梯度hook（自动化方法1）
        自动在反向传播时应用mask
        """
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.masks:
                mask = self.masks[name]
                
                def hook_fn(grad, mask=mask):
                    """梯度hook函数"""
                    return grad * (~mask).float()  # ~mask: False的位置保留梯度
                
                handle = param.register_hook(hook_fn)
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """移除所有注册的hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def verify_sparsity(self, module, tolerance=1e-6):
        """
        验证稀疏性是否被保持
        
        Returns:
            dict: 每个参数的验证结果
        """
        results = {}
        for name, param in module.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                # 检查mask中是否有True值（即是否有被剪枝的位置）
                if mask.any():
                    # 检查被mask的位置是否接近0
                    violations = torch.abs(param.data[mask]) > tolerance
                    num_violations = violations.sum().item()
                    max_violation = torch.abs(param.data[mask]).max().item()
                else:
                    # 如果没有被剪枝的位置，则无违规
                    num_violations = 0
                    max_violation = 0.0
                
                results[name] = {
                    'violations': num_violations,
                    'max_violation': max_violation,
                    'is_valid': num_violations == 0
                }
        return results
    
    def get_sparsity_stats(self, module):
        """获取当前的稀疏性统计"""
        stats = {}
        for name, param in module.named_parameters():
            if name in self.masks:
                total = param.numel()
                actual_zeros = (torch.abs(param.data) < 1e-6).sum().item()
                expected_zeros = self.masks[name].sum().item()
                stats[name] = {
                    'total': total,
                    'expected_zeros': expected_zeros,
                    'actual_zeros': actual_zeros,
                    'sparsity': actual_zeros / total * 100 if total > 0 else 0.0
                }
        return stats


def layer_compensation_training_sparse(
    layer_idx, 
    adapted_layer, 
    pruned_hidden_states, 
    unpruned_hidden_states, 
    attention_mask, 
    position_ids, 
    sparsity_constraint: SparsityConstraint,  # 新增参数
    config: CompensationConfig,
    logger, 
    device
):
    """
    全层训练函数 - 支持稀疏性约束
    
    Args:
        layer_idx: 层索引
        adapted_layer: 带适配器的层
        pruned_hidden_states: 剪枝后的hidden states
        unpruned_hidden_states: 未剪枝的hidden states (作为目标)
        attention_mask: 注意力掩码
        position_ids: 位置编码
        sparsity_constraint: 稀疏性约束管理器 (新增)
        config: 补偿配置
        logger: 日志记录器
        device: 设备
    """
    # === 稀疏性信息记录 ===
    logger.info(f'\n{"="*80}')
    logger.info(f'Layer {layer_idx}: Initializing Sparse Compensation Training')
    logger.info(f'{"="*80}')
    
    initial_stats = sparsity_constraint.get_sparsity_stats(adapted_layer)
    logger.info('Initial Sparsity Status:')
    for name, stat in initial_stats.items():
        logger.info(
            f'  {name}: {stat["sparsity"]:.2f}% sparse '
            f'({stat["expected_zeros"]}/{stat["total"]} zeros)'
        )
    logger.info('='*80 + '\n')
    
    # === 原有训练设置 ===
    adapted_layer = adapted_layer.float()
    
    # 设置训练模式
    params = list(adapted_layer.parameters())
    for param in params:
        param.requires_grad = True
    adapted_layer.train()
    
    # 优化器
    optimizer = optim.AdamW(
        params,
        lr=config.stage1_lr,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.stage1_epochs,
        eta_min=1e-7
    )
    
    # 混合精度
    scaler = GradScaler() if config.use_amp else None
    
    # 损失函数
    loss_module = CompensationLosses()
    
    # 数据加载器
    dataset = DoubleFeatureDataset(
        pruned_hidden_states, 
        unpruned_hidden_states, 
        device=device
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config.stage1_batch_size, 
        shuffle=True, 
        drop_last=False
    )
    
    logger.info(f'=== Stage 1: Layer {layer_idx} Compensation Training (Sparse-aware) ===')
    logger.info(f'Trainable parameters: {sum(p.numel() for p in params):,}')
    logger.info(f'Training samples: {len(dataset)}')
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    # === 新增：稀疏性验证历史 ===
    sparsity_history = []
    
    for epoch in range(config.stage1_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_cosine = 0.0
        num_batches = 0
        
        for batch_idx, (idx, pruned_batch, unpruned_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                outputs = adapted_layer(
                    pruned_batch,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                    use_cache=False
                )
                
                compensated_hidden = outputs[0]
                
                # 计算多种损失
                loss_mse = loss_module.mse_loss(
                    compensated_hidden, 
                    unpruned_batch
                )
                loss_cosine = loss_module.cosine_loss(
                    compensated_hidden,
                    unpruned_batch
                )
                
                # 主损失：MSE
                loss = loss_mse
                
                # 可选：L2正则化
                if config.l2_reg > 0:
                    l2_penalty = sum(torch.norm(p, 2)**2 for p in params)
                    loss += config.l2_reg * l2_penalty
            
            # 反向传播
            if config.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # ===== 关键步骤：应用稀疏性约束 =====
                sparsity_constraint.apply_mask_to_gradients(adapted_layer)
                # ===================================
                
                torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # ===== 关键步骤：应用稀疏性约束 =====
                sparsity_constraint.apply_mask_to_gradients(adapted_layer)
                # ===================================
                
                torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)
                optimizer.step()
            
            # ===== 可选：双重保险 - 投影回稀疏约束 =====
            # 如果担心数值误差，可以取消下面这行的注释
            # sparsity_constraint.apply_mask_to_weights(adapted_layer)
            # ==========================================
            
            # 统计
            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_cosine += loss_cosine.item()
            num_batches += 1
            
            # 定期日志
            if batch_idx % config.log_interval == 0:
                logger.info(
                    f'Layer {layer_idx}, Epoch {epoch}, Batch {batch_idx}: '
                    f'Loss={loss.item():.6f}, MSE={loss_mse.item():.6f}, '
                    f'Cosine={loss_cosine.item():.6f}'
                )
        
        # Epoch统计
        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_cosine = epoch_cosine / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        training_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'mse': avg_mse,
            'cosine': avg_cosine,
            'lr': current_lr
        })
        
        # ===== 新增：定期验证稀疏性 =====
        if epoch % 5 == 0 or epoch == config.stage1_epochs - 1:
            violations = sparsity_constraint.verify_sparsity(adapted_layer)
            total_violations = sum(v['violations'] for v in violations.values())
            max_violation = max((v['max_violation'] for v in violations.values()), default=0.0)
            
            sparsity_history.append({
                'epoch': epoch,
                'total_violations': total_violations,
                'max_violation': max_violation
            })
            
            logger.info(
                f'Layer {layer_idx}, Epoch {epoch} Summary: '
                f'Loss={avg_loss:.6f}, MSE={avg_mse:.6f}, '
                f'Cosine={avg_cosine:.6f}, LR={current_lr:.2e} | '
                f'Sparsity Violations: {total_violations}, Max Violation: {max_violation:.2e}'
            )
            
            # 如果发现稀疏性被破坏，立即修正
            if total_violations > 0:
                logger.warning(
                    f'⚠️  Sparsity constraint violated at epoch {epoch}! '
                    f'Applying correction...'
                )
                sparsity_constraint.apply_mask_to_weights(adapted_layer)
        else:
            logger.info(
                f'Layer {layer_idx}, Epoch {epoch} Summary: '
                f'Loss={avg_loss:.6f}, MSE={avg_mse:.6f}, '
                f'Cosine={avg_cosine:.6f}, LR={current_lr:.2e}'
            )
        # ================================
        
        # 学习率调度
        scheduler.step()
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # 保存最佳模型（可选）
            if config.save_checkpoint:
                checkpoint_path = f"{config.checkpoint_dir}/layer_{layer_idx}_best.pt"
                # torch.save({
                #     'layer_idx': layer_idx,
                #     'epoch': epoch,
                #     'state_dict': adapted_layer.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'loss': best_loss,
                #     'sparsity_history': sparsity_history,  # 新增：保存稀疏性历史
                # }, checkpoint_path)
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stop_patience:
            logger.info(f'Early stopping at epoch {epoch} for layer {layer_idx}')
            break
    
    # ===== 训练完成后的最终验证 =====
    logger.info('\n' + '='*80)
    logger.info(f'Layer {layer_idx}: Final Sparsity Verification')
    logger.info('='*80)
    
    # 获取最终稀疏性统计
    final_stats = sparsity_constraint.get_sparsity_stats(adapted_layer)
    for name, stat in final_stats.items():
        logger.info(
            f'{name}: {stat["sparsity"]:.2f}% sparse '
            f'({stat["actual_zeros"]}/{stat["total"]} zeros, '
            f'expected: {stat["expected_zeros"]})'
        )
    
    # 验证稀疏性约束
    final_violations = sparsity_constraint.verify_sparsity(adapted_layer)
    total_final_violations = sum(v['violations'] for v in final_violations.values())
    
    if total_final_violations > 0:
        logger.warning(
            f'⚠️  Final correction needed: {total_final_violations} violations detected'
        )
        sparsity_constraint.apply_mask_to_weights(adapted_layer)
        logger.info('✓ Final correction applied')
    else:
        logger.info('✓ Sparsity constraints maintained successfully throughout training')
    
    logger.info('='*80 + '\n')
    # ====================================
    
    # 训练完成
    adapted_layer.eval()
    for param in params:
        param.requires_grad = False
    
    logger.info(
        f'=== Layer {layer_idx} Stage 1 Completed ===\n'
        f'Final Loss: {best_loss:.6f}, Epochs: {epoch+1}/{config.stage1_epochs}\n'
        f'Sparsity Status: {"Maintained" if total_final_violations == 0 else "Corrected"}'
    )
    
    # ===== 新增：在历史中包含稀疏性信息 =====
    training_history_with_sparsity = {
        'loss_history': training_history,
        'sparsity_history': sparsity_history
    }
    
    return best_loss, training_history_with_sparsity




def layer_compensation_training_v3_modular(
    layer_idx: int,
    adapted_layer,
    pruned_hidden_states,
    unpruned_hidden_states,
    attention_mask,
    position_ids,
    sparsity_constraint: SparsityConstraint,
    config: CompensationConfig,
    logger,
    device,
    train_component: Literal["attention", "ffn", "full"] = "full"  # 新增参数
):
    """
    模块化层训练函数 - 支持分别训练attention和FFN
    
    Args:
        layer_idx: 层索引
        adapted_layer: 带适配器的层
        pruned_hidden_states: 剪枝后的hidden states
        unpruned_hidden_states: 未剪枝的hidden states (作为目标)
        attention_mask: 注意力掩码
        position_ids: 位置编码
        sparsity_constraint: 稀疏性约束管理器
        config: 补偿配置
        logger: 日志记录器
        device: 设备
        train_component: 训练组件类型 - "attention", "ffn", 或 "full" #这里通过不同的control去控制
    """
    
    # === 参数准备和冻结策略 ===
    adapted_layer = adapted_layer.float()
    
    # 根据训练组件选择参数
    trainable_params, frozen_params = _select_trainable_params(
        adapted_layer, 
        train_component
    )
    
    # 设置参数训练状态
    for param in trainable_params:
        param.requires_grad = True
    for param in frozen_params:
        param.requires_grad = False
    
    adapted_layer.train()
    
    # === 稀疏性信息记录 ===
    logger.info(f'\n{"="*80}')
    logger.info(
        f'Layer {layer_idx}: Initializing Sparse Compensation Training '
        f'({train_component.upper()} Component)'
    )
    logger.info(f'{"="*80}')
    
    # 只检查当前训练组件的稀疏性
    initial_stats = _get_component_sparsity_stats(
        adapted_layer, 
        sparsity_constraint, 
        train_component
    )
    logger.info(f'Initial Sparsity Status ({train_component.upper()}):')
    for name, stat in initial_stats.items():
        logger.info(
            f'  {name}: {stat["sparsity"]:.2f}% sparse '
            f'({stat["expected_zeros"]}/{stat["total"]} zeros)'
        )
    logger.info('='*80 + '\n')
    
    # === 优化器设置 ===
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.stage1_lr,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.stage1_epochs,
        eta_min=1e-7
    )
    
    scaler = GradScaler() if config.use_amp else None
    loss_module = CompensationLosses()
    
    # === 数据加载器 ===
    dataset = DoubleFeatureDataset(
        pruned_hidden_states, 
        unpruned_hidden_states, 
        device=device
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config.stage1_batch_size, 
        shuffle=True, 
        drop_last=False
    )
    
    logger.info(
        f'=== Stage 1: Layer {layer_idx} {train_component.upper()} '
        f'Compensation Training (Sparse-aware) ==='
    )
    logger.info(
        f'Trainable parameters: '
        f'{sum(p.numel() for p in trainable_params):,}'
    )
    logger.info(f'Training samples: {len(dataset)}')
    
    # === 训练循环 ===
    best_loss = float('inf')
    patience_counter = 0
    training_history = []
    sparsity_history = []
    
    for epoch in range(config.stage1_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_cosine = 0.0
        num_batches = 0
        
        for batch_idx, (idx, pruned_batch, unpruned_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                outputs = adapted_layer(
                    pruned_batch,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                    use_cache=False
                )
                
                compensated_hidden = outputs[0]
                
                # 计算损失
                loss_mse = loss_module.mse_loss(
                    compensated_hidden, 
                    unpruned_batch
                )
                loss_cosine = loss_module.cosine_loss(
                    compensated_hidden,
                    unpruned_batch
                )
                
                loss = loss_mse
                
                # L2正则化
                if config.l2_reg > 0:
                    l2_penalty = sum(
                        torch.norm(p, 2)**2 for p in trainable_params
                    )
                    loss += config.l2_reg * l2_penalty
            
            # 反向传播
            if config.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # 只对当前训练组件应用稀疏性约束
                _apply_component_sparsity_constraint(
                    adapted_layer,
                    sparsity_constraint,
                    train_component
                )
                
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, 
                    max_norm=config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # 只对当前训练组件应用稀疏性约束
                _apply_component_sparsity_constraint(
                    adapted_layer,
                    sparsity_constraint,
                    train_component
                )
                
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, 
                    max_norm=config.max_grad_norm
                )
                optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_cosine += loss_cosine.item()
            num_batches += 1
            
            # 定期日志
            if batch_idx % config.log_interval == 0:
                logger.info(
                    f'Layer {layer_idx} [{train_component.upper()}], '
                    f'Epoch {epoch}, Batch {batch_idx}: '
                    f'Loss={loss.item():.6f}, MSE={loss_mse.item():.6f}, '
                    f'Cosine={loss_cosine.item():.6f}'
                )
        
        # Epoch统计
        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_cosine = epoch_cosine / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        training_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'mse': avg_mse,
            'cosine': avg_cosine,
            'lr': current_lr
        })
        
        # 定期验证稀疏性
        if epoch % 5 == 0 or epoch == config.stage1_epochs - 1:
            violations = _verify_component_sparsity(
                adapted_layer,
                sparsity_constraint,
                train_component
            )
            total_violations = sum(v['violations'] for v in violations.values())
            max_violation = max(
                (v['max_violation'] for v in violations.values()), 
                default=0.0
            )
            
            sparsity_history.append({
                'epoch': epoch,
                'total_violations': total_violations,
                'max_violation': max_violation
            })
            
            logger.info(
                f'Layer {layer_idx} [{train_component.upper()}], Epoch {epoch} Summary: '
                f'Loss={avg_loss:.6f}, MSE={avg_mse:.6f}, '
                f'Cosine={avg_cosine:.6f}, LR={current_lr:.2e} | '
                f'Sparsity Violations: {total_violations}, '
                f'Max Violation: {max_violation:.2e}'
            )
            
            # 如果发现稀疏性被破坏,立即修正
            if total_violations > 0:
                logger.warning(
                    f'⚠️  Sparsity constraint violated at epoch {epoch}! '
                    f'Applying correction...'
                )
                _apply_component_mask_to_weights(
                    adapted_layer,
                    sparsity_constraint,
                    train_component
                )
        else:
            logger.info(
                f'Layer {layer_idx} [{train_component.upper()}], Epoch {epoch} Summary: '
                f'Loss={avg_loss:.6f}, MSE={avg_mse:.6f}, '
                f'Cosine={avg_cosine:.6f}, LR={current_lr:.2e}'
            )
        
        # 学习率调度
        scheduler.step()
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stop_patience:
            logger.info(
                f'Early stopping at epoch {epoch} for layer {layer_idx} '
                f'[{train_component.upper()}]'
            )
            break
    
    # === 训练完成后的最终验证 ===
    logger.info('\n' + '='*80)
    logger.info(
        f'Layer {layer_idx} [{train_component.upper()}]: '
        f'Final Sparsity Verification'
    )
    logger.info('='*80)
    
    final_stats = _get_component_sparsity_stats(
        adapted_layer,
        sparsity_constraint,
        train_component
    )
    for name, stat in final_stats.items():
        logger.info(
            f'{name}: {stat["sparsity"]:.2f}% sparse '
            f'({stat["actual_zeros"]}/{stat["total"]} zeros, '
            f'expected: {stat["expected_zeros"]})'
        )
    
    final_violations = _verify_component_sparsity(
        adapted_layer,
        sparsity_constraint,
        train_component
    )
    total_final_violations = sum(v['violations'] for v in final_violations.values())
    
    if total_final_violations > 0:
        logger.warning(
            f'⚠️  Final correction needed: {total_final_violations} violations detected'
        )
        _apply_component_mask_to_weights(
            adapted_layer,
            sparsity_constraint,
            train_component
        )
        logger.info('✓ Final correction applied')
    else:
        logger.info(
            '✓ Sparsity constraints maintained successfully throughout training'
        )
    
    logger.info('='*80 + '\n')
    
    # 训练完成
    adapted_layer.eval()
    for param in trainable_params:
        param.requires_grad = False
    
    logger.info(
        f'=== Layer {layer_idx} [{train_component.upper()}] Stage 1 Completed ===\n'
        f'Final Loss: {best_loss:.6f}, Epochs: {epoch+1}/{config.stage1_epochs}\n'
        f'Sparsity Status: {"Maintained" if total_final_violations == 0 else "Corrected"}'
    )
    
    training_history_with_sparsity = {
        'component': train_component,
        'loss_history': training_history,
        'sparsity_history': sparsity_history
    }
    
    return best_loss, training_history_with_sparsity

# ============================================================================
# 辅助函数 - 修正版
# ============================================================================

def _select_trainable_params(layer, component: str):
    """根据组件类型选择可训练参数"""
    trainable_params = []
    frozen_params = []
    
    named_params = dict(layer.named_parameters())
    
    if component == "attention":
        attn_keywords = [
            'self_attn', 'attention', 
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'query', 'key', 'value', 'dense'
        ]
        
        for name, param in named_params.items():
            if any(keyword in name.lower() for keyword in attn_keywords):
                trainable_params.append(param)
            else:
                frozen_params.append(param)
                
    elif component == "ffn":
        ffn_keywords = [
            'mlp', 'ffn', 'feed_forward',
            'fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj',
            'intermediate', 'output'
        ]
        
        for name, param in named_params.items():
            if any(keyword in name.lower() for keyword in ffn_keywords):
                trainable_params.append(param)
            else:
                frozen_params.append(param)
                
    elif component == "full":
        trainable_params = list(layer.parameters())
        frozen_params = []
    else:
        raise ValueError(
            f"Unknown component type: {component}. "
            f"Must be 'attention', 'ffn', or 'full'"
        )
    
    return trainable_params, frozen_params


def _get_component_keywords(component: str) -> List[str]:
    """获取组件关键词列表"""
    if component == "attention":
        return [
            'self_attn', 'attention',
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'query', 'key', 'value', 'dense'
        ]
    elif component == "ffn":
        return [
            'mlp', 'ffn', 'feed_forward',
            'fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj',
            'intermediate', 'output'
        ]
    else:
        return []


def _get_component_sparsity_stats(layer, sparsity_constraint, component: str):
    """获取特定组件的稀疏性统计"""
    all_stats = sparsity_constraint.get_sparsity_stats(layer)
    
    if component == "full":
        return all_stats
    
    component_stats = {}
    filter_keywords = _get_component_keywords(component)
    
    for name, stat in all_stats.items():
        if any(keyword in name.lower() for keyword in filter_keywords):
            component_stats[name] = stat
    
    return component_stats


def _verify_component_sparsity(layer, sparsity_constraint, component: str):
    """验证特定组件的稀疏性约束"""
    all_violations = sparsity_constraint.verify_sparsity(layer)
    
    if component == "full":
        return all_violations
    
    component_violations = {}
    filter_keywords = _get_component_keywords(component)
    
    for name, violation in all_violations.items():
        if any(keyword in name.lower() for keyword in filter_keywords):
            component_violations[name] = violation
    
    return component_violations


def _apply_component_sparsity_constraint(
    layer, 
    sparsity_constraint, 
    component: str
):
    """✅ 修正版：对特定组件应用稀疏性约束到梯度"""
    if component == "full":
        # 使用原始的完整方法
        sparsity_constraint.apply_mask_to_gradients(layer)
    else:
        # 只对特定组件的参数应用mask
        filter_keywords = _get_component_keywords(component)
        
        for name, param in layer.named_parameters():
            if param.grad is not None and name in sparsity_constraint.masks:
                # 检查是否属于当前训练组件
                if any(keyword in name.lower() for keyword in filter_keywords):
                    # ✅ 将被剪枝位置（mask=True）的梯度置零
                    mask = sparsity_constraint.masks[name]
                    param.grad.data[mask] = 0


def _apply_component_mask_to_weights(
    layer, 
    sparsity_constraint, 
    component: str
):
    """✅ 修正版：对特定组件应用稀疏性约束到权重"""
    if component == "full":
        # 使用原始的完整方法
        sparsity_constraint.apply_mask_to_weights(layer)
    else:
        # 只对特定组件的参数应用mask
        filter_keywords = _get_component_keywords(component)
        
        for name, param in layer.named_parameters():
            if name in sparsity_constraint.masks:
                # 检查是否属于当前训练组件
                if any(keyword in name.lower() for keyword in filter_keywords):
                    # ✅ 强制将被剪枝位置设为0
                    mask = sparsity_constraint.masks[name]
                    with torch.no_grad():
                        param.data[mask] = 0


def qwen_layer_compensation_training_sparse(
    layer_idx, 
    adapted_layer, 
    position_embeddings,
    pruned_hidden_states, 
    unpruned_hidden_states, 
    attention_mask, 
    position_ids, 
    sparsity_constraint: SparsityConstraint,  # 新增参数
    config: CompensationConfig,
    logger, 
    device
):
    """
    全层训练函数 - 支持稀疏性约束
    
    Args:
        layer_idx: 层索引
        adapted_layer: 带适配器的层
        pruned_hidden_states: 剪枝后的hidden states
        unpruned_hidden_states: 未剪枝的hidden states (作为目标)
        attention_mask: 注意力掩码
        position_ids: 位置编码
        sparsity_constraint: 稀疏性约束管理器 (新增)
        config: 补偿配置
        logger: 日志记录器
        device: 设备
    """
    # === 稀疏性信息记录 ===
    logger.info(f'\n{"="*80}')
    logger.info(f'Layer {layer_idx}: Initializing Sparse Compensation Training')
    logger.info(f'{"="*80}')
    
    initial_stats = sparsity_constraint.get_sparsity_stats(adapted_layer)
    logger.info('Initial Sparsity Status:')
    for name, stat in initial_stats.items():
        logger.info(
            f'  {name}: {stat["sparsity"]:.2f}% sparse '
            f'({stat["expected_zeros"]}/{stat["total"]} zeros)'
        )
    logger.info('='*80 + '\n')
    
    # === 原有训练设置 ===
    adapted_layer = adapted_layer.float()
    
    # 设置训练模式
    params = list(adapted_layer.parameters())
    for param in params:
        param.requires_grad = True
    adapted_layer.train()
    
    # 优化器
    optimizer = optim.AdamW(
        params,
        lr=config.stage1_lr,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.stage1_epochs,
        eta_min=1e-7
    )
    
    # 混合精度
    scaler = GradScaler() if config.use_amp else None
    
    # 损失函数
    loss_module = CompensationLosses()
    
    # 数据加载器
    dataset = DoubleFeatureDataset(
        pruned_hidden_states, 
        unpruned_hidden_states, 
        device=device
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config.stage1_batch_size, 
        shuffle=True, 
        drop_last=False
    )
    
    logger.info(f'=== Stage 1: Layer {layer_idx} Compensation Training (Sparse-aware) ===')
    logger.info(f'Trainable parameters: {sum(p.numel() for p in params):,}')
    logger.info(f'Training samples: {len(dataset)}')
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    # === 新增：稀疏性验证历史 ===
    sparsity_history = []
    
    for epoch in range(config.stage1_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_cosine = 0.0
        num_batches = 0
        
        for batch_idx, (idx, pruned_batch, unpruned_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                outputs = adapted_layer(
                    pruned_batch,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                    position_embeddings=position_embeddings, # 新增参数
                    use_cache=False
                )
                
                compensated_hidden = outputs[0]
                
                # 计算多种损失
                loss_mse = loss_module.mse_loss(
                    compensated_hidden, 
                    unpruned_batch
                )
                loss_cosine = loss_module.cosine_loss(
                    compensated_hidden,
                    unpruned_batch
                )
                
                # 主损失：MSE
                loss = loss_mse
                
                # 可选：L2正则化
                if config.l2_reg > 0:
                    l2_penalty = sum(torch.norm(p, 2)**2 for p in params)
                    loss += config.l2_reg * l2_penalty
            
            # 反向传播
            if config.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # ===== 关键步骤：应用稀疏性约束 =====
                sparsity_constraint.apply_mask_to_gradients(adapted_layer)
                # ===================================
                
                torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # ===== 关键步骤：应用稀疏性约束 =====
                sparsity_constraint.apply_mask_to_gradients(adapted_layer)
                # ===================================
                
                torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)
                optimizer.step()
            
            # ===== 可选：双重保险 - 投影回稀疏约束 =====
            # 如果担心数值误差，可以取消下面这行的注释
            # sparsity_constraint.apply_mask_to_weights(adapted_layer)
            # ==========================================
            
            # 统计
            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_cosine += loss_cosine.item()
            num_batches += 1
            
            # 定期日志
            if batch_idx % config.log_interval == 0:
                logger.info(
                    f'Layer {layer_idx}, Epoch {epoch}, Batch {batch_idx}: '
                    f'Loss={loss.item():.6f}, MSE={loss_mse.item():.6f}, '
                    f'Cosine={loss_cosine.item():.6f}'
                )
        
        # Epoch统计
        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_cosine = epoch_cosine / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        training_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'mse': avg_mse,
            'cosine': avg_cosine,
            'lr': current_lr
        })
        
        # ===== 新增：定期验证稀疏性 =====
        if epoch % 5 == 0 or epoch == config.stage1_epochs - 1:
            violations = sparsity_constraint.verify_sparsity(adapted_layer)
            total_violations = sum(v['violations'] for v in violations.values())
            max_violation = max((v['max_violation'] for v in violations.values()), default=0.0)
            
            sparsity_history.append({
                'epoch': epoch,
                'total_violations': total_violations,
                'max_violation': max_violation
            })
            
            logger.info(
                f'Layer {layer_idx}, Epoch {epoch} Summary: '
                f'Loss={avg_loss:.6f}, MSE={avg_mse:.6f}, '
                f'Cosine={avg_cosine:.6f}, LR={current_lr:.2e} | '
                f'Sparsity Violations: {total_violations}, Max Violation: {max_violation:.2e}'
            )
            
            # 如果发现稀疏性被破坏，立即修正
            if total_violations > 0:
                logger.warning(
                    f'⚠️  Sparsity constraint violated at epoch {epoch}! '
                    f'Applying correction...'
                )
                sparsity_constraint.apply_mask_to_weights(adapted_layer)
        else:
            logger.info(
                f'Layer {layer_idx}, Epoch {epoch} Summary: '
                f'Loss={avg_loss:.6f}, MSE={avg_mse:.6f}, '
                f'Cosine={avg_cosine:.6f}, LR={current_lr:.2e}'
            )
        # ================================
        
        # 学习率调度
        scheduler.step()
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # 保存最佳模型（可选）
            if config.save_checkpoint:
                checkpoint_path = f"{config.checkpoint_dir}/layer_{layer_idx}_best.pt"
                # torch.save({
                #     'layer_idx': layer_idx,
                #     'epoch': epoch,
                #     'state_dict': adapted_layer.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'loss': best_loss,
                #     'sparsity_history': sparsity_history,  # 新增：保存稀疏性历史
                # }, checkpoint_path)
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stop_patience:
            logger.info(f'Early stopping at epoch {epoch} for layer {layer_idx}')
            break
    
    # ===== 训练完成后的最终验证 =====
    logger.info('\n' + '='*80)
    logger.info(f'Layer {layer_idx}: Final Sparsity Verification')
    logger.info('='*80)
    
    # 获取最终稀疏性统计
    final_stats = sparsity_constraint.get_sparsity_stats(adapted_layer)
    for name, stat in final_stats.items():
        logger.info(
            f'{name}: {stat["sparsity"]:.2f}% sparse '
            f'({stat["actual_zeros"]}/{stat["total"]} zeros, '
            f'expected: {stat["expected_zeros"]})'
        )
    
    # 验证稀疏性约束
    final_violations = sparsity_constraint.verify_sparsity(adapted_layer)
    total_final_violations = sum(v['violations'] for v in final_violations.values())
    
    if total_final_violations > 0:
        logger.warning(
            f'⚠️  Final correction needed: {total_final_violations} violations detected'
        )
        sparsity_constraint.apply_mask_to_weights(adapted_layer)
        logger.info('✓ Final correction applied')
    else:
        logger.info('✓ Sparsity constraints maintained successfully throughout training')
    
    logger.info('='*80 + '\n')
    # ====================================
    
    # 训练完成
    adapted_layer.eval()
    for param in params:
        param.requires_grad = False
    
    logger.info(
        f'=== Layer {layer_idx} Stage 1 Completed ===\n'
        f'Final Loss: {best_loss:.6f}, Epochs: {epoch+1}/{config.stage1_epochs}\n'
        f'Sparsity Status: {"Maintained" if total_final_violations == 0 else "Corrected"}'
    )
    
    # ===== 新增：在历史中包含稀疏性信息 =====
    training_history_with_sparsity = {
        'loss_history': training_history,
        'sparsity_history': sparsity_history
    }
    
    return best_loss, training_history_with_sparsity


def end_to_end_logits_distillation_sparse(
    model: nn.Module,
    cache_dir: str,
    dataloader: torch.utils.data.DataLoader,
    sparsity_constraint: SparsityConstraint,
    config: 'CompensationConfig',
    logger,
    device: torch.device
):
    """
    阶段2：使用预计算的 Logits 训练 Student，支持稀疏性约束
    
    主要特性：
    1. 在训练过程中维持剪枝后的稀疏性
    2. 支持断点续训
    3. 定期验证稀疏性
    4. 修复了所有已知的Loss聚合问题
    """
    
    if not config.enable_stage2:
        logger.info('Stage 2 is disabled, skipping...')
        return model, []

    logger.info('='*80)
    logger.info('STAGE 2: Sparse-aware Logits-Only Knowledge Distillation (Cached)')
    logger.info('='*80)
    
    # ===== 稀疏性初始状态 =====
    logger.info('\n[Step 0] Initial Sparsity Status:')
    initial_stats = sparsity_constraint.get_sparsity_stats(model)
    total_params = sum(s['total'] for s in initial_stats.values())
    total_zeros = sum(s['expected_zeros'] for s in initial_stats.values())
    overall_sparsity = (total_zeros / total_params * 100) if total_params > 0 else 0
    logger.info(f'  Overall sparsity: {overall_sparsity:.2f}% ({total_zeros:,}/{total_params:,} zeros)')
    
    logger.info('\n[Step 1] Preparing student model for training...')
    
    # 确保 Student 在 GPU
    model = model.to(device).float()
    model.train()

    # 梯度检查点
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info('✓ Gradient checkpointing enabled')
        
    # 优化器配置
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    use_8bit = getattr(config, 'use_8bit_optimizer', True)
    
    # 优化器定义
    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params, lr=config.stage2_lr, weight_decay=config.weight_decay,
                betas=(config.adam_beta1, config.adam_beta2)
            )
            logger.info('✓ Using 8-bit AdamW optimizer')
        except ImportError:
            logger.warning('⚠ bitsandbytes not found, using standard AdamW')
            optimizer = torch.optim.AdamW(
                trainable_params, lr=config.stage2_lr, weight_decay=config.weight_decay,
                betas=(config.adam_beta1, config.adam_beta2)
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=config.stage2_lr, weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2)
        )
        
    # 调度器
    total_steps = config.stage2_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

    # 断点和恢复逻辑
    latest_ckpt_path = os.path.join(config.checkpoint_dir, 'stage2_latest_sparse.pt')
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(latest_ckpt_path) and getattr(config, 'resume_stage2', True):
        logger.info(f"🔄 Found checkpoint: {latest_ckpt_path}. Loading Stage 2 progress...")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location='cpu')
            
            # 恢复模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # 恢复优化器和调度器
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复进度
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('best_loss', float('inf'))
            
            # 恢复 Scaler
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                 
            logger.info(f"✅ Resumed from Epoch {start_epoch}. Best Loss: {best_loss:.6f}")
            
            # ===== 验证恢复后的稀疏性 =====
            violations = sparsity_constraint.verify_sparsity(model)
            total_violations = sum(v['violations'] for v in violations.values())
            if total_violations > 0:
                logger.warning(f"⚠️  Loaded checkpoint has {total_violations} sparsity violations. Correcting...")
                sparsity_constraint.apply_mask_to_weights(model)
        
        except Exception as e:
            logger.error(f"❌ Failed to resume Stage 2: {e}. Starting from scratch.")
            start_epoch = 0
    else:
        logger.info(f"No checkpoint found at {latest_ckpt_path}. Starting from scratch.")
            
    # ========== 训练循环 ==========
    logger.info('\n[Step 2] Starting sparse-aware training...')
    training_history = []
    sparsity_history = []
    
    for epoch in range(start_epoch, config.stage2_epochs):
        
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            
            input_ids = batch[0].to(device)
            batch_size, seq_len = input_ids.shape
            
            # 加载 Cache
            cache_path = os.path.join(cache_dir, f'batch_{batch_idx}.pt')
            if not os.path.exists(cache_path):
                logger.warning(f"Cache not found: {cache_path}, skipping batch {batch_idx}")
                continue
                
            teacher_cache = torch.load(cache_path, map_location='cpu')
            teacher_logits = teacher_cache['logits'].to(device)
            
            # 构造辅助变量
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Student 前向
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                student_outputs = model(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                
                # 计算 Loss 
                loss_tensor = compute_kl_divergence_loss(
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_logits,
                    attention_mask=attention_mask,
                    temperature=config.kl_temperature
                )
                
                loss = loss_tensor.mean() 
                loss_scaled = loss / accumulation_steps
                
            # 反向传播
            if config.use_amp and scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            
            # ===== 关键步骤：应用稀疏性约束到梯度 =====
            sparsity_constraint.apply_mask_to_gradients(model)
            # =========================================
            
            is_update_step = (batch_idx + 1) % accumulation_steps == 0
            
            # 梯度更新
            if is_update_step:
                if config.use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                # ===== 可选：双重保险 - 投影回稀疏约束 =====
                # 如果担心数值误差，可以取消下面这行的注释
                # sparsity_constraint.apply_mask_to_weights(model)
                # ==========================================
            
            # 统计
            batch_loss = loss.item() * accumulation_steps
            epoch_loss += batch_loss
            num_batches += 1
            
            if batch_idx % config.log_interval == 0:
                gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
                logger.info(
                    f'Epoch {epoch+1}/{config.stage2_epochs}, Batch {batch_idx}: '
                    f'Loss={batch_loss:.6f}, LR={scheduler.get_last_lr()[0]:.2e}, GPU={gpu_mem:.2f}GB'
                )
                
            # 及时释放
            del teacher_logits, teacher_cache, student_outputs
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        # Epoch 结束统计
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        training_history.append({'epoch': epoch, 'loss': avg_loss})
        
        # ===== 定期验证稀疏性 =====
        if epoch % 5 == 0 or epoch == config.stage2_epochs - 1:
            logger.info(f'\n{"="*80}')
            logger.info(f'Epoch {epoch+1} - Sparsity Verification')
            logger.info(f'{"="*80}')
            
            violations = sparsity_constraint.verify_sparsity(model)
            total_violations = sum(v['violations'] for v in violations.values())
            max_violation = max((v['max_violation'] for v in violations.values()), default=0.0)
            
            sparsity_history.append({
                'epoch': epoch,
                'total_violations': total_violations,
                'max_violation': max_violation,
                'avg_loss': avg_loss
            })
            
            logger.info(f'Avg Loss: {avg_loss:.6f}')
            logger.info(f'Sparsity Violations: {total_violations}')
            logger.info(f'Max Violation: {max_violation:.2e}')
            
            if total_violations > 0:
                logger.warning(f'⚠️  Sparsity violated! Applying correction...')
                sparsity_constraint.apply_mask_to_weights(model)
                
                # 验证修正是否成功
                post_violations = sparsity_constraint.verify_sparsity(model)
                post_total = sum(v['violations'] for v in post_violations.values())
                if post_total == 0:
                    logger.info('✓ Sparsity corrected successfully')
                else:
                    logger.error(f'❌ Correction failed: {post_total} violations remain')
            else:
                logger.info('✓ Sparsity maintained')
            
            logger.info(f'{"="*80}\n')
        else:
            logger.info(f'Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.6f}')
        
        # ===== Epoch 结束时保存断点 =====
        if epoch % 3 == 0 and epoch != 0: 
            try:
                ckpt_dict = {
                    'epoch': epoch + 1,
                    'best_loss': best_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'sparsity_history': sparsity_history,  # 保存稀疏性历史
                }
                if scaler:
                    ckpt_dict['scaler_state_dict'] = scaler.state_dict()
                    
                tmp_path = latest_ckpt_path + ".tmp"
                torch.save(ckpt_dict, tmp_path)
                os.rename(tmp_path, latest_ckpt_path)
                logger.info(f'💾 Saved epoch-end checkpoint for Epoch {epoch+1}')
                
            except Exception as e:
                logger.error(f"❌ Failed to save epoch-end checkpoint: {e}")

        # 最佳模型保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            if config.save_checkpoint:
                ckpt_path = os.path.join(config.checkpoint_dir, 'stage2_best_sparse.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                    'sparsity_history': sparsity_history
                }, ckpt_path)
                logger.info(f'✓ Saved best sparse model to {ckpt_path}')
    
    # ===== 最终稀疏性验证 =====
    logger.info('\n' + '='*80)
    logger.info('Final Sparsity Verification')
    logger.info('='*80)
    
    final_stats = sparsity_constraint.get_sparsity_stats(model)
    total_params = sum(s['total'] for s in final_stats.values())
    total_actual_zeros = sum(s['actual_zeros'] for s in final_stats.values())
    total_expected_zeros = sum(s['expected_zeros'] for s in final_stats.values())
    final_sparsity = (total_actual_zeros / total_params * 100) if total_params > 0 else 0
    
    logger.info(f'Final overall sparsity: {final_sparsity:.2f}%')
    logger.info(f'  Expected zeros: {total_expected_zeros:,}')
    logger.info(f'  Actual zeros: {total_actual_zeros:,}')
    logger.info(f'  Difference: {abs(total_actual_zeros - total_expected_zeros):,}')
    
    final_violations = sparsity_constraint.verify_sparsity(model)
    total_final_violations = sum(v['violations'] for v in final_violations.values())
    
    if total_final_violations > 0:
        logger.warning(f'⚠️  Final correction needed: {total_final_violations} violations')
        sparsity_constraint.apply_mask_to_weights(model)
        logger.info('✓ Final correction applied')
    else:
        logger.info('✓ Sparsity constraints maintained successfully throughout training')
    
    logger.info('='*80 + '\n')
    
    model.eval()
    
    # 返回包含稀疏性信息的历史
    training_history_with_sparsity = {
        'loss_history': training_history,
        'sparsity_history': sparsity_history
    }
    
    return model, training_history_with_sparsity
