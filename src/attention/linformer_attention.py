"""
Linformer注意力实现模块
通过低秩投影实现线性注意力复杂度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger("attn_experiment")

def get_linformer_attention_config(k_ratio=0.25):
    """
    获取Linformer注意力机制配置
    
    Args:
        k_ratio: 投影维度与序列长度的比例，控制压缩程度
    
    Returns:
        config: 配置字典
    """
    return {
        "name": "Linformer Attention",
        "description": "通过低秩投影矩阵实现线性复杂度的注意力机制",
        "k_ratio": k_ratio
    }

class LinformerAttention(nn.Module):
    """
    Linformer自注意力模块，将序列长度降维以实现线性复杂度
    """
    def __init__(self, hidden_size, num_attention_heads, k_ratio=0.25, max_seq_length=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 计算投影维度k
        self.k = max(1, int(max_seq_length * k_ratio))
        
        # E和F投影矩阵，用于将键和值的序列长度从seq_len降为k
        self.E = nn.Parameter(torch.Tensor(self.num_attention_heads, max_seq_length, self.k))
        self.F = nn.Parameter(torch.Tensor(self.num_attention_heads, max_seq_length, self.k))
        
        # 初始化投影矩阵
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)
        
        # 用于计算注意力权重的查询、键、值投影
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # 输出投影
        self.output = nn.Linear(self.all_head_size, hidden_size)
        
        logger.info(f"初始化Linformer注意力: hidden_size={hidden_size}, heads={num_attention_heads}, k={self.k}")
    
    def transpose_for_scores(self, x):
        """将张量重塑为[batch_size, num_heads, seq_len, head_size]"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # 确保序列长度不超过预设的max_seq_length
        if seq_len > self.E.size(1):
            raise ValueError(f"输入序列长度 {seq_len} 超过了Linformer的最大序列长度 {self.E.size(1)}")
        
        # 生成查询、键、值
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # 重塑维度以支持多头
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 使用投影矩阵E和F降低序列长度维度
        # E: [num_heads, seq_len, k]
        # key_layer: [batch_size, num_heads, seq_len, head_size]
        # 投影后: [batch_size, num_heads, k, head_size]
        projected_keys = torch.matmul(
            self.E[:, :seq_len, :].unsqueeze(0), 
            key_layer
        )
        
        # 使用投影矩阵F投影值
        projected_values = torch.matmul(
            self.F[:, :seq_len, :].unsqueeze(0), 
            value_layer
        )
        
        # 计算注意力分数 (q @ k.T / sqrt(head_size))
        # [batch_size, num_heads, seq_len, k]
        attention_scores = torch.matmul(query_layer, projected_keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 调整掩码维度，扩展为[batch_size, 1, seq_len, 1]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            # 将掩码转换为float
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # 应用掩码
            attention_scores = attention_scores + extended_attention_mask
        
        # 应用softmax得到注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 应用头掩码（如果提供）
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # 应用注意力权重到投影的值上
        # [batch_size, num_heads, seq_len, head_size]
        context_layer = torch.matmul(attention_probs, projected_values)
        
        # 转置回原始维度
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # 合并注意力头
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 投影到输出维度
        outputs = self.output(context_layer)
        
        if output_attentions:
            # Linformer不直接返回原始注意力矩阵，因为它使用了投影
            return outputs, None
        
        return outputs

def replace_with_linformer_attention(model, k_ratio=0.25, max_seq_length=512, last_layer_only=False):
    """
    将模型的注意力机制替换为Linformer注意力机制
    
    Args:
        model: 原始模型
        k_ratio: k的比例，用于确定投影维度
        max_seq_length: 最大序列长度
        last_layer_only: 是否只替换最后一层注意力，默认为False
    
    Returns:
        model: 替换后的模型
    """
    logger.info(f"替换为Linformer注意力机制: k_ratio={k_ratio}, max_seq_length={max_seq_length}")
    
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果是自注意力模块
        if "self_attn" in name and not "output" in name and hasattr(module, "query_key_value"):
            # 获取原始模块的配置
            hidden_size = module.hidden_size
            num_attention_heads = module.num_attention_heads
            
            # 创建Linformer注意力模块
            linformer_attn = LinformerAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                k_ratio=k_ratio,
                max_seq_length=max_seq_length
            )
            
            # 暂存原始的前向函数
            orig_forward = module.forward
            
            # 创建新的前向函数
            def new_forward(*args, **kwargs):
                # 调用Linformer注意力的前向函数
                return linformer_attn(*args, **kwargs)
            
            # 替换前向函数
            module.forward = new_forward
            
            logger.info(f"已替换模块 {name} 为Linformer注意力")
    
    return model 