"""
Realformer (RealAttn) 注意力实现模块
通过残差注意力连接提高模型性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger("attn_experiment")

def get_realformer_attention_config():
    """
    获取Realformer注意力机制配置
    
    Returns:
        config: 配置字典
    """
    return {
        "name": "Realformer Attention",
        "description": "通过残差注意力连接提高模型性能的注意力机制"
    }

class RealformerAttention(nn.Module):
    """
    Realformer自注意力模块，使用残差注意力连接
    """
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 用于计算注意力权重的查询、键、值投影
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # 输出投影
        self.output = nn.Linear(self.all_head_size, hidden_size)
        
        # 保存之前层的注意力权重，用于残差连接
        self.prev_attn_probs = None
        
        logger.info(f"初始化Realformer注意力: hidden_size={hidden_size}, heads={num_attention_heads}")
    
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
        
        # 生成查询、键、值
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # 重塑维度以支持多头
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 计算注意力分数 (q @ k.T / sqrt(head_size))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 调整掩码维度，扩展为[batch_size, 1, 1, seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 将掩码转换为float
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # 应用掩码
            attention_scores = attention_scores + extended_attention_mask
        
        # 应用softmax得到注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 关键的Realformer残差注意力连接：
        # 如果有前一层的注意力权重，添加它们作为残差连接
        if self.prev_attn_probs is not None:
            # 确保维度匹配
            if self.prev_attn_probs.shape == attention_probs.shape:
                attention_probs = attention_probs + self.prev_attn_probs
                # 重新归一化
                attention_probs = F.softmax(attention_probs, dim=-1)
        
        # 保存当前的注意力权重，以便下一次调用使用
        self.prev_attn_probs = attention_probs.detach()
        
        # 应用头掩码（如果提供）
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # 应用注意力权重到值上
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 转置回原始维度
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # 合并注意力头
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 投影到输出维度
        outputs = self.output(context_layer)
        
        if output_attentions:
            return outputs, attention_probs
        
        return outputs

def replace_with_realformer_attention(model):
    """
    将模型的注意力机制替换为Realformer注意力机制
    
    Args:
        model: 原始模型
    
    Returns:
        model: 替换后的模型
    """
    logger.info(f"替换为Realformer注意力机制")
    
    # 创建一个字典，存储每一层的Realformer注意力模块
    realformer_layers = {}
    
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果是自注意力模块
        if "self_attn" in name and not "output" in name and hasattr(module, "query_key_value"):
            # 获取原始模块的配置
            hidden_size = module.hidden_size
            num_attention_heads = module.num_attention_heads
            
            # 创建Realformer注意力模块
            realformer_attn = RealformerAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads
            )
            
            # 存储到字典中
            realformer_layers[name] = realformer_attn
            
            # 暂存原始的前向函数
            orig_forward = module.forward
            
            # 创建新的前向函数
            def make_forward(attn_module):
                def new_forward(*args, **kwargs):
                    # 调用Realformer注意力的前向函数
                    return attn_module(*args, **kwargs)
                return new_forward
            
            # 替换前向函数
            module.forward = make_forward(realformer_attn)
            
            logger.info(f"已替换模块 {name} 为Realformer注意力")
    
    return model 