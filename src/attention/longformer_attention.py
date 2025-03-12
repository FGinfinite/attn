"""
Longformer注意力实现模块
结合局部窗口注意力和全局注意力的混合机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger("attn_experiment")

def get_longformer_attention_config(window_size=128, global_tokens_ratio=0.1):
    """
    获取Longformer注意力机制配置
    
    Args:
        window_size: 局部窗口大小
        global_tokens_ratio: 全局token比例
    
    Returns:
        config: 配置字典
    """
    return {
        "name": "Longformer Attention",
        "description": "结合局部窗口注意力和全局注意力的混合机制",
        "window_size": window_size,
        "global_tokens_ratio": global_tokens_ratio
    }

class LongformerAttention(nn.Module):
    """
    Longformer自注意力模块，结合局部滑动窗口注意力和全局注意力
    """
    def __init__(self, hidden_size, num_attention_heads, window_size=128, global_tokens_ratio=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.global_tokens_ratio = global_tokens_ratio
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 用于计算注意力权重的查询、键、值投影
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # 输出投影
        self.output = nn.Linear(self.all_head_size, hidden_size)
        
        logger.info(f"初始化Longformer注意力: hidden_size={hidden_size}, heads={num_attention_heads}, window_size={window_size}")
    
    def transpose_for_scores(self, x):
        """将张量重塑为[batch_size, num_heads, seq_len, head_size]"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def _create_sliding_window_mask(self, seq_len, device):
        """创建滑动窗口注意力掩码"""
        # 创建位置索引矩阵
        positions = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)
        positions_t = positions.transpose(-1, -2)
        
        # 计算位置之间的距离
        distance = positions - positions_t
        
        # 创建滑动窗口掩码，窗口大小为window_size
        # 对于位置i，只有|i-j| <= window_size//2的位置j是可见的
        window_mask = torch.abs(distance) <= (self.window_size // 2)
        
        return window_mask
    
    def _select_global_tokens(self, seq_len, device):
        """选择全局tokens"""
        # 确定全局token数量
        num_global_tokens = max(1, int(seq_len * self.global_tokens_ratio))
        
        # 默认选择序列中的前几个token作为全局token（可以根据需要修改策略）
        global_token_indices = torch.arange(num_global_tokens, device=device)
        
        return global_token_indices
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        
        # 生成查询、键、值
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # 重塑维度以支持多头
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 创建滑动窗口掩码
        window_mask = self._create_sliding_window_mask(seq_len, device)
        
        # 选择全局tokens
        global_token_indices = self._select_global_tokens(seq_len, device)
        
        # 创建全局掩码
        # 全局token可以注意到所有token，所有token也可以注意到全局token
        global_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # 全局token可以看到所有token
        global_mask[global_token_indices, :] = True
        
        # 所有token都可以看到全局token
        global_mask[:, global_token_indices] = True
        
        # 合并窗口掩码和全局掩码
        combined_mask = torch.logical_or(window_mask, global_mask)
        
        # 扩展掩码维度以适应注意力计算
        # [1, 1, seq_len, seq_len]
        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)
        
        # 计算注意力分数 (q @ k.T / sqrt(head_size))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用Longformer掩码
        # 将不在mask中的位置设为很小的负数，使softmax后几乎为0
        attention_mask_for_scores = (~combined_mask) * -10000.0
        attention_scores = attention_scores + attention_mask_for_scores
        
        # 如果提供了额外的注意力掩码，也应用它
        if attention_mask is not None:
            # 调整掩码维度，扩展为[batch_size, 1, seq_len, seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(-1, -1, seq_len, -1)
            # 将掩码转换为float
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # 应用掩码
            attention_scores = attention_scores + extended_attention_mask
        
        # 应用softmax得到注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        
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

def replace_with_longformer_attention(model, window_size=128, global_tokens_ratio=0.1):
    """
    使用Longformer注意力机制替换模型中的标准自注意力
    
    Args:
        model: 需要修改的模型
        window_size: 局部滑动窗口大小
        global_tokens_ratio: 全局token的比例
    
    Returns:
        model: 修改后的模型
    """
    logger.info(f"替换为Longformer注意力机制: window_size={window_size}, global_tokens_ratio={global_tokens_ratio}")
    
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果是自注意力模块
        if "self_attn" in name and not "output" in name and hasattr(module, "query_key_value"):
            # 获取原始模块的配置
            hidden_size = module.hidden_size
            num_attention_heads = module.num_attention_heads
            
            # 创建Longformer注意力模块
            longformer_attn = LongformerAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                window_size=window_size,
                global_tokens_ratio=global_tokens_ratio
            )
            
            # 暂存原始的前向函数
            orig_forward = module.forward
            
            # 创建新的前向函数
            def new_forward(*args, **kwargs):
                # 调用Longformer注意力的前向函数
                return longformer_attn(*args, **kwargs)
            
            # 替换前向函数
            module.forward = new_forward
            
            logger.info(f"已替换模块 {name} 为Longformer注意力")
    
    return model 