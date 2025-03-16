"""
Reformer注意力实现模块
通过局部敏感哈希(LSH)实现高效的注意力计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger("attn_experiment")

def get_reformer_attention_config(num_hashes=4):
    """
    获取Reformer注意力机制配置
    
    Args:
        num_hashes: 哈希函数数量
    
    Returns:
        config: 配置字典
    """
    return {
        "name": "Reformer Attention",
        "description": "通过局部敏感哈希(LSH)实现高效的注意力计算",
        "num_hashes": num_hashes
    }

def _lsh_projection(query, num_hashes, seed=None):
    """
    使用局部敏感哈希将查询投影到多个哈希桶
    
    Args:
        query: 查询张量 [batch_size, seq_len, num_heads, head_dim]
        num_hashes: 哈希函数数量
        seed: 随机种子，确保每次投影结果一致
    
    Returns:
        buckets: 哈希桶索引 [batch_size, seq_len, num_heads, num_hashes]
    """
    batch_size, seq_len, num_heads, head_dim = query.shape
    
    # 为每个哈希创建随机投影向量
    if seed is not None:
        torch.manual_seed(seed)
    
    # 创建随机投影矩阵
    random_vectors = torch.randn(num_heads, num_hashes, head_dim, 
                               device=query.device).type_as(query)
    random_vectors = F.normalize(random_vectors, dim=-1)  # 单位长度
    
    # 用随机投影向量计算点积
    # [batch_size, seq_len, num_heads, head_dim] x [num_heads, num_hashes, head_dim]
    # -> [batch_size, seq_len, num_heads, num_hashes]
    projection = torch.einsum("bsnd,nhd->bsnh", query, random_vectors)
    
    # 将投影结果转换为桶索引
    # 根据投影值的符号(+/-)分桶，相当于超平面划分空间
    projection_bin = torch.sign(projection)  # [-1, 1]
    
    # 将二进制编码转换为桶索引: 每个位取值{-1,1}，可以编码为整数
    indices = torch.sum(projection_bin * torch.arange(1, num_hashes + 1, 
                                                    device=query.device).view(1, 1, 1, -1), 
                       dim=-1)  # [batch_size, seq_len, num_heads]
    
    return indices

class LSHSelfAttention(nn.Module):
    """基于LSH的自注意力层"""
    def __init__(self, hidden_size, num_attention_heads, num_hashes=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hashes = num_hashes
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 用于计算注意力权重的查询、键、值投影
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # 输出投影
        self.output = nn.Linear(self.all_head_size, hidden_size)
        
        # 固定随机种子，确保每次投影一致
        self.hash_seed = 42
        
        logger.info(f"初始化Reformer注意力: hidden_size={hidden_size}, heads={num_attention_heads}, num_hashes={num_hashes}")
    
    def transpose_for_scores(self, x):
        """将张量重塑为[batch_size, seq_len, num_heads, head_size]"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x
    
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
        
        # 对查询和键进行LSH哈希投影
        # [batch_size, seq_len, num_heads, num_hashes]
        query_buckets = _lsh_projection(query_layer, self.num_hashes, self.hash_seed)
        key_buckets = _lsh_projection(key_layer, self.num_hashes, self.hash_seed)
        
        # 创建相同hash桶的注意力掩码
        # 只有在相同hash桶中的token才会相互计算注意力
        attention_mask_lsh = torch.eq(query_buckets.unsqueeze(-1), 
                                    key_buckets.unsqueeze(-2))  # [batch, seq_len, num_heads, seq_len]
        
        # 合并所有哈希的结果（任意一个哈希函数将两个token放入相同桶，都认为它们可能相关）
        attention_mask_lsh = attention_mask_lsh.any(dim=-1)  # [batch, seq_len, num_heads, seq_len]
        
        # 应用原始注意力掩码（如果提供）
        if attention_mask is not None:
            # [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # 确保attention_mask与lsh_mask维度匹配后相乘
            attention_mask_lsh = attention_mask_lsh & (attention_mask > 0)
        
        # 将布尔掩码转换为浮点掩码
        attention_mask_float = torch.zeros_like(attention_mask_lsh, dtype=torch.float)
        attention_mask_float.masked_fill_(~attention_mask_lsh, -10000.0)
        attention_mask_float = attention_mask_float.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, seq_len]
        
        # 计算注意力分数
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(
            query_layer.permute(0, 2, 1, 3), 
            key_layer.permute(0, 2, 3, 1)
        ) / math.sqrt(self.attention_head_size)
        
        # 应用LSH注意力掩码
        attention_scores = attention_scores + attention_mask_float
        
        # 应用softmax获得注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 应用头掩码（如果提供）
        if head_mask is not None:
            attention_probs = attention_probs * head_mask.unsqueeze(1).unsqueeze(1)
        
        # 计算加权和
        context_layer = torch.matmul(attention_probs, value_layer.permute(0, 2, 1, 3))
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # 合并头
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 投影到输出维度
        outputs = self.output(context_layer)
        
        if output_attentions:
            return outputs, attention_probs
        
        return outputs

class ReformerAttention(nn.Module):
    """
    Reformer自注意力模块封装，便于替换现有注意力
    """
    def __init__(self, hidden_size, num_attention_heads, num_hashes=4):
        super().__init__()
        self.lsh_attention = LSHSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hashes=num_hashes
        )
    
    def forward(self, *args, **kwargs):
        return self.lsh_attention(*args, **kwargs)

def replace_with_reformer_attention(model, num_hashes=4, last_layer_only=False):
    """
    将模型的注意力机制替换为Reformer注意力机制
    
    Args:
        model: 原始模型
        num_hashes: LSH哈希数量
        last_layer_only: 是否只替换最后一层注意力，默认为False
    
    Returns:
        model: 替换后的模型
    """
    logger.info(f"替换为Reformer注意力机制: num_hashes={num_hashes}")
    
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果是自注意力模块
        if "self_attn" in name and not "output" in name and hasattr(module, "query_key_value"):
            # 获取原始模块的配置
            hidden_size = module.hidden_size
            num_attention_heads = module.num_attention_heads
            
            # 创建Reformer注意力模块
            reformer_attn = ReformerAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hashes=num_hashes
            )
            
            # 暂存原始的前向函数
            orig_forward = module.forward
            
            # 创建新的前向函数
            def new_forward(*args, **kwargs):
                # 调用Reformer注意力的前向函数
                return reformer_attn(*args, **kwargs)
            
            # 替换前向函数
            module.forward = new_forward
            
            logger.info(f"已替换模块 {name} 为Reformer注意力")
    
    return model 