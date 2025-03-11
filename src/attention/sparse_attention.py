"""
稀疏注意力机制模块
"""

import math
import torch
import logging
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger("attn_experiment")

def get_sparse_attention_config(sparsity=0.8):
    """
    获取稀疏注意力机制配置
    
    Args:
        sparsity: 稀疏度，表示保留的注意力比例
    
    Returns:
        config: 配置字典
    """
    return {
        "name": "Sparse Attention",
        "description": "修改RotaryEmbedding模块并集成动态稀疏掩码",
        "type": "sparse",
        "sparsity": sparsity
    }

def replace_with_sparse_attention(model, sparsity=0.8):
    """
    将模型的注意力机制替换为稀疏注意力机制
    
    Args:
        model: 原始模型
        sparsity: 稀疏度，表示保留的注意力比例
    
    Returns:
        model: 替换后的模型
    """
    logger.info(f"使用稀疏注意力机制，稀疏度: {sparsity}")
    
    # 遍历模型的所有层
    for name, module in model.named_modules():
        # 查找注意力模块
        if "self_attn" in name and hasattr(module, "compute_attention"):
            # 保存原始的计算注意力函数
            original_compute_attention = module.compute_attention
            
            # 定义新的计算注意力函数
            def sparse_compute_attention(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                output_attentions=False,
            ):
                # 计算注意力分数
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
                    query_states.size(-1)
                )
                
                # 应用注意力掩码
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                
                # 创建稀疏掩码
                batch_size, num_heads, seq_len, _ = attn_weights.shape
                
                # 对每个头的注意力分数进行排序
                sorted_weights, _ = torch.sort(attn_weights, dim=-1, descending=True)
                
                # 计算阈值（保留前k个最大值）
                k = int(seq_len * (1 - sparsity))
                if k < 1:
                    k = 1
                
                # 获取阈值
                threshold = sorted_weights[:, :, :, k-1:k]
                
                # 创建稀疏掩码
                sparse_mask = (attn_weights < threshold).float() * -10000.0
                
                # 应用稀疏掩码
                attn_weights = attn_weights + sparse_mask
                
                # 应用softmax
                attn_weights = F.softmax(attn_weights, dim=-1)
                
                # 计算输出
                attn_output = torch.matmul(attn_weights, value_states)
                
                outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
                return outputs
            
            # 替换计算注意力函数
            module.compute_attention = sparse_compute_attention.__get__(module, type(module))
            
            logger.info(f"已替换模块 {name} 的注意力计算函数")
    
    return model

class SparseSelfAttention(torch.nn.Module):
    """
    稀疏自注意力机制
    """
    
    def __init__(self, config, sparsity=0.8):
        """
        初始化
        
        Args:
            config: 配置
            sparsity: 稀疏度，表示保留的注意力比例
        """
        super().__init__()
        self.config = config
        self.sparsity = sparsity
        
        # 创建查询、键、值投影
        self.q_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: 输入隐藏状态
            attention_mask: 注意力掩码
            position_ids: 位置编码
            past_key_value: 过去的键值对
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用缓存
        
        Returns:
            attn_output: 注意力输出
            attn_weights: 注意力权重
            past_key_value: 更新后的键值对
        """
        batch_size, seq_len = hidden_states.shape[:2]
        
        # 投影查询、键、值
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 处理过去的键值对
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # 如果使用缓存，保存当前的键值对
        if use_cache:
            past_key_value = (key_states, value_states)
        
        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # 创建稀疏掩码
        seq_len = key_states.size(2)
        
        # 对每个头的注意力分数进行排序
        sorted_weights, _ = torch.sort(attn_weights, dim=-1, descending=True)
        
        # 计算阈值（保留前k个最大值）
        k = int(seq_len * (1 - self.sparsity))
        if k < 1:
            k = 1
        
        # 获取阈值
        threshold = sorted_weights[:, :, :, k-1:k]
        
        # 创建稀疏掩码
        sparse_mask = (attn_weights < threshold).float() * -10000.0
        
        # 应用稀疏掩码
        attn_weights = attn_weights + sparse_mask
        
        # 应用softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output, attn_weights, past_key_value) if output_attentions else (attn_output, None, past_key_value)
        return outputs 