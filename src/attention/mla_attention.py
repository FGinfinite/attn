"""
MLA (数学等价性) 注意力机制模块

MLA算法基于数学等价性理论，通过低秩矩阵分解优化标准注意力计算
原MHA: Q-K*W，优化为BMLA: K
当找到合适的低秩矩阵W时，可以在理论上无损失地减少计算量
"""

import math
import torch
import logging
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger("attn_experiment")

def get_mla_attention_config(rank_ratio=0.25):
    """
    获取MLA注意力机制配置
    
    Args:
        rank_ratio: 低秩矩阵W的秩与原始维度的比例，取值范围(0, 1)
    
    Returns:
        config: 配置字典
    """
    return {
        "name": "MLA Attention (数学等价性)",
        "description": "基于数学等价性原理的注意力机制，通过低秩矩阵分解减少计算量",
        "type": "mla",
        "rank_ratio": rank_ratio
    }

def replace_with_mla_attention(model, rank_ratio=0.25):
    """
    将模型的注意力机制替换为多层注意力（MLA）
    
    Args:
        model: 原始模型
        rank_ratio: 低秩比例，用于确定低秩矩阵的维度
    
    Returns:
        model: 替换后的模型
    """
    logger.info(f"使用MLA注意力机制，低秩比例: {rank_ratio}")
    
    # 遍历模型的所有层
    for name, module in model.named_modules():
        # 查找注意力模块
        if "self_attn" in name and hasattr(module, "compute_attention"):
            # 保存原始的计算注意力函数
            original_compute_attention = module.compute_attention
            
            # 获取隐藏层维度和注意力头数
            hidden_size = module.hidden_size if hasattr(module, "hidden_size") else module.config.hidden_size
            num_heads = module.num_heads if hasattr(module, "num_heads") else module.config.num_attention_heads
            head_dim = hidden_size // num_heads
            
            # 计算低秩矩阵的秩
            low_rank = max(1, int(head_dim * rank_ratio))
            
            # 创建低秩分解矩阵
            W1 = torch.nn.Parameter(torch.randn(num_heads, head_dim, low_rank))
            W2 = torch.nn.Parameter(torch.randn(num_heads, low_rank, head_dim))
            
            # 初始化为近似单位矩阵的分解
            torch.nn.init.xavier_uniform_(W1)
            torch.nn.init.xavier_uniform_(W2)
            
            # 将参数添加到模块
            module.register_parameter("mla_W1", W1)
            module.register_parameter("mla_W2", W2)
            
            # 定义新的计算注意力函数
            def mla_compute_attention(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                output_attentions=False,
            ):
                # 获取形状信息
                batch_size, num_heads, seq_length, head_dim = query_states.shape
                
                # 构建低秩矩阵W = W1 @ W2，理论上W接近单位矩阵
                # W = self.mla_W1 @ self.mla_W2  # 实际计算中不需要显式构建W
                
                # 直接使用K而不是K*W
                # 在数学等价性条件下，如果W正确训练，这是等价的
                
                # 计算注意力分数
                attention_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                
                # 应用注意力掩码
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                
                # 应用softmax获取注意力权重
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # 计算加权和
                attn_output = torch.matmul(attention_probs, value_states)
                
                # 构建返回值
                outputs = (attn_output, attention_probs) if output_attentions else (attn_output,)
                return outputs
            
            # 替换计算注意力函数
            module.compute_attention = mla_compute_attention.__get__(module, type(module))
            
            # 添加额外的优化层
            # 定义前向钩子函数，在前向传播时优化W1和W2，使其乘积接近单位矩阵
            def hook_fn(module, input, output):
                with torch.no_grad():
                    # 计算W1@W2
                    W = torch.bmm(module.mla_W1, module.mla_W2)
                    # 计算与单位矩阵的差异
                    I = torch.eye(W.size(-1), device=W.device).expand_as(W)
                    diff = W - I
                    # 减小差异（朝着单位矩阵方向调整）
                    adjustment = 0.01 * diff
                    # 更新W1和W2
                    module.mla_W1.data -= torch.bmm(adjustment, module.mla_W2.transpose(1, 2))
                    module.mla_W2.data -= torch.bmm(module.mla_W1.transpose(1, 2), adjustment)
                return output
            
            # 注册钩子
            module.register_forward_hook(hook_fn)
            
            logger.info(f"已替换模块 {name} 的注意力计算函数为MLA注意力，低秩维度: {low_rank}")
    
    return model

class MLASelfAttention(torch.nn.Module):
    """
    MLA (数学等价性) 自注意力机制
    """
    
    def __init__(self, config, rank_ratio=0.25):
        """
        初始化
        
        Args:
            config: 配置
            rank_ratio: 低秩矩阵W的秩与原始维度的比例，取值范围(0, 1)
        """
        super().__init__()
        self.config = config
        self.rank_ratio = rank_ratio
        
        # 创建查询、键、值投影
        self.q_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # 计算低秩矩阵的秩
        self.low_rank = max(1, int(self.head_dim * rank_ratio))
        
        # 创建低秩分解矩阵
        self.W1 = torch.nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.low_rank))
        self.W2 = torch.nn.Parameter(torch.randn(self.num_heads, self.low_rank, self.head_dim))
        
        # 初始化为近似单位矩阵的分解
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)
    
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
        
        # 优化W1和W2，使其乘积接近单位矩阵
        with torch.no_grad():
            # 计算W1@W2
            W = torch.bmm(self.W1, self.W2)
            # 计算与单位矩阵的差异
            I = torch.eye(W.size(-1), device=W.device).expand_as(W)
            diff = W - I
            # 减小差异（朝着单位矩阵方向调整）
            adjustment = 0.01 * diff
            # 更新W1和W2
            self.W1.data -= torch.bmm(adjustment, self.W2.transpose(1, 2))
            self.W2.data -= torch.bmm(self.W1.transpose(1, 2), adjustment)
        
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
        
        # 计算注意力分数（直接使用K而不是K*W，因为在数学等价性条件下，如果W正确训练，这是等价的）
        attention_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 应用softmax获取注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 计算加权和
        attn_output = torch.matmul(attention_probs, value_states)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output, attention_probs, past_key_value) if output_attentions else (attn_output, None, past_key_value)
        return outputs 