"""
低秩分解注意力机制模块

该模块实现了一种使用奇异值分解（SVD）进行低秩近似的注意力机制。
通过将查询、键、值的权重矩阵分解为两个低秩矩阵的乘积，显著减少计算所需的显存资源。
"""

import math
import torch
import logging
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger("attn_experiment")

def get_low_rank_attention_config(rank_ratio=0.5):
    """
    获取低秩分解注意力机制配置
    
    Args:
        rank_ratio: 低秩比例，表示保留的奇异值比例，取值范围(0, 1)
    
    Returns:
        config: 配置字典
    """
    return {
        "name": "低秩分解注意力机制",
        "description": "使用SVD分解权重矩阵为两个低秩子矩阵的乘积",
        "type": "low_rank",
        "rank_ratio": rank_ratio
    }

def replace_with_low_rank_attention(model, rank_ratio=0.5):
    """
    将模型的注意力机制替换为低秩分解注意力机制
    
    Args:
        model: 原始模型
        rank_ratio: 低秩比例，表示保留的奇异值比例，取值范围(0, 1)
    
    Returns:
        model: 替换后的模型
    """
    logger.info(f"使用低秩分解注意力机制，rank_ratio: {rank_ratio}")
    
    # 遍历模型的所有层
    for name, module in model.named_modules():
        # 查找注意力模块
        if "self_attn" in name and hasattr(module, "compute_attention"):
            # 保存原始的计算注意力函数
            original_compute_attention = module.compute_attention
            
            # 保存原始权重尺寸和数据类型
            original_q_weight = module.q_proj.weight.data
            original_k_weight = module.k_proj.weight.data
            original_v_weight = module.v_proj.weight.data
            
            original_dtype = original_q_weight.dtype
            
            # 将权重转换为float32以进行SVD操作
            original_q_weight_fp32 = original_q_weight.to(torch.float32)
            original_k_weight_fp32 = original_k_weight.to(torch.float32)
            original_v_weight_fp32 = original_v_weight.to(torch.float32)
            
            in_features, out_features = original_q_weight.size()
            
            # 计算低秩维度
            low_rank_dim = max(1, int(min(in_features, out_features) * rank_ratio))
            
            # 使用SVD分解权重矩阵
            U_q, S_q, V_q = torch.svd(original_q_weight_fp32)
            U_k, S_k, V_k = torch.svd(original_k_weight_fp32)
            U_v, S_v, V_v = torch.svd(original_v_weight_fp32)
            
            # 截断到低秩维度
            U_q_low = U_q[:, :low_rank_dim]
            V_q_low = V_q[:, :low_rank_dim]
            S_q_low = S_q[:low_rank_dim]
            
            U_k_low = U_k[:, :low_rank_dim]
            V_k_low = V_k[:, :low_rank_dim]
            S_k_low = S_k[:low_rank_dim]
            
            U_v_low = U_v[:, :low_rank_dim]
            V_v_low = V_v[:, :low_rank_dim]
            S_v_low = S_v[:low_rank_dim]
            
            # 计算最终的分解矩阵
            # W_q ≈ U_q_low * diag(S_q_low) * V_q_low.T = (U_q_low * diag(sqrt(S_q_low))) * (V_q_low * diag(sqrt(S_q_low))).T
            sqrt_S_q = torch.sqrt(S_q_low).diag()
            U_q_final = torch.matmul(U_q_low, sqrt_S_q)
            V_q_final = torch.matmul(V_q_low, sqrt_S_q)
            
            sqrt_S_k = torch.sqrt(S_k_low).diag()
            U_k_final = torch.matmul(U_k_low, sqrt_S_k)
            V_k_final = torch.matmul(V_k_low, sqrt_S_k)
            
            sqrt_S_v = torch.sqrt(S_v_low).diag()
            U_v_final = torch.matmul(U_v_low, sqrt_S_v)
            V_v_final = torch.matmul(V_v_low, sqrt_S_v)
            
            # 将分解后的矩阵转换回原始数据类型
            U_q_final = U_q_final.to(original_dtype)
            V_q_final = V_q_final.to(original_dtype)
            U_k_final = U_k_final.to(original_dtype)
            V_k_final = V_k_final.to(original_dtype)
            U_v_final = U_v_final.to(original_dtype)
            V_v_final = V_v_final.to(original_dtype)
            
            # 定义新的计算注意力函数
            def low_rank_compute_attention(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                output_attentions=False,
            ):
                # 保存原始的查询、键、值状态
                original_query_states = query_states
                original_key_states = key_states
                original_value_states = value_states
                
                # 获取形状信息
                batch_size, num_heads, seq_len, head_dim = query_states.shape
                
                # 重塑为二维张量以便于矩阵乘法
                query_states_2d = query_states.transpose(1, 2).reshape(batch_size * seq_len, -1)
                key_states_2d = key_states.transpose(1, 2).reshape(batch_size * seq_len, -1)
                value_states_2d = value_states.transpose(1, 2).reshape(batch_size * seq_len, -1)
                
                # 应用低秩分解
                # Q = W_q * x ≈ U_q * (V_q * x)
                query_mid = F.linear(query_states_2d, V_q_final.t())
                query_states_2d = F.linear(query_mid, U_q_final.t())
                
                # K = W_k * x ≈ U_k * (V_k * x)
                key_mid = F.linear(key_states_2d, V_k_final.t())
                key_states_2d = F.linear(key_mid, U_k_final.t())
                
                # V = W_v * x ≈ U_v * (V_v * x)
                value_mid = F.linear(value_states_2d, V_v_final.t())
                value_states_2d = F.linear(value_mid, U_v_final.t())
                
                # 重塑回原始形状
                query_states = query_states_2d.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                key_states = key_states_2d.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                value_states = value_states_2d.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                
                # 使用原始的注意力计算函数计算注意力
                return original_compute_attention(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    output_attentions,
                )
            
            # 替换计算注意力函数
            module.compute_attention = low_rank_compute_attention.__get__(module, type(module))
            
            logger.info(f"已替换模块 {name} 的注意力计算为低秩分解模式，rank_ratio={rank_ratio}, low_rank_dim={low_rank_dim}")
    
    return model

class LowRankSelfAttention(torch.nn.Module):
    """
    低秩分解自注意力机制
    """
    
    def __init__(self, config, rank_ratio=0.5):
        """
        初始化
        
        Args:
            config: 配置
            rank_ratio: 低秩比例，表示保留的奇异值比例，取值范围(0, 1)
        """
        super().__init__()
        self.config = config
        self.rank_ratio = rank_ratio
        
        # 创建模型参数
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # 计算低秩维度
        in_features = self.hidden_size
        out_features = self.hidden_size
        self.low_rank_dim = max(1, int(min(in_features, out_features) * rank_ratio))
        
        # 创建查询、键、值的低秩投影矩阵
        self.q_proj_U = torch.nn.Linear(self.low_rank_dim, out_features, bias=False)
        self.q_proj_V = torch.nn.Linear(in_features, self.low_rank_dim, bias=False)
        
        self.k_proj_U = torch.nn.Linear(self.low_rank_dim, out_features, bias=False)
        self.k_proj_V = torch.nn.Linear(in_features, self.low_rank_dim, bias=False)
        
        self.v_proj_U = torch.nn.Linear(self.low_rank_dim, out_features, bias=False)
        self.v_proj_V = torch.nn.Linear(in_features, self.low_rank_dim, bias=False)
        
        # 输出投影
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
    
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
        
        # 应用低秩投影: W_q * x ≈ U_q * (V_q * x)
        query_mid_states = self.q_proj_V(hidden_states)
        query_states = self.q_proj_U(query_mid_states)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        key_mid_states = self.k_proj_V(hidden_states)
        key_states = self.k_proj_U(key_mid_states)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        value_mid_states = self.v_proj_V(hidden_states)
        value_states = self.v_proj_U(value_mid_states)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
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
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # 应用softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        outputs = (attn_output, attn_weights, past_key_value)
        return outputs 