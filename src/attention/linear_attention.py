"""
线性注意力机制模块
"""

import math
import torch
import logging
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger("attn_experiment")

def get_linear_attention_config(kernel_function="elu"):
    """
    获取线性注意力机制配置
    
    Args:
        kernel_function: 核函数，可选值为"elu", "relu", "softmax"
    
    Returns:
        config: 配置字典
    """
    return {
        "name": "Linear Attention",
        "description": "基于线性注意力公式重写计算逻辑",
        "type": "linear",
        "kernel_function": kernel_function
    }

def replace_with_linear_attention(model, kernel_function="elu", last_layer_only=False):
    """
    将模型的注意力机制替换为线性注意力机制
    
    Args:
        model: 原始模型
        kernel_function: 核函数，可选值为"elu", "relu", "softmax"
        last_layer_only: 是否只替换最后一层注意力，默认为False
    
    Returns:
        model: 替换后的模型
    """
    logger.info(f"使用线性注意力机制，核函数: {kernel_function}")
    if last_layer_only:
        logger.info(f"注意：只替换最后一层注意力机制")
    
    # 定义核函数
    def apply_kernel(x, kernel_type):
        if kernel_type == "elu":
            return F.elu(x) + 1.0
        elif kernel_type == "relu":
            return F.relu(x)
        elif kernel_type == "softmax":
            return F.softmax(x, dim=-1)
        else:
            raise ValueError(f"不支持的核函数: {kernel_type}")
    
    # 收集所有注意力模块
    attention_modules = []
    for name, module in model.named_modules():
        # 查找注意力模块
        if any(attn_name in name.lower() for attn_name in ["self_attn", "sdpaattention", "attention"]) and hasattr(module, "compute_attention"):
            attention_modules.append((name, module))
            logger.debug(f"发现注意力模块: {name}")
            
    if not attention_modules:
        logger.warning("未找到任何符合条件的注意力模块!")
        return model
    
    # 确定要替换的模块
    if last_layer_only and attention_modules:
        # 只处理最后一个注意力模块
        modules_to_replace = [attention_modules[-1]]
        name, _ = modules_to_replace[0]
        logger.info(f"将替换最后一层注意力模块: {name}")
    else:
        # 处理所有注意力模块
        modules_to_replace = attention_modules
        logger.info(f"将替换所有 {len(modules_to_replace)} 个注意力模块")
    
    # 替换选定的模块
    replaced_count = 0
    for name, module in modules_to_replace:
        # 保存原始的计算注意力函数
        original_compute_attention = module.compute_attention
        
        # 定义新的计算注意力函数
        def linear_compute_attention(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            output_attentions=False,
        ):
            # 添加注意力类型标记
            self._attn_type = "linear"
            self._kernel_function = kernel_function
            
            # 应用核函数
            query_prime = apply_kernel(query_states, kernel_function)
            key_prime = apply_kernel(key_states, kernel_function)
            
            # 计算KV矩阵
            kv = torch.matmul(key_prime.transpose(2, 3), value_states)
            
            # 计算分母
            z = torch.matmul(query_prime, torch.sum(key_prime, dim=2).unsqueeze(-1))
            
            # 计算线性注意力输出
            attn_output = torch.matmul(query_prime, kv) / (z + 1e-6)
            
            # 如果需要输出注意力权重，计算标准注意力权重
            if output_attentions:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
                    query_states.size(-1)
                )
                
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                
                attn_weights = F.softmax(attn_weights, dim=-1)
            else:
                attn_weights = None
            
            outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
            return outputs
        
        # 添加标记属性
        module._attn_type = "linear"
        module._kernel_function = kernel_function
        
        # 替换计算注意力函数
        try:
            module.compute_attention = linear_compute_attention.__get__(module, type(module))
            replaced_count += 1
            logger.info(f"已替换模块 {name} 的注意力计算函数")
        except Exception as e:
            logger.warning(f"替换模块 {name} 失败: {str(e)}")
    
    if replaced_count > 0:
        if last_layer_only:
            logger.info(f"已成功替换最后一层注意力模块")
        else:
            logger.info(f"已成功替换 {replaced_count} 个注意力模块")
    else:
        logger.warning("未能替换任何注意力模块!")
    
    return model

class LinearSelfAttention(torch.nn.Module):
    """
    线性自注意力机制
    """
    
    def __init__(self, config, kernel_function="elu"):
        """
        初始化
        
        Args:
            config: 配置
            kernel_function: 核函数，可选值为"elu", "relu", "softmax"
        """
        super().__init__()
        self.config = config
        self.kernel_function = kernel_function
        
        # 创建查询、键、值投影
        self.q_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
    
    def apply_kernel(self, x):
        """
        应用核函数
        
        Args:
            x: 输入张量
        
        Returns:
            x_prime: 应用核函数后的张量
        """
        if self.kernel_function == "elu":
            return F.elu(x) + 1.0
        elif self.kernel_function == "relu":
            return F.relu(x)
        elif self.kernel_function == "softmax":
            return F.softmax(x, dim=-1)
        else:
            raise ValueError(f"不支持的核函数: {self.kernel_function}")
    
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
        
        # 应用核函数
        query_prime = self.apply_kernel(query_states)
        key_prime = self.apply_kernel(key_states)
        
        # 计算KV矩阵
        kv = torch.matmul(key_prime.transpose(2, 3), value_states)
        
        # 计算分母
        z = torch.matmul(query_prime, torch.sum(key_prime, dim=2).unsqueeze(-1))
        
        # 计算线性注意力输出
        attn_output = torch.matmul(query_prime, kv) / (z + 1e-6)
        
        # 如果需要输出注意力权重，计算标准注意力权重
        if output_attentions:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            attn_weights = None
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output, attn_weights, past_key_value) if output_attentions else (attn_output, None, past_key_value)
        return outputs 