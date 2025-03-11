"""
注意力机制工具模块，用于统一管理不同的注意力机制
"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("attn_experiment")

# 导入注意力机制
from src.attention.standard_attention import (
    get_standard_attention_config, replace_with_standard_attention
)
from src.attention.sparse_attention import (
    get_sparse_attention_config, replace_with_sparse_attention
)
from src.attention.linear_attention import (
    get_linear_attention_config, replace_with_linear_attention
)

def get_attention_config(attn_type, **kwargs):
    """
    获取注意力机制配置
    
    Args:
        attn_type: 注意力机制类型，可选值为"standard", "sparse", "linear"
        **kwargs: 其他参数
    
    Returns:
        config: 配置字典
    """
    if attn_type == "standard":
        return get_standard_attention_config()
    
    elif attn_type == "sparse":
        sparsity = kwargs.get("sparsity", 0.8)
        return get_sparse_attention_config(sparsity=sparsity)
    
    elif attn_type == "linear":
        kernel_function = kwargs.get("kernel_function", "elu")
        return get_linear_attention_config(kernel_function=kernel_function)
    
    else:
        raise ValueError(f"不支持的注意力机制类型: {attn_type}")

def replace_attention_mechanism(model, attn_type, **kwargs):
    """
    替换模型的注意力机制
    
    Args:
        model: 原始模型
        attn_type: 注意力机制类型，可选值为"standard", "sparse", "linear"
        **kwargs: 其他参数
    
    Returns:
        model: 替换后的模型
    """
    if attn_type == "standard":
        return replace_with_standard_attention(model)
    
    elif attn_type == "sparse":
        sparsity = kwargs.get("sparsity", 0.8)
        return replace_with_sparse_attention(model, sparsity=sparsity)
    
    elif attn_type == "linear":
        kernel_function = kwargs.get("kernel_function", "elu")
        return replace_with_linear_attention(model, kernel_function=kernel_function)
    
    else:
        raise ValueError(f"不支持的注意力机制类型: {attn_type}")

def get_attention_info(model, attn_type):
    """
    获取模型的注意力机制信息
    
    Args:
        model: 模型
        attn_type: 注意力机制类型
    
    Returns:
        info: 注意力机制信息字典
    """
    # 获取注意力机制配置
    attn_config = get_attention_config(attn_type)
    
    # 构建信息字典
    info = {
        "attention_type": attn_type,
        "attention_name": attn_config["name"],
        "attention_description": attn_config["description"],
    }
    
    # 添加特定类型的信息
    if attn_type == "sparse":
        info["sparsity"] = attn_config.get("sparsity", 0.8)
    
    elif attn_type == "linear":
        info["kernel_function"] = attn_config.get("kernel_function", "elu")
    
    return info 