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
# 导入新增的注意力机制
from src.attention.reformer_attention import (
    get_reformer_attention_config, replace_with_reformer_attention
)
from src.attention.linformer_attention import (
    get_linformer_attention_config, replace_with_linformer_attention
)
from src.attention.longformer_attention import (
    get_longformer_attention_config, replace_with_longformer_attention
)
from src.attention.realformer_attention import (
    get_realformer_attention_config, replace_with_realformer_attention
)
# 导入MLA注意力机制
from src.attention.mla_attention import (
    get_mla_attention_config, replace_with_mla_attention
)

def get_attention_config(attn_type, **kwargs):
    """
    获取注意力机制配置
    
    Args:
        attn_type: 注意力机制类型，可选值为"standard", "sparse", "linear", "reformer", "linformer", "longformer", "realformer", "mla"
        **kwargs: 其他参数
    
    Returns:
        config: 配置字典
    """
    # 配置字典中添加替换最后一层的标记
    base_config = {"last_layer_only": kwargs.get("last_layer_only", False)}
    
    if attn_type == "standard":
        config = get_standard_attention_config()
        config.update(base_config)
        return config
    
    elif attn_type == "sparse":
        sparsity = kwargs.get("sparsity", 0.8)
        config = get_sparse_attention_config(sparsity=sparsity)
        config.update(base_config)
        return config
    
    elif attn_type == "linear":
        kernel_function = kwargs.get("kernel_function", "elu")
        config = get_linear_attention_config(kernel_function=kernel_function)
        config.update(base_config)
        return config
    
    elif attn_type == "reformer":
        num_hashes = kwargs.get("num_hashes", 4)
        config = get_reformer_attention_config(num_hashes=num_hashes)
        config.update(base_config)
        return config
    
    elif attn_type == "linformer":
        k_ratio = kwargs.get("k_ratio", 0.25)
        config = get_linformer_attention_config(k_ratio=k_ratio)
        config.update(base_config)
        return config
    
    elif attn_type == "longformer":
        window_size = kwargs.get("window_size", 128)
        global_tokens_ratio = kwargs.get("global_tokens_ratio", 0.1)
        config = get_longformer_attention_config(
            window_size=window_size,
            global_tokens_ratio=global_tokens_ratio
        )
        config.update(base_config)
        return config
    
    elif attn_type == "realformer":
        config = get_realformer_attention_config()
        config.update(base_config)
        return config
    
    elif attn_type == "mla":
        rank_ratio = kwargs.get("rank_ratio", 0.25)
        config = get_mla_attention_config(rank_ratio=rank_ratio)
        config.update(base_config)
        return config
    
    elif attn_type == "low_rank":
        rank_ratio = kwargs.get("rank_ratio")
        return get_low_rank_attention_config(rank_ratio=rank_ratio)
    
    else:
        raise ValueError(f"不支持的注意力机制类型: {attn_type}")

def replace_attention_mechanism(model, attn_type, **kwargs):
    """
    替换模型的注意力机制
    
    Args:
        model: 原始模型
        attn_type: 注意力机制类型，可选值为"standard", "sparse", "linear", "reformer", "linformer", "longformer", "realformer", "mla"
        last_layer_only: 是否只替换最后一层注意力，默认为False
        **kwargs: 其他参数
    
    Returns:
        model: 替换后的模型
    """
    # 获取是否只替换最后一层的参数
    last_layer_only = kwargs.get("last_layer_only", False)
    
    # 如果指定了只替换最后一层，添加到日志
    if last_layer_only:
        logger.info(f"注意: 只替换最后一层注意力机制为 {attn_type} 类型")
    
    if attn_type == "standard":
        return replace_with_standard_attention(model, last_layer_only=last_layer_only)
    
    elif attn_type == "sparse":
        sparsity = kwargs.get("sparsity", 0.8)
        return replace_with_sparse_attention(model, sparsity=sparsity, last_layer_only=last_layer_only)
    
    elif attn_type == "linear":
        kernel_function = kwargs.get("kernel_function", "elu")
        return replace_with_linear_attention(model, kernel_function=kernel_function, last_layer_only=last_layer_only)
    
    elif attn_type == "reformer":
        num_hashes = kwargs.get("num_hashes", 4)
        return replace_with_reformer_attention(model, num_hashes=num_hashes, last_layer_only=last_layer_only)
    
    elif attn_type == "linformer":
        k_ratio = kwargs.get("k_ratio", 0.25)
        max_seq_length = kwargs.get("max_seq_length", 512)
        return replace_with_linformer_attention(
            model, 
            k_ratio=k_ratio,
            max_seq_length=max_seq_length,
            last_layer_only=last_layer_only
        )
    
    elif attn_type == "longformer":
        window_size = kwargs.get("window_size", 128)
        global_tokens_ratio = kwargs.get("global_tokens_ratio", 0.1)
        return replace_with_longformer_attention(
            model, 
            window_size=window_size,
            global_tokens_ratio=global_tokens_ratio,
            last_layer_only=last_layer_only
        )
    
    elif attn_type == "realformer":
        return replace_with_realformer_attention(model, last_layer_only=last_layer_only)
    
    elif attn_type == "mla":
        rank_ratio = kwargs.get("rank_ratio", 0.25)
        return replace_with_mla_attention(model, rank_ratio=rank_ratio, last_layer_only=last_layer_only)
    
    elif attn_type == "low_rank":
        rank_ratio = kwargs.get("rank_ratio")
        return replace_with_low_rank_attention(model, rank_ratio=rank_ratio)
    
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
    
    elif attn_type == "reformer":
        info["num_hashes"] = attn_config.get("num_hashes", 4)
    
    elif attn_type == "linformer":
        info["k_ratio"] = attn_config.get("k_ratio", 0.25)
    
    elif attn_type == "longformer":
        info["window_size"] = attn_config.get("window_size", 128)
        info["global_tokens_ratio"] = attn_config.get("global_tokens_ratio", 0.1)
    
    elif attn_type == "mla":
        info["rank_ratio"] = attn_config.get("rank_ratio", 0.25)
    
    # 添加是否只替换最后一层的信息
    info["last_layer_only"] = attn_config.get("last_layer_only", False)
    
    return info 