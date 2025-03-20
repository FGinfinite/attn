"""
注意力机制工具模块，用于统一管理不同的注意力机制
"""

import logging

# 导入自定义注意力机制（用于测试）
from src.attention.custom_attention import (
    get_custom_attention_config,
    replace_with_custom_attention,
)
from src.attention.linear_attention import (
    get_linear_attention_config,
    replace_with_linear_attention,
)
from src.attention.linformer_attention import (
    get_linformer_attention_config,
    replace_with_linformer_attention,
)
from src.attention.longformer_attention import (
    get_longformer_attention_config,
    replace_with_longformer_attention,
)

# 导入MLA注意力机制
from src.attention.mla_attention import (
    get_mla_attention_config,
    replace_with_mla_attention,
)
from src.attention.realformer_attention import (
    get_realformer_attention_config,
    replace_with_realformer_attention,
)

# 导入新增的注意力机制
from src.attention.reformer_attention import (
    get_reformer_attention_config,
    replace_with_reformer_attention,
)
from src.attention.sparse_attention import (
    get_sparse_attention_config,
    replace_with_sparse_attention,
)

# 导入注意力机制
from src.attention.standard_attention import get_standard_attention_config

logger = logging.getLogger("attn_experiment")


def get_attention_config(attn_type, **kwargs):
    """
    获取注意力机制配置

    Args:
        attn_type: 注意力机制类型，可选值为"standard", "sparse", "linear", "reformer", "linformer", "longformer", "realformer", "mla", "custom"
        **kwargs: 其他参数

    Returns:
        config: 配置字典
    """
    # 配置字典中添加替换最后一层的标记
    base_config = {}

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
            window_size=window_size, global_tokens_ratio=global_tokens_ratio
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

    elif attn_type == "custom":
        noise_level = kwargs.get("noise_level", 0.01)
        config = get_custom_attention_config(noise_level=noise_level)
        config.update(base_config)
        return config

    else:
        raise ValueError(f"不支持的注意力机制类型: {attn_type}")


def replace_attention_mechanism(model, attn_type, **kwargs):
    """
    替换模型的注意力机制 - 针对Qwen2模型优化的版本

    Args:
        model: 原始模型
        attn_type: 注意力机制类型，可选值为"standard", "sparse", "linear", "reformer", "linformer", "longformer", "realformer", "mla", "custom"
        **kwargs: 其他参数

    Returns:
        model: 替换后的模型
    """

    # 检查模型是否包含Qwen2Attention层
    has_qwen2_attention = False
    for name, module in model.named_modules():
        if (
            hasattr(module, "self_attn")
            and hasattr(module.self_attn, "__class__")
            and module.self_attn.__class__.__name__ == "Qwen2Attention"
        ):
            has_qwen2_attention = True
            break

    if has_qwen2_attention:
        logger.info("检测到Qwen2模型，使用针对Qwen2的注意力替换方法")
        return replace_qwen2_attention_mechanism(model, attn_type, **kwargs)
    else:
        raise ValueError("未检测到Qwen2模型，无法进行注意力机制替换")


def replace_qwen2_attention_mechanism(model, attn_type, **kwargs):
    """
    专门针对Qwen2模型的注意力机制替换方法的实现

    Args:
        model: 原始模型
        attn_type: 注意力机制类型
        **kwargs: 其他参数

    Returns:
        model: 替换后的模型
    """

    if attn_type == "standard":
        # 标准注意力不需要替换
        logger.info("使用标准注意力机制（原生实现）")
        return model

    elif attn_type == "sparse":
        sparsity = kwargs.get("sparsity", 0.8)
        return replace_with_sparse_attention(model, sparsity=sparsity)

    elif attn_type == "linear":
        kernel_function = kwargs.get("kernel_function", "elu")
        return replace_with_linear_attention(model, kernel_function=kernel_function)

    elif attn_type == "reformer":
        num_hashes = kwargs.get("num_hashes", 4)
        return replace_with_reformer_attention(model, num_hashes=num_hashes)

    elif attn_type == "linformer":
        k_ratio = kwargs.get("k_ratio", 0.25)
        max_seq_length = kwargs.get("max_seq_length", 512)
        return replace_with_linformer_attention(
            model,
            k_ratio=k_ratio,
            max_seq_length=max_seq_length,
        )

    elif attn_type == "longformer":
        window_size = kwargs.get("window_size", 128)
        global_tokens_ratio = kwargs.get("global_tokens_ratio", 0.1)
        return replace_with_longformer_attention(
            model,
            window_size=window_size,
            global_tokens_ratio=global_tokens_ratio,
        )

    elif attn_type == "realformer":
        return replace_with_realformer_attention(model)

    elif attn_type == "mla":
        rank_ratio = kwargs.get("rank_ratio", 0.25)
        return replace_with_mla_attention(model, rank_ratio=rank_ratio)

    elif attn_type == "custom":
        noise_level = kwargs.get("noise_level", 0.01)
        return replace_with_custom_attention(model, noise_level=noise_level)

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

    elif attn_type == "custom":
        info["noise_level"] = attn_config.get("noise_level", 0.01)

    return info
