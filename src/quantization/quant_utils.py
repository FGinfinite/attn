"""
量化工具模块，用于统一管理不同的量化方法
"""

import os
import torch
import logging
from pathlib import Path

logger = logging.getLogger("attn_experiment")

# 导入量化工具
from src.quantization.awq_utils import (
    check_awq_available, load_awq_model, quantize_model_awq, get_awq_model_info
)
from src.quantization.gptq_utils import (
    check_gptq_available, load_gptq_model, quantize_model_gptq, get_gptq_model_info
)

def load_quantized_model(model_path, quant_type, device="cuda"):
    """
    加载量化模型
    
    Args:
        model_path: 模型路径或名称
        quant_type: 量化类型，可选值为"awq"或"gptq"
        device: 设备
    
    Returns:
        model: 量化后的模型
    """
    if quant_type == "awq":
        if not check_awq_available():
            raise ImportError("未安装autoawq库，无法使用AWQ量化")
        return load_awq_model(model_path, device)
    
    elif quant_type == "gptq":
        if not check_gptq_available():
            raise ImportError("未安装auto-gptq库，无法使用GPTQ量化")
        return load_gptq_model(model_path, device)
    
    else:
        raise ValueError(f"不支持的量化类型: {quant_type}")

def quantize_model(model, tokenizer, quant_type, quant_config, output_dir):
    """
    量化模型
    
    Args:
        model: 原始模型
        tokenizer: 分词器
        quant_type: 量化类型，可选值为"awq"或"gptq"
        quant_config: 量化配置
        output_dir: 输出目录
    
    Returns:
        quantized_model: 量化后的模型
    """
    if quant_type == "awq":
        if not check_awq_available():
            raise ImportError("未安装autoawq库，无法使用AWQ量化")
        return quantize_model_awq(model, tokenizer, quant_config, output_dir)
    
    elif quant_type == "gptq":
        if not check_gptq_available():
            raise ImportError("未安装auto-gptq库，无法使用GPTQ量化")
        return quantize_model_gptq(model, tokenizer, quant_config, output_dir)
    
    else:
        raise ValueError(f"不支持的量化类型: {quant_type}")

def get_quantized_model_info(model, quant_type):
    """
    获取量化模型信息
    
    Args:
        model: 量化模型
        quant_type: 量化类型，可选值为"awq"或"gptq"
    
    Returns:
        info: 模型信息字典
    """
    if quant_type == "awq":
        return get_awq_model_info(model)
    
    elif quant_type == "gptq":
        return get_gptq_model_info(model)
    
    else:
        raise ValueError(f"不支持的量化类型: {quant_type}")

def check_quantization_error(original_model, quantized_model, tokenizer, prompt, device="cuda"):
    """
    检查量化误差
    
    Args:
        original_model: 原始模型
        quantized_model: 量化模型
        tokenizer: 分词器
        prompt: 提示文本
        device: 设备
    
    Returns:
        error: 量化误差（余弦相似度）
    """
    from torch.nn.functional import cosine_similarity
    
    logger.info(f"检查量化误差，提示文本: {prompt}")
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 获取原始模型输出
    with torch.no_grad():
        original_outputs = original_model(**inputs)
        original_logits = original_outputs.logits[:, -1, :]
    
    # 获取量化模型输出
    with torch.no_grad():
        quantized_outputs = quantized_model(**inputs)
        quantized_logits = quantized_outputs.logits[:, -1, :]
    
    # 计算余弦相似度
    similarity = cosine_similarity(
        original_logits.float().view(1, -1),
        quantized_logits.float().view(1, -1)
    ).item()
    
    logger.info(f"量化误差（余弦相似度）: {similarity}")
    
    return similarity 