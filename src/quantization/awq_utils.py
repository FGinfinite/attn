"""
AWQ量化工具模块
"""

import os
import torch
import logging
from pathlib import Path

logger = logging.getLogger("attn_experiment")

try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False
    logger.warning("未安装autoawq库，无法使用AWQ量化")

def check_awq_available():
    """检查AWQ是否可用"""
    return AWQ_AVAILABLE

def load_awq_model(model_path, device="cuda"):
    """
    加载AWQ量化模型
    
    Args:
        model_path: 模型路径或名称
        device: 设备
    
    Returns:
        model: 量化后的模型
    """
    if not AWQ_AVAILABLE:
        raise ImportError("未安装autoawq库，无法使用AWQ量化")
    
    logger.info(f"加载AWQ量化模型: {model_path}")
    
    try:
        model = AutoAWQForCausalLM.from_quantized(
            model_path,
            fuse_layers=True,
            trust_remote_code=True,
            safetensors=True
        )
        
        logger.info(f"AWQ量化模型加载完成")
        return model
    
    except Exception as e:
        logger.error(f"AWQ量化模型加载失败: {str(e)}")
        raise

def quantize_model_awq(model, tokenizer, quant_config, output_dir):
    """
    使用AWQ量化模型
    
    Args:
        model: 原始模型
        tokenizer: 分词器
        quant_config: 量化配置
        output_dir: 输出目录
    
    Returns:
        quantized_model: 量化后的模型
    """
    if not AWQ_AVAILABLE:
        raise ImportError("未安装autoawq库，无法使用AWQ量化")
    
    from awq import AutoAWQQuantizer
    
    logger.info(f"开始AWQ量化模型")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取量化参数
    bits = quant_config.get("bits", 4)
    group_size = quant_config.get("group_size", 128)
    zero_point = quant_config.get("zero_point", True)
    
    logger.info(f"量化参数: bits={bits}, group_size={group_size}, zero_point={zero_point}")
    
    try:
        # 创建量化器
        quantizer = AutoAWQQuantizer(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            zero_point=zero_point
        )
        
        # 量化模型
        quantizer.quantize(
            output_dir=str(output_dir),
            safetensors=True
        )
        
        # 加载量化后的模型
        quantized_model = AutoAWQForCausalLM.from_quantized(
            str(output_dir),
            fuse_layers=True,
            trust_remote_code=True,
            safetensors=True
        )
        
        logger.info(f"AWQ量化完成，模型保存在: {output_dir}")
        return quantized_model
    
    except Exception as e:
        logger.error(f"AWQ量化失败: {str(e)}")
        raise

def get_awq_model_info(model):
    """
    获取AWQ量化模型信息
    
    Args:
        model: AWQ量化模型
    
    Returns:
        info: 模型信息字典
    """
    info = {
        "model_type": model.__class__.__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": next(model.parameters()).device.type,
        "quantization": "AWQ",
        "bits": 4,  # AWQ默认为4bit
    }
    
    return info 