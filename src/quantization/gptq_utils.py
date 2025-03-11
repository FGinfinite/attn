"""
GPTQ量化工具模块
"""

import os
import torch
import logging
from pathlib import Path

logger = logging.getLogger("attn_experiment")

try:
    from auto_gptq import AutoGPTQForCausalLM
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False
    logger.warning("未安装auto-gptq库，无法使用GPTQ量化")

def check_gptq_available():
    """检查GPTQ是否可用"""
    return GPTQ_AVAILABLE

def load_gptq_model(model_path, device="cuda"):
    """
    加载GPTQ量化模型
    
    Args:
        model_path: 模型路径或名称
        device: 设备
    
    Returns:
        model: 量化后的模型
    """
    if not GPTQ_AVAILABLE:
        raise ImportError("未安装auto-gptq库，无法使用GPTQ量化")
    
    logger.info(f"加载GPTQ量化模型: {model_path}")
    
    try:
        model = AutoGPTQForCausalLM.from_quantized(
            model_path,
            use_safetensors=True,
            trust_remote_code=True,
            device=device,
            use_triton=False,
            inject_fused_attention=False
        )
        
        logger.info(f"GPTQ量化模型加载完成")
        return model
    
    except Exception as e:
        logger.error(f"GPTQ量化模型加载失败: {str(e)}")
        raise

def quantize_model_gptq(model, tokenizer, quant_config, output_dir):
    """
    使用GPTQ量化模型
    
    Args:
        model: 原始模型
        tokenizer: 分词器
        quant_config: 量化配置
        output_dir: 输出目录
    
    Returns:
        quantized_model: 量化后的模型
    """
    if not GPTQ_AVAILABLE:
        raise ImportError("未安装auto-gptq库，无法使用GPTQ量化")
    
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from auto_gptq.utils.exllama_utils import exllama_set_max_input_length
    
    logger.info(f"开始GPTQ量化模型")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取量化参数
    bits = quant_config.get("bits", 4)
    group_size = quant_config.get("group_size", 128)
    act_order = quant_config.get("act_order", True)
    
    logger.info(f"量化参数: bits={bits}, group_size={group_size}, act_order={act_order}")
    
    try:
        # 创建量化配置
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=act_order
        )
        
        # 量化模型
        quantized_model = AutoGPTQForCausalLM.from_pretrained(
            model,
            quantize_config=quantize_config,
            trust_remote_code=True
        )
        
        # 使用一些示例数据进行校准
        examples = [
            tokenizer("你好，请介绍一下自己。", return_tensors="pt").input_ids,
            tokenizer("请解释一下量子力学的基本原理。", return_tensors="pt").input_ids,
            tokenizer("如何使用Python进行数据分析？", return_tensors="pt").input_ids,
        ]
        
        # 量化模型
        quantized_model.quantize(examples)
        
        # 保存量化后的模型
        quantized_model.save_quantized(str(output_dir), use_safetensors=True)
        
        # 重新加载量化后的模型
        quantized_model = AutoGPTQForCausalLM.from_quantized(
            str(output_dir),
            use_safetensors=True,
            trust_remote_code=True
        )
        
        logger.info(f"GPTQ量化完成，模型保存在: {output_dir}")
        return quantized_model
    
    except Exception as e:
        logger.error(f"GPTQ量化失败: {str(e)}")
        raise

def get_gptq_model_info(model):
    """
    获取GPTQ量化模型信息
    
    Args:
        model: GPTQ量化模型
    
    Returns:
        info: 模型信息字典
    """
    info = {
        "model_type": model.__class__.__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": next(model.parameters()).device.type,
        "quantization": "GPTQ",
        "bits": model.quantize_config.bits if hasattr(model, "quantize_config") else 4,
    }
    
    return info 