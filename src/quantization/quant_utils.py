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
        quant_type: 量化类型，可选值为"awq"、"gptq"、"fp16"、"bf16"或"none"
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
    
    elif quant_type in ["fp16", "bf16", "none"]:
        # 对于fp16、bf16和none，使用标准的模型加载方式
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 设置数据类型
        torch_dtype = None
        if quant_type == "fp16":
            torch_dtype = torch.float16
            dtype_name = "半精度(fp16)"
        elif quant_type == "bf16":
            torch_dtype = torch.bfloat16
            dtype_name = "脑浮点(bf16)"
        else:
            dtype_name = "原始精度"
        
        logger.info(f"使用{dtype_name}加载模型: {model_path}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 检查模型是否真的是指定的数据类型
            if quant_type in ["fp16", "bf16"]:
                expected_dtype = torch_dtype
                param_dtypes = {p.dtype for p in model.parameters()}
                logger.info(f"模型参数数据类型: {param_dtypes}")
                
                if expected_dtype not in param_dtypes:
                    logger.warning(f"警告：模型参数中没有发现{quant_type}类型，可能未成功应用量化")
                    # 强制转换为指定类型
                    logger.info(f"正在强制将模型转换为{quant_type}...")
                    if quant_type == "fp16":
                        model = model.half()
                    else:  # bf16
                        model = model.to(torch.bfloat16)
                    
                    # 再次检查
                    param_dtypes = {p.dtype for p in model.parameters()}
                    logger.info(f"转换后模型参数数据类型: {param_dtypes}")
                    
                    if expected_dtype not in param_dtypes:
                        logger.error(f"错误：无法将模型转换为{quant_type}格式")
                else:
                    logger.info(f"确认：模型已成功加载为{quant_type}格式")
            
            logger.info(f"模型加载完成")
            return model
        
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    else:
        raise ValueError(f"不支持的量化类型: {quant_type}")

def quantize_model(model, tokenizer, quant_type, quant_config, output_dir):
    """
    量化模型
    
    Args:
        model: 原始模型
        tokenizer: 分词器
        quant_type: 量化类型，可选值为"awq"、"gptq"、"fp16"、"bf16"或"none"
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
    
    elif quant_type in ["fp16", "bf16"]:
        # 半精度或脑浮点量化，直接转换模型
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        if quant_type == "fp16":
            logger.info(f"正在进行半精度(fp16)量化...")
            target_dtype = torch.float16
        else:  # bf16
            logger.info(f"正在进行脑浮点(bf16)量化...")
            target_dtype = torch.bfloat16
        
        # 检查原始模型的数据类型
        original_dtypes = {p.dtype for p in model.parameters()}
        logger.info(f"原始模型参数数据类型: {original_dtypes}")
        
        # 转换为目标类型
        if quant_type == "fp16":
            model = model.half()
        else:  # bf16
            model = model.to(torch.bfloat16)
        
        # 验证转换是否成功
        converted_dtypes = {p.dtype for p in model.parameters()}
        logger.info(f"转换后模型参数数据类型: {converted_dtypes}")
        
        if target_dtype not in converted_dtypes:
            logger.error(f"错误：模型未能成功转换为{quant_type}格式")
            raise ValueError(f"无法将模型转换为{quant_type}格式")
        
        # 保存模型和分词器，确保以指定格式保存
        logger.info(f"保存{quant_type}模型...")
        model.config.torch_dtype = target_dtype  # 在配置中明确指定dtype
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        logger.info(f"{quant_type}量化完成，模型保存在: {output_dir}")
        
        # 重新加载模型以确保它以指定格式加载
        logger.info(f"重新加载模型以验证{quant_type}格式...")
        from transformers import AutoModelForCausalLM
        reloaded_model = AutoModelForCausalLM.from_pretrained(
            str(output_dir),
            torch_dtype=target_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 验证重新加载的模型是否为指定类型
        reloaded_dtypes = {p.dtype for p in reloaded_model.parameters()}
        logger.info(f"重新加载的模型参数数据类型: {reloaded_dtypes}")
        
        if target_dtype not in reloaded_dtypes:
            logger.warning(f"警告：重新加载的模型不是{quant_type}格式，可能存在保存或加载问题")
        else:
            logger.info(f"确认：模型已成功保存和加载为{quant_type}格式")
        
        return reloaded_model
    
    elif quant_type == "none":
        # 不进行量化，直接返回原始模型
        logger.info("不进行量化，使用原始模型")
        return model
    
    else:
        raise ValueError(f"不支持的量化类型: {quant_type}")

def get_quantized_model_info(model, quant_type):
    """
    获取量化模型信息
    
    Args:
        model: 量化模型
        quant_type: 量化类型，可选值为"awq"、"gptq"、"fp16"、"bf16"或"none"
    
    Returns:
        info: 模型信息字典
    """
    if quant_type == "awq":
        return get_awq_model_info(model)
    
    elif quant_type == "gptq":
        return get_gptq_model_info(model)
    
    elif quant_type == "fp16":
        # 获取半精度模型信息
        info = {
            "model_type": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": next(model.parameters()).device.type,
            "quantization": "FP16",
            "bits": 16,
        }
        return info
    
    elif quant_type == "bf16":
        # 获取脑浮点模型信息
        info = {
            "model_type": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": next(model.parameters()).device.type,
            "quantization": "BF16",
            "bits": 16,
        }
        return info
    
    elif quant_type == "none":
        # 获取原始模型信息
        info = {
            "model_type": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": next(model.parameters()).device.type,
            "quantization": "None",
            "bits": 32,
            "dtype": next(model.parameters()).dtype
        }
        return info
    
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
    
    try:
        # 确保两个模型都在同一设备上
        original_device = next(original_model.parameters()).device
        quantized_device = next(quantized_model.parameters()).device
        
        logger.info(f"原始模型设备: {original_device}, 量化模型设备: {quantized_device}")
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 获取原始模型输出
        with torch.no_grad():
            original_inputs = {k: v.to(original_device) for k, v in inputs.items()}
            original_outputs = original_model(**original_inputs)
            original_logits = original_outputs.logits[:, -1, :].to("cpu")  # 移到CPU
        
        # 获取量化模型输出
        with torch.no_grad():
            quantized_inputs = {k: v.to(quantized_device) for k, v in inputs.items()}
            quantized_outputs = quantized_model(**quantized_inputs)
            quantized_logits = quantized_outputs.logits[:, -1, :].to("cpu")  # 移到CPU
        
        # 计算余弦相似度（在CPU上）
        similarity = cosine_similarity(
            original_logits.float().view(1, -1),
            quantized_logits.float().view(1, -1)
        ).item()
        
        logger.info(f"量化误差（余弦相似度）: {similarity}")
        
        return similarity
    
    except Exception as e:
        logger.error(f"检查量化误差失败: {str(e)}")
        # 返回一个默认值表示检查失败
        return -1.0 