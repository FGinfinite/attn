"""
模型工具模块，用于加载和验证模型
"""

import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger("attn_experiment")

def load_model_and_tokenizer(model_path, dtype="float16", device="cuda"):
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径或名称
        dtype: 数据类型
        device: 设备
    
    Returns:
        model: 模型
        tokenizer: 分词器
    """
    logger.info(f"加载模型: {model_path}, 数据类型: {dtype}, 设备: {device}")
    
    # 设置数据类型
    torch_dtype = getattr(torch, dtype)
    logger.info(f"使用torch数据类型: {torch_dtype}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 记录GPU显存使用情况（加载模型前）
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            logger.info(f"加载模型前 - GPU {i} 显存使用情况: {allocated:.2f} MB / {reserved:.2f} MB")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    # 记录GPU显存使用情况（加载模型后）
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            logger.info(f"加载模型后 - GPU {i} 显存使用情况: {allocated:.2f} MB / {reserved:.2f} MB")
    
    # 检查模型参数的数据类型
    param_dtypes = {p.dtype for p in model.parameters()}
    logger.info(f"模型参数数据类型: {param_dtypes}")
    
    # 检查是否成功应用了指定的数据类型
    if torch_dtype not in param_dtypes:
        logger.warning(f"警告：模型参数中没有找到指定的数据类型 {torch_dtype}，可能未成功应用")
        
        # 如果是fp16但未成功应用，尝试强制转换
        if dtype == "float16":
            logger.info("尝试强制转换为fp16...")
            model = model.half()
            
            # 再次检查
            param_dtypes = {p.dtype for p in model.parameters()}
            logger.info(f"转换后模型参数数据类型: {param_dtypes}")
            
            if torch.float16 not in param_dtypes:
                logger.error("错误：无法将模型转换为fp16格式")
    else:
        logger.info(f"确认：模型已成功加载为 {torch_dtype} 格式")
    
    # 检查模型配置中的dtype
    if hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
        logger.info(f"模型配置中的dtype: {model.config.torch_dtype}")
        # 确保配置中的dtype与实际使用的一致
        model.config.torch_dtype = torch_dtype
    
    logger.info(f"模型加载完成")
    return model, tokenizer

def verify_model(model, tokenizer, prompt="你好，请介绍一下自己。", max_new_tokens=50):
    """
    验证模型是否能正常运行
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 提示文本
        max_new_tokens: 最大生成token数
    
    Returns:
        success: 是否成功
        output: 输出文本
    """
    logger.info(f"验证模型，提示文本: {prompt}")
    
    try:
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成输出
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None
            )
        
        # 解码输出
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"模型验证成功")
        logger.info(f"输出: {output_text}")
        
        return True, output_text
    
    except Exception as e:
        logger.error(f"模型验证失败: {str(e)}")
        return False, str(e)

def get_model_info(model):
    """
    获取模型信息
    
    Args:
        model: 模型
    
    Returns:
        info: 模型信息字典
    """
    info = {
        "model_type": model.__class__.__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": next(model.parameters()).device.type,
        "dtype": next(model.parameters()).dtype,
    }
    
    return info 