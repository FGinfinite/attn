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
    logger.info(f"加载模型: {model_path}")
    
    # 设置数据类型
    torch_dtype = getattr(torch, dtype)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
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