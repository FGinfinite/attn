"""
vLLM加速工具模块
"""

import os
import time
import torch
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("attn_experiment")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("未安装vllm库，无法使用vLLM加速")

def check_vllm_available():
    """检查vLLM是否可用"""
    return VLLM_AVAILABLE

def load_vllm_model(model_path, quantization=None, tensor_parallel_size=1, gpu_memory_utilization=0.85, enforce_eager=True):
    """
    加载vLLM模型
    
    Args:
        model_path: 模型路径或名称
        quantization: 量化方式，可选值为"awq", "gptq", None
        tensor_parallel_size: 张量并行大小
        gpu_memory_utilization: GPU内存利用率
        enforce_eager: 是否强制使用eager模式（兼容自定义注意力）
    
    Returns:
        vllm_model: vLLM模型
    """
    if not VLLM_AVAILABLE:
        raise ImportError("未安装vllm库，无法使用vLLM加速")
    
    logger.info(f"加载vLLM模型: {model_path}")
    logger.info(f"量化方式: {quantization}")
    logger.info(f"张量并行大小: {tensor_parallel_size}")
    logger.info(f"GPU内存利用率: {gpu_memory_utilization}")
    logger.info(f"强制使用eager模式: {enforce_eager}")
    
    try:
        vllm_model = LLM(
            model=model_path,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            trust_remote_code=True
        )
        
        logger.info(f"vLLM模型加载完成")
        return vllm_model
    
    except Exception as e:
        logger.error(f"vLLM模型加载失败: {str(e)}")
        raise

def generate_with_vllm(vllm_model, prompts, max_tokens=20, temperature=0.0, top_p=1.0, top_k=None):
    """
    使用vLLM生成文本
    
    Args:
        vllm_model: vLLM模型
        prompts: 提示文本列表
        max_tokens: 最大生成token数
        temperature: 温度
        top_p: top-p采样参数
        top_k: top-k采样参数
    
    Returns:
        outputs: 生成的文本列表
    """
    if not VLLM_AVAILABLE:
        raise ImportError("未安装vllm库，无法使用vLLM加速")
    
    logger.info(f"使用vLLM生成文本，提示数量: {len(prompts)}")
    
    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens
    )
    
    # 生成文本
    outputs = vllm_model.generate(prompts, sampling_params)
    
    # 提取生成的文本
    generated_texts = [output.outputs[0].text for output in outputs]
    
    return generated_texts

def measure_vllm_latency(vllm_model, prompts, max_tokens=20, num_runs=5, warmup_runs=2):
    """
    测量vLLM生成延迟
    
    Args:
        vllm_model: vLLM模型
        prompts: 提示文本列表
        max_tokens: 最大生成token数
        num_runs: 运行次数
        warmup_runs: 预热运行次数
    
    Returns:
        latency: 平均延迟（毫秒）
        tokens_per_second: 平均生成速度（token/s）
    """
    if not VLLM_AVAILABLE:
        raise ImportError("未安装vllm库，无法使用vLLM加速")
    
    logger.info(f"测量vLLM生成延迟，提示数量: {len(prompts)}")
    
    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_tokens=max_tokens
    )
    
    # 预热
    for _ in range(warmup_runs):
        _ = vllm_model.generate(prompts, sampling_params)
    
    # 测量延迟
    latencies = []
    tokens_per_second_list = []
    
    for _ in range(num_runs):
        # 测量时间
        start_time = time.time()
        
        outputs = vllm_model.generate(prompts, sampling_params)
        
        end_time = time.time()
        
        # 计算延迟
        latency = (end_time - start_time) * 1000  # 转换为毫秒
        latencies.append(latency)
        
        # 计算生成的token数
        num_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        
        # 计算生成速度
        tokens_per_second = num_generated_tokens / (end_time - start_time)
        tokens_per_second_list.append(tokens_per_second)
    
    # 计算平均值
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list)
    
    logger.info(f"平均延迟: {avg_latency:.2f} ms")
    logger.info(f"平均生成速度: {avg_tokens_per_second:.2f} token/s")
    
    return avg_latency, avg_tokens_per_second

def get_vllm_model_info(vllm_model):
    """
    获取vLLM模型信息
    
    Args:
        vllm_model: vLLM模型
    
    Returns:
        info: 模型信息字典
    """
    if not VLLM_AVAILABLE:
        raise ImportError("未安装vllm库，无法使用vLLM加速")
    
    info = {
        "model_type": "vLLM",
        "model_path": vllm_model.llm_engine.model_config.model,
        "quantization": vllm_model.llm_engine.model_config.quantization,
        "tensor_parallel_size": vllm_model.llm_engine.model_config.tensor_parallel_size,
        "gpu_memory_utilization": vllm_model.llm_engine.model_config.gpu_memory_utilization,
    }
    
    return info 