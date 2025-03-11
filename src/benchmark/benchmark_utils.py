"""
基准测试工具模块
"""

import time
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

logger = logging.getLogger("attn_experiment")

class BenchmarkResult:
    """基准测试结果类"""
    
    def __init__(self):
        """初始化"""
        self.metrics = {
            "latency": [],  # 延迟（毫秒）
            "tokens_per_second": [],  # 生成速度（token/s）
            "memory_usage": [],  # 显存使用（MB）
            "perplexity": [],  # 困惑度
        }
        self.config = {}
        self.summary = {}
    
    def add_metric(self, name, value):
        """添加指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
    
    def set_config(self, config):
        """设置配置"""
        self.config = config.copy()  # 使用copy确保配置不会被修改
        # 将配置信息添加到摘要中
        self.summary.update({
            "model_path": config.get("model_path", ""),
            "quantization": config.get("quant", "none"),
            "attention_type": config.get("attention", "standard"),
            "batch_size": config.get("batch_size", 1),
            "input_length": config.get("input_length", 512),
            "output_length": config.get("output_length", 128)
        })
    
    def compute_summary(self):
        """计算摘要"""
        for name, values in self.metrics.items():
            if len(values) > 0:
                self.summary[f"{name}_mean"] = np.mean(values)
                self.summary[f"{name}_std"] = np.std(values)
                self.summary[f"{name}_min"] = np.min(values)
                self.summary[f"{name}_max"] = np.max(values)
        
        return self.summary
    
    def to_dict(self):
        """转换为字典"""
        return {
            "metrics": self.metrics,
            "config": self.config,
            "summary": self.summary,
            "model_info": {
                "quantization": self.config.get("model_config", {}).get("quantization", "none"),
                "attention_type": self.config.get("model_config", {}).get("attention_type", "standard")
            }
        }

def measure_latency(model, tokenizer, prompt, max_new_tokens=20, num_runs=5, warmup_runs=2):
    """
    测量生成延迟
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 提示文本
        max_new_tokens: 最大生成token数
        num_runs: 运行次数
        warmup_runs: 预热运行次数
    
    Returns:
        latency: 平均延迟（毫秒）
        tokens_per_second: 平均生成速度（token/s）
    """
    logger.info(f"测量生成延迟，提示文本: {prompt}")
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 预热
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None
            )
    
    # 测量延迟
    latencies = []
    tokens_per_second_list = []
    
    for _ in range(num_runs):
        # 清除缓存
        torch.cuda.empty_cache()
        
        # 测量时间
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None
            )
        
        end_time = time.time()
        
        # 计算延迟
        latency = (end_time - start_time) * 1000  # 转换为毫秒
        latencies.append(latency)
        
        # 计算生成的token数
        num_generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        
        # 计算生成速度
        tokens_per_second = num_generated_tokens / (end_time - start_time)
        tokens_per_second_list.append(tokens_per_second)
    
    # 计算平均值
    avg_latency = np.mean(latencies)
    avg_tokens_per_second = np.mean(tokens_per_second_list)
    
    logger.info(f"平均延迟: {avg_latency:.2f} ms")
    logger.info(f"平均生成速度: {avg_tokens_per_second:.2f} token/s")
    
    return avg_latency, avg_tokens_per_second

def measure_memory_usage(model, tokenizer, prompt, max_new_tokens=20):
    """
    测量显存使用
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 提示文本
        max_new_tokens: 最大生成token数
    
    Returns:
        memory_usage: 显存使用（MB）
    """
    logger.info(f"测量显存使用，提示文本: {prompt}")
    
    # 清除缓存
    torch.cuda.empty_cache()
    
    # 记录初始显存使用
    torch.cuda.synchronize()
    initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成输出
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
    
    # 记录最终显存使用
    torch.cuda.synchronize()
    final_memory = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
    
    # 计算显存使用
    memory_usage = final_memory - initial_memory
    
    logger.info(f"显存使用: {memory_usage:.2f} MB")
    
    return memory_usage

def calculate_perplexity(model, tokenizer, text, stride=512):
    """
    计算困惑度
    
    Args:
        model: 模型
        tokenizer: 分词器
        text: 文本
        stride: 步长
    
    Returns:
        perplexity: 困惑度
    """
    logger.info(f"计算困惑度，文本长度: {len(text)}")
    
    # 编码文本
    encodings = tokenizer(text, return_tensors="pt")
    
    # 获取输入ID
    input_ids = encodings.input_ids.to(model.device)
    
    # 计算序列长度
    seq_len = input_ids.size(1)
    
    # 初始化
    nlls = []
    prev_end_loc = 0
    
    # 使用滑动窗口计算困惑度
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + 1024, seq_len)
        trg_len = end_loc - prev_end_loc
        
        # 提取输入
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()
        
        # 设置目标
        target_ids[:, :-trg_len] = -100
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            
            # 获取负对数似然
            neg_log_likelihood = outputs.loss * trg_len
        
        # 添加到列表
        nlls.append(neg_log_likelihood)
        
        # 更新位置
        prev_end_loc = end_loc
        
        # 如果到达序列末尾，退出循环
        if end_loc == seq_len:
            break
    
    # 计算困惑度
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    
    logger.info(f"困惑度: {ppl.item():.2f}")
    
    return ppl.item()

def run_benchmark(model, tokenizer, config):
    """
    运行基准测试
    
    Args:
        model: 模型
        tokenizer: 分词器
        config: 配置
    
    Returns:
        result: 基准测试结果
    """
    logger.info(f"开始运行基准测试")
    
    # 创建结果对象
    result = BenchmarkResult()
    result.set_config(config)
    
    # 获取配置
    prompts = config.get("prompts", ["你好，请介绍一下自己。"])
    max_new_tokens = config.get("max_new_tokens", 20)
    num_runs = config.get("num_runs", 5)
    warmup_runs = config.get("warmup_runs", 2)
    
    # 测量延迟和生成速度
    for prompt in prompts:
        latency, tokens_per_second = measure_latency(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        result.add_metric("latency", latency)
        result.add_metric("tokens_per_second", tokens_per_second)
    
    # 测量显存使用
    for prompt in prompts:
        memory_usage = measure_memory_usage(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )
        
        result.add_metric("memory_usage", memory_usage)
    
    # 计算困惑度
    if "perplexity_texts" in config:
        for text in config["perplexity_texts"]:
            perplexity = calculate_perplexity(
                model=model,
                tokenizer=tokenizer,
                text=text
            )
            
            result.add_metric("perplexity", perplexity)
    
    # 计算摘要
    result.compute_summary()
    
    logger.info(f"基准测试完成，摘要: {result.summary}")
    
    return result 