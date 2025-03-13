"""
基准测试工具模块
"""

import time
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from .test_cases import get_test_case, get_all_test_cases

logger = logging.getLogger("attn_experiment")

class BenchmarkResult:
    """基准测试结果类"""
    
    def __init__(self, hardware_monitor=None):
        """
        初始化
        
        Args:
            hardware_monitor: 可选的硬件监控器实例
        """
        self.metrics = {
            "latency": [],  # 延迟（毫秒）
            "tokens_per_second": [],  # 生成速度（token/s）
            "memory_usage": [],  # 显存使用（MB）
            "perplexity": [],  # 困惑度
        }
        self.config = {}
        self.summary = {}
        self.hardware_monitor = hardware_monitor
        self.model_outputs = {}  # 存储每个测试用例的模型输出
        self.case_results = {}  # 存储每个测试用例的具体结果
    
    def add_metric(self, name, value):
        """添加指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # 如果有硬件监控器，也向其添加指标
        if self.hardware_monitor is not None:
            self.hardware_monitor.add_model_metric(name, value)
    
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
            "case_results": self.case_results,
            "model_outputs": self.model_outputs,
            "model_info": {
                "quantization": self.config.get("model_config", {}).get("quantization", "none"),
                "attention_type": self.config.get("model_config", {}).get("attention_type", "standard")
            }
        }
    
    def add_model_output(self, case_name, prompt, output):
        """添加模型输出"""
        self.model_outputs[case_name] = {
            "prompt": prompt,
            "output": output
        }
    
    def add_case_result(self, case_name, result):
        """添加测试用例结果"""
        self.case_results[case_name] = result

def measure_performance(model, tokenizer, prompt, max_new_tokens=20, num_runs=5, warmup_runs=2, hardware_monitor=None):
    """
    测量模型性能（延迟、显存使用和生成速度）
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 提示文本
        max_new_tokens: 最大生成token数
        num_runs: 运行次数
        warmup_runs: 预热运行次数
        hardware_monitor: 可选的硬件监控器实例
    
    Returns:
        tuple: (平均延迟, 平均生成速度, 显存使用, 模型输出)
    """
    logger.info(f"测量性能，提示文本: {prompt}")
    
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
    
    # 测量延迟和显存
    latencies = []
    tokens_per_second_list = []
    
    # 记录初始显存使用
    torch.cuda.synchronize()
    initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
    
    # 存储最后一次运行的输出
    last_output = None
    
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
        
        # 如果有硬件监控器，也向其添加实时指标
        if hardware_monitor is not None:
            hardware_monitor.add_model_metric("latency", latency)
            hardware_monitor.add_model_metric("tokens_per_second", tokens_per_second)
        
        # 保存最后一次运行的输出
        last_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 记录最终显存使用
    torch.cuda.synchronize()
    final_memory = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
    
    # 计算显存使用
    memory_usage = final_memory - initial_memory
    
    # 如果有硬件监控器，也向其添加显存指标
    if hardware_monitor is not None:
        hardware_monitor.add_model_metric("memory_usage", memory_usage)
    
    # 计算平均值
    avg_latency = np.mean(latencies)
    avg_tokens_per_second = np.mean(tokens_per_second_list)
    
    logger.info(f"平均延迟: {avg_latency:.2f} ms")
    logger.info(f"平均生成速度: {avg_tokens_per_second:.2f} token/s")
    logger.info(f"显存使用: {memory_usage:.2f} MB")
    logger.info(f"模型输出: {last_output}")
    
    return avg_latency, avg_tokens_per_second, memory_usage, last_output

def calculate_perplexity(model, tokenizer, text, stride=512, hardware_monitor=None):
    """
    计算困惑度
    
    Args:
        model: 模型
        tokenizer: 分词器
        text: 文本
        stride: 步长
        hardware_monitor: 可选的硬件监控器实例
    
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
    
    # 如果有硬件监控器，也向其添加指标
    if hardware_monitor is not None:
        hardware_monitor.add_model_metric("perplexity", ppl.item())
    
    logger.info(f"困惑度: {ppl.item():.2f}")
    
    return ppl.item()

def run_benchmark(model, tokenizer, config, hardware_monitor=None):
    """
    运行基准测试
    
    Args:
        model: 模型
        tokenizer: 分词器
        config: 配置字典，可以包含以下键：
            - max_test_cases: 最大测试用例数量，默认为3
            - num_runs: 每个测试运行次数
            - warmup_runs: 预热运行次数
            - stride: 计算困惑度时的步长
        hardware_monitor: 可选的硬件监控器实例
    
    Returns:
        BenchmarkResult: 测试结果
    """
    result = BenchmarkResult(hardware_monitor)
    result.set_config(config)
    
    # 获取测试用例，默认最多使用3个
    max_test_cases = config.get("max_test_cases", 3)
    test_cases = get_all_test_cases(max_test_cases)
    
    for case in test_cases:
        logger.info(f"运行测试用例: {case['name']}")
        logger.info(f"描述: {case['description']}")
        
        # 测量性能（延迟、显存使用和生成速度）
        latency, tokens_per_second, memory_usage, model_output = measure_performance(
            model,
            tokenizer,
            case["prompt"],
            max_new_tokens=case["max_new_tokens"],
            num_runs=config.get("num_runs", 5),
            warmup_runs=config.get("warmup_runs", 2),
            hardware_monitor=hardware_monitor
        )
        
        # 计算困惑度
        perplexity = calculate_perplexity(
            model,
            tokenizer,
            case["prompt"],
            stride=config.get("stride", 512),
            hardware_monitor=hardware_monitor
        )
        
        # 记录结果
        result.add_metric("latency", latency)
        result.add_metric("tokens_per_second", tokens_per_second)
        result.add_metric("memory_usage", memory_usage)
        result.add_metric("perplexity", perplexity)
        
        # 记录模型输出
        result.add_model_output(case["name"], case["prompt"], model_output)
        
        # 记录每个测试用例的具体结果
        result.add_case_result(case["name"], {
            "latency": latency,
            "tokens_per_second": tokens_per_second,
            "memory_usage": memory_usage,
            "perplexity": perplexity,
            "model_output": model_output
        })
    
    # 计算总体摘要
    result.compute_summary()
    
    return result 