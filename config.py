#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件
包含项目的所有配置参数
"""

import os
import yaml
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# 加载YAML配置
def load_config():
    config_path = ROOT_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# 模型配置
DEFAULT_MODEL_PATH = CONFIG["model"]["default_model_path"]
SUPPORTED_MODELS = CONFIG["model"]["supported_models"]

# 量化配置
SUPPORTED_QUANT_TYPES = CONFIG["quantization"]["supported_types"]
AWQ_CONFIG = CONFIG["quantization"]["awq"]
GPTQ_CONFIG = CONFIG["quantization"]["gptq"]

# 注意力机制配置
SUPPORTED_ATTENTION_TYPES = CONFIG["attention"]["supported_types"]
SPARSE_ATTENTION_CONFIG = CONFIG["attention"]["sparse"]
LINEAR_ATTENTION_CONFIG = CONFIG["attention"]["linear"]
REFORMER_ATTENTION_CONFIG = CONFIG["attention"].get("reformer", {})
LINFORMER_ATTENTION_CONFIG = CONFIG["attention"].get("linformer", {})
LONGFORMER_ATTENTION_CONFIG = CONFIG["attention"].get("longformer", {})
REALFORMER_ATTENTION_CONFIG = CONFIG["attention"].get("realformer", {})
LOW_RANK_ATTENTION_CONFIG = CONFIG["attention"].get("low_rank", {})

# 基准测试配置
DEFAULT_BATCH_SIZE = CONFIG["benchmark"]["default_batch_size"]
DEFAULT_INPUT_LENGTH = CONFIG["benchmark"]["default_input_length"]
DEFAULT_OUTPUT_LENGTH = CONFIG["benchmark"]["default_output_length"]
DEFAULT_NUM_RUNS = CONFIG["benchmark"]["default_num_runs"]
DEFAULT_WARMUP_RUNS = CONFIG["benchmark"]["default_warmup_runs"]

# 日志配置
LOG_DIR = CONFIG["logging"]["log_dir"]
RESULTS_DIR = CONFIG["logging"]["results_dir"]
ANALYSIS_DIR = CONFIG["logging"]["analysis_dir"]

# 脚本路径配置
SCRIPTS_CONFIG = {
    "model": {
        "verify": "scripts/model/verify_model.py",
        "quantize": "scripts/model/quantize_model.py",
        "test_attention": "scripts/model/test_attention.py",
        "test_vllm": "scripts/model/test_vllm.py"
    },
    "benchmark": {
        "run_benchmark": "scripts/benchmark/run_benchmark.py",
        "run_all_tests": "scripts/benchmark/run_all_tests.py"
    },
    "analysis": {
        "analyze_results": "scripts/analysis/analyze_results.py"
    }
}

# 模型配置
MODEL_CONFIG = {
    "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
    "max_seq_len": 4096,
    "dtype": "float16",
}

# 量化配置
QUANTIZATION_CONFIG = {
    "awq": {
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "fuse_layers": True,
    },
    "gptq": {
        "bits": 4,
        "group_size": 128,
        "act_order": True,
        "use_safetensors": True,
    }
}

# 注意力机制配置
ATTENTION_CONFIG = {
    "standard": {
        "name": "标准Self-Attention",
        "description": "使用Qwen2Model原生实现",
    },
    "sparse": {
        "name": "Sparse Attention",
        "description": "修改RotaryEmbedding模块并集成动态稀疏掩码",
        "sparsity": 0.8,  # 稀疏度，表示保留的注意力比例
    },
    "linear": {
        "name": "Linear Attention",
        "description": "基于线性注意力公式重写计算逻辑",
        "kernel_function": "elu",  # 可选：elu, relu, softmax
    },
    "longformer": {
        "name": "Longformer",
        "description": "适配Qwen2的RoPE位置编码实现滑动窗口注意力",
        "window_size": 512,
    },
    "reformer": {
        "name": "Reformer",
        "description": "使用LSH哈希实现局部敏感注意力",
        "num_hashes": 4,
    },
    "linformer": {
        "name": "Linformer",
        "description": "使用低秩投影降低序列长度维度的复杂度",
        "k_ratio": 0.25,
        "max_seq_length": 512,
    },
    "realformer": {
        "name": "RealFormer",
        "description": "使用残差连接累积注意力分数",
    },
    "low_rank": {
        "name": "低秩分解注意力",
        "description": "使用SVD分解权重矩阵为两个低秩子矩阵的乘积",
        "rank_ratio": 0.5,  # 低秩比例，表示保留的奇异值比例
    },
}

# vLLM配置
VLLM_CONFIG = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "enforce_eager": True,  # 兼容自定义注意力
}

# 实验配置
EXPERIMENT_CONFIG = {
    "batch_sizes": [1, 4, 8],
    "input_lengths": [512, 1024, 2048, 4096],
    "output_lengths": [128, 256, 512],
    "seeds": [42, 123, 456],
    "metrics": ["token_per_second", "latency", "memory_usage", "perplexity"],
    "results_dir": ROOT_DIR / "data" / "results",
}

# 日志配置
LOGGING_CONFIG = {
    "log_dir": ROOT_DIR / "logs",
    "log_level": "INFO",
    "log_to_file": True,
    "log_to_console": True,
} 