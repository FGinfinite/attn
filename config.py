"""
注意力机制对比实验项目配置文件
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

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