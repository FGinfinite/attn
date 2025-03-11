"""
注意力机制对比实验项目主入口文件
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, QUANTIZATION_CONFIG, ATTENTION_CONFIG, VLLM_CONFIG, LOGGING_CONFIG
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="注意力机制对比实验")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default=MODEL_CONFIG["model_name_or_path"],
                        help="模型路径或名称")
    parser.add_argument("--max_seq_len", type=int, default=MODEL_CONFIG["max_seq_len"],
                        help="最大序列长度")
    
    # 量化相关参数
    parser.add_argument("--quant", type=str, default="none", choices=["none", "awq", "gptq"],
                        help="量化方式：none(FP16), awq, gptq")
    
    # 注意力机制相关参数
    parser.add_argument("--attention", type=str, default="standard", 
                        choices=list(ATTENTION_CONFIG.keys()),
                        help="注意力机制类型")
    
    # vLLM相关参数
    parser.add_argument("--use_vllm", action="store_true", help="是否使用vLLM加速")
    parser.add_argument("--vllm_tensor_parallel", type=int, 
                        default=VLLM_CONFIG["tensor_parallel_size"],
                        help="vLLM张量并行大小")
    
    # 实验相关参数
    parser.add_argument("--run_benchmark", action="store_true", help="是否运行基准测试")
    parser.add_argument("--save_results", action="store_true", help="是否保存结果")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logger = setup_logger(
        name="attn_experiment",
        log_dir=LOGGING_CONFIG["log_dir"],
        log_level=LOGGING_CONFIG["log_level"],
        log_to_file=LOGGING_CONFIG["log_to_file"],
        log_to_console=LOGGING_CONFIG["log_to_console"]
    )
    
    logger.info(f"启动注意力机制对比实验")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"量化方式: {args.quant}")
    logger.info(f"注意力机制: {args.attention}")
    logger.info(f"使用vLLM: {args.use_vllm}")
    
    # TODO: 实现模型加载逻辑
    
    # TODO: 实现注意力机制替换逻辑
    
    # TODO: 实现基准测试逻辑
    
    logger.info(f"实验完成")

if __name__ == "__main__":
    main() 