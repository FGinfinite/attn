#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
注意力机制对比实验项目主入口
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="注意力机制对比实验")
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 验证模型命令
    verify_parser = subparsers.add_parser("verify", help="验证模型")
    verify_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    verify_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    
    # 量化模型命令
    quant_parser = subparsers.add_parser("quantize", help="量化模型")
    quant_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    quant_parser.add_argument("--quant", type=str, choices=["awq", "gptq"], default="awq", help="量化方法")
    quant_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    
    # 测试注意力机制命令
    attn_parser = subparsers.add_parser("test_attention", help="测试注意力机制")
    attn_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    attn_parser.add_argument("--attention", type=str, choices=["standard", "sparse", "linear"], default="standard", help="注意力机制类型")
    attn_parser.add_argument("--sparsity", type=float, default=0.8, help="稀疏注意力的稀疏度")
    attn_parser.add_argument("--kernel_function", type=str, default="elu", help="线性注意力的核函数")
    attn_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    
    # 测试vLLM命令
    vllm_parser = subparsers.add_parser("test_vllm", help="测试vLLM加速")
    vllm_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    vllm_parser.add_argument("--quant", type=str, choices=["none", "awq", "gptq"], default="none", help="量化方法")
    vllm_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    
    # 运行基准测试命令
    bench_parser = subparsers.add_parser("benchmark", help="运行基准测试")
    bench_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    bench_parser.add_argument("--quant", type=str, choices=["none", "awq", "gptq"], default="none", help="量化方法")
    bench_parser.add_argument("--attention", type=str, choices=["standard", "sparse", "linear"], default="standard", help="注意力机制类型")
    bench_parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    bench_parser.add_argument("--input_length", type=int, default=512, help="输入长度")
    bench_parser.add_argument("--output_length", type=int, default=128, help="输出长度")
    bench_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    bench_parser.add_argument("--save_results", action="store_true", help="是否保存结果")
    
    # 运行自动化测试命令
    auto_parser = subparsers.add_parser("auto_test", help="运行自动化测试")
    auto_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    auto_parser.add_argument("--quant_types", type=str, default="none,awq,gptq", help="量化方法列表，用逗号分隔")
    auto_parser.add_argument("--attention_types", type=str, default="standard,sparse,linear", help="注意力机制类型列表，用逗号分隔")
    auto_parser.add_argument("--batch_sizes", type=str, default="1", help="批处理大小列表，用逗号分隔")
    auto_parser.add_argument("--input_lengths", type=str, default="512,1024,2048", help="输入长度列表，用逗号分隔")
    auto_parser.add_argument("--output_lengths", type=str, default="128", help="输出长度列表，用逗号分隔")
    auto_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    auto_parser.add_argument("--save_results", action="store_true", help="是否保存结果")
    auto_parser.add_argument("--results_dir", type=str, default="data/results", help="结果保存目录")
    
    # 分析结果命令
    analyze_parser = subparsers.add_parser("analyze", help="分析结果")
    analyze_parser.add_argument("--results_dir", type=str, default="data/results", help="结果目录")
    analyze_parser.add_argument("--output_dir", type=str, default="analysis", help="输出目录")
    analyze_parser.add_argument("--metrics", type=str, default="latency,tokens_per_second,memory_usage,perplexity", help="要分析的指标，用逗号分隔")
    
    # 初始化项目命令
    init_parser = subparsers.add_parser("init", help="初始化项目")
    init_parser.add_argument("--force", action="store_true", help="强制重新初始化项目")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    logger = setup_logger(
        name="attn_experiment",
        log_dir=config["logging"]["log_dir"],
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info(f"运行命令: {args.command}")
    
    # 根据命令调用相应的脚本
    if args.command == "verify":
        from scripts.model.verify_model import main as verify_main
        sys.argv = [sys.argv[0]] + ["--model_path", args.model_path]
        if args.monitor:
            sys.argv.append("--monitor")
        verify_main()
    
    elif args.command == "quantize":
        from scripts.model.quantize_model import main as quantize_main
        sys.argv = [sys.argv[0]] + ["--model_path", args.model_path, "--quant", args.quant]
        if args.monitor:
            sys.argv.append("--monitor")
        quantize_main()
    
    elif args.command == "test_attention":
        from scripts.model.test_attention import main as test_attention_main
        sys.argv = [sys.argv[0]] + [
            "--model_path", args.model_path,
            "--attention", args.attention,
            "--sparsity", str(args.sparsity),
            "--kernel_function", args.kernel_function
        ]
        if args.monitor:
            sys.argv.append("--monitor")
        test_attention_main()
    
    elif args.command == "test_vllm":
        from scripts.model.test_vllm import main as test_vllm_main
        sys.argv = [sys.argv[0]] + ["--model_path", args.model_path, "--quant", args.quant]
        if args.monitor:
            sys.argv.append("--monitor")
        test_vllm_main()
    
    elif args.command == "benchmark":
        from scripts.benchmark.run_benchmark import main as benchmark_main
        sys.argv = [sys.argv[0]] + [
            "--model_path", args.model_path,
            "--quant", args.quant,
            "--attention", args.attention,
            "--batch_size", str(args.batch_size),
            "--input_length", str(args.input_length),
            "--output_length", str(args.output_length)
        ]
        if args.monitor:
            sys.argv.append("--monitor")
        if args.save_results:
            sys.argv.append("--save_results")
        benchmark_main()
    
    elif args.command == "auto_test":
        from scripts.benchmark.run_all_tests import main as auto_test_main
        sys.argv = [sys.argv[0]] + [
            "--model_path", args.model_path,
            "--quant_types", args.quant_types,
            "--attention_types", args.attention_types,
            "--batch_sizes", args.batch_sizes,
            "--input_lengths", args.input_lengths,
            "--output_lengths", args.output_lengths,
            "--results_dir", args.results_dir
        ]
        if args.monitor:
            sys.argv.append("--monitor")
        if args.save_results:
            sys.argv.append("--save_results")
        auto_test_main()
    
    elif args.command == "analyze":
        from scripts.analysis.analyze_results import main as analyze_main
        sys.argv = [sys.argv[0]] + [
            "--results_dir", args.results_dir,
            "--output_dir", args.output_dir,
            "--metrics", args.metrics
        ]
        analyze_main()
    
    elif args.command == "init":
        from init_project import main as init_main
        sys.argv = [sys.argv[0]]
        if args.force:
            sys.argv.append("--force")
        init_main()
    
    else:
        logger.error(f"未知命令: {args.command}")
        sys.exit(1)
    
    logger.info(f"命令 {args.command} 执行完成")

if __name__ == "__main__":
    main() 