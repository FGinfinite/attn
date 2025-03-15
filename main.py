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
    verify_parser.add_argument("--flops_profiler", action="store_true", help="是否使用DeepSpeed的Flops Profiler分析FLOPs")
    
    # 量化模型命令
    quant_parser = subparsers.add_parser("quantize", help="量化模型")
    quant_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    quant_parser.add_argument("--quant", type=str, choices=["awq", "gptq", "fp16", "bf16"], default="awq", help="量化方法")
    quant_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    quant_parser.add_argument("--flops_profiler", action="store_true", help="是否使用DeepSpeed的Flops Profiler分析FLOPs")
    
    # 测试注意力机制命令
    attn_parser = subparsers.add_parser("test_attention", help="测试注意力机制")
    attn_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    attn_parser.add_argument("--attention", type=str, choices=["standard", "sparse", "linear", "reformer", "linformer", "longformer", "realformer"], default="standard", help="注意力机制类型")
    attn_parser.add_argument("--sparsity", type=float, default=0.8, help="稀疏注意力的稀疏度")
    attn_parser.add_argument("--kernel_function", type=str, default="elu", help="线性注意力的核函数")
    attn_parser.add_argument("--num_hashes", type=int, default=4, help="Reformer注意力的哈希数")
    attn_parser.add_argument("--k_ratio", type=float, default=0.25, help="Linformer注意力的k比例")
    attn_parser.add_argument("--window_size", type=int, default=128, help="Longformer注意力的窗口大小")
    attn_parser.add_argument("--global_tokens_ratio", type=float, default=0.1, help="Longformer注意力的全局token比例")
    attn_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    attn_parser.add_argument("--flops_profiler", action="store_true", help="是否使用DeepSpeed的Flops Profiler分析FLOPs")
    
    # 测试vLLM命令
    vllm_parser = subparsers.add_parser("test_vllm", help="测试vLLM加速")
    vllm_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    vllm_parser.add_argument("--quant", type=str, choices=["none", "awq", "gptq", "fp16", "bf16"], default="none", help="量化方法")
    vllm_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    vllm_parser.add_argument("--flops_profiler", action="store_true", help="是否使用DeepSpeed的Flops Profiler分析FLOPs")
    
    # 微调模型命令
    finetune_parser = subparsers.add_parser("finetune", help="微调模型.")
    finetune_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    finetune_parser.add_argument("--dataset_path", type=str, default="data/finetune/dataset.json", help="数据集路径")
    finetune_parser.add_argument("--output_dir", type=str, default="models/finetuned", help="微调模型输出目录")
    finetune_parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="fp16", help="训练精度")
    finetune_parser.add_argument("--max_steps", type=int, default=50, help="最大训练步数")
    finetune_parser.add_argument("--batch_size", type=int, default=1, help="训练批次大小")
    finetune_parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    finetune_parser.add_argument("--generate_dataset", action="store_true", help="是否生成示例数据集")
    finetune_parser.add_argument("--dataset_size", type=int, default=100, help="生成的数据集大小")
    finetune_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    finetune_parser.add_argument("--flops_profiler", action="store_true", help="是否使用DeepSpeed的Flops Profiler分析FLOPs")
    finetune_parser.add_argument("--profile_frequency", type=int, default=5, help="FLOPs分析频率（每N步分析一次）")
    
    # 运行基准测试命令
    bench_parser = subparsers.add_parser("benchmark", help="运行基准测试")
    bench_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    bench_parser.add_argument("--quant", type=str, choices=["none", "awq", "gptq", "fp16", "bf16"], default="none", help="量化方法")
    bench_parser.add_argument("--attention", type=str, choices=["standard", "sparse", "linear", "reformer", "linformer", "longformer", "realformer"], default="standard", help="注意力机制类型")
    bench_parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    bench_parser.add_argument("--input_length", type=int, default=512, help="输入长度")
    bench_parser.add_argument("--output_length", type=int, default=128, help="输出长度")
    bench_parser.add_argument("--sparsity", type=float, default=0.8, help="稀疏注意力的稀疏度")
    bench_parser.add_argument("--kernel_function", type=str, default="elu", help="线性注意力的核函数")
    bench_parser.add_argument("--num_hashes", type=int, default=4, help="Reformer注意力的哈希数")
    bench_parser.add_argument("--k_ratio", type=float, default=0.25, help="Linformer注意力的k比例")
    bench_parser.add_argument("--window_size", type=int, default=128, help="Longformer注意力的窗口大小")
    bench_parser.add_argument("--global_tokens_ratio", type=float, default=0.1, help="Longformer注意力的全局token比例")
    bench_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    bench_parser.add_argument("--flops_profiler", action="store_true", help="是否使用DeepSpeed的Flops Profiler分析FLOPs")
    bench_parser.add_argument("--save_results", action="store_true", help="是否保存结果")
    bench_parser.add_argument("--max_test_cases", type=int, default=-1, help="最大测试用例数量，设置为None或负数表示使用所有测试用例")
    
    # 运行自动化测试命令
    auto_parser = subparsers.add_parser("auto_test", help="运行自动化测试")
    auto_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    auto_parser.add_argument("--quant_types", type=str, default="none,awq,gptq", help="量化方法列表，用逗号分隔")
    auto_parser.add_argument("--attention_types", type=str, default="standard,sparse,linear,reformer,linformer,longformer,realformer", help="注意力机制类型列表，用逗号分隔")
    auto_parser.add_argument("--batch_sizes", type=str, default="1", help="批处理大小列表，用逗号分隔")
    auto_parser.add_argument("--input_lengths", type=str, default="512,1024,2048", help="输入长度列表，用逗号分隔")
    auto_parser.add_argument("--output_lengths", type=str, default="128", help="输出长度列表，用逗号分隔")
    auto_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    auto_parser.add_argument("--flops_profiler", action="store_true", help="是否使用DeepSpeed的Flops Profiler分析FLOPs")
    auto_parser.add_argument("--save_results", action="store_true", help="是否保存结果")
    auto_parser.add_argument("--results_dir", type=str, default="data/results", help="结果保存目录")
    
    # 分析结果命令
    analyze_parser = subparsers.add_parser("analyze", help="分析结果")
    analyze_parser.add_argument("--results_dir", type=str, default="data/results", help="结果目录")
    analyze_parser.add_argument("--output_dir", type=str, default="analysis", help="输出目录")
    analyze_parser.add_argument("--metrics", type=str, default="latency,tokens_per_second,memory_usage,perplexity,flops_flops,flops_macs,flops_params,flops_per_second", help="要分析的指标，用逗号分隔")
    
    # 测试微调模型命令
    test_finetune_parser = subparsers.add_parser("test_finetune", help="测试微调前后的模型效果差异")
    test_finetune_parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="基础模型路径")
    test_finetune_parser.add_argument("--finetuned_model", type=str, default="models/finetuned/final", help="微调模型路径")
    test_finetune_parser.add_argument("--dataset_path", type=str, default="data/finetune/dataset.json", help="数据集路径")
    test_finetune_parser.add_argument("--num_samples", type=int, default=3, help="测试样本数量")
    test_finetune_parser.add_argument("--output_file", type=str, default="test_results.txt", help="输出结果文件")
    test_finetune_parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="fp16", help="模型精度")
    test_finetune_parser.add_argument("--merge_weights", action="store_true", help="是否合并LoRA权重以加快推理速度")
    
    # 测试FLOPs命令
    test_flops_parser = subparsers.add_parser("test_flops", help="测试FLOPs分析功能")
    test_flops_parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    test_flops_parser.add_argument("--attention", type=str, choices=["standard", "sparse", "linear", "reformer", "linformer", "longformer", "realformer"], default="standard", help="注意力机制类型")
    test_flops_parser.add_argument("--input_length", type=int, default=512, help="输入长度")
    test_flops_parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    test_flops_parser.add_argument("--output_length", type=int, default=128, help="输出长度")
    test_flops_parser.add_argument("--detailed", action="store_true", help="是否输出详细的FLOPs分析信息")
    test_flops_parser.add_argument("--dynamic", action="store_true", help="是否进行动态FLOPs分析")
    test_flops_parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    
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
        if args.flops_profiler:
            sys.argv.append("--flops_profiler")
        verify_main()
    
    elif args.command == "quantize":
        from scripts.model.quantize_model import main as quantize_main
        sys.argv = [sys.argv[0]] + ["--model_path", args.model_path, "--quant", args.quant]
        if args.monitor:
            sys.argv.append("--monitor")
        if args.flops_profiler:
            sys.argv.append("--flops_profiler")
        quantize_main()
    
    elif args.command == "test_attention":
        from scripts.model.test_attention import main as test_attention_main
        sys.argv = [sys.argv[0]] + [
            "--model_path", args.model_path,
            "--attention", args.attention,
            "--sparsity", str(args.sparsity),
            "--kernel_function", args.kernel_function,
            "--num_hashes", str(args.num_hashes),
            "--k_ratio", str(args.k_ratio),
            "--window_size", str(args.window_size),
            "--global_tokens_ratio", str(args.global_tokens_ratio)
        ]
        if args.monitor:
            sys.argv.append("--monitor")
        if args.flops_profiler:
            sys.argv.append("--flops_profiler")
        test_attention_main()
    
    elif args.command == "test_vllm":
        from scripts.model.test_vllm import main as test_vllm_main
        sys.argv = [sys.argv[0]] + ["--model_path", args.model_path, "--quant", args.quant]
        if args.monitor:
            sys.argv.append("--monitor")
        if args.flops_profiler:
            sys.argv.append("--flops_profiler")
        test_vllm_main()
    
    elif args.command == "finetune":
        from scripts.model.finetune_model import main as finetune_main
        sys.argv = [sys.argv[0]] + [
            "--model_path", args.model_path,
            "--dataset_path", args.dataset_path,
            "--output_dir", args.output_dir,
            "--precision", args.precision,
            "--max_steps", str(args.max_steps),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate)
        ]
        if args.generate_dataset:
            sys.argv.append("--generate_dataset")
            sys.argv.extend(["--dataset_size", str(args.dataset_size)])
        if args.monitor:
            sys.argv.append("--monitor")
        if args.flops_profiler:
            sys.argv.append("--flops_profiler")
        if hasattr(args, "profile_frequency"):
            sys.argv.extend(["--profile_frequency", str(args.profile_frequency)])
        finetune_main()
    
    elif args.command == "benchmark":
        from scripts.benchmark.run_benchmark import main as benchmark_main
        sys.argv = [sys.argv[0]] + [
            "--model_path", args.model_path,
            "--quant", args.quant,
            "--attention", args.attention,
            "--batch_size", str(args.batch_size),
            "--input_length", str(args.input_length),
            "--output_length", str(args.output_length),
            "--max_test_cases", str(args.max_test_cases)
        ]
        
        # 根据注意力机制类型添加特定参数
        if args.attention == "sparse":
            sys.argv.extend(["--sparsity", str(args.sparsity)])
        elif args.attention == "linear":
            sys.argv.extend(["--kernel_function", args.kernel_function])
        elif args.attention == "reformer":
            sys.argv.extend(["--num_hashes", str(args.num_hashes)])
        elif args.attention == "linformer":
            sys.argv.extend(["--k_ratio", str(args.k_ratio)])
        elif args.attention == "longformer":
            sys.argv.extend(["--window_size", str(args.window_size), 
                            "--global_tokens_ratio", str(args.global_tokens_ratio)])
        
        if args.monitor:
            sys.argv.append("--monitor")
        if args.flops_profiler:
            sys.argv.append("--flops_profiler")
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
        if args.flops_profiler:
            sys.argv.append("--flops_profiler")
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
    
    elif args.command == "test_finetune":
        from scripts.model.test_finetuned_model import main as test_finetune_main
        sys.argv = [sys.argv[0]] + [
            "--base_model", args.base_model,
            "--finetuned_model", args.finetuned_model,
            "--dataset_path", args.dataset_path,
            "--num_samples", str(args.num_samples),
            "--output_file", args.output_file,
            "--precision", args.precision
        ]
        if args.merge_weights:
            sys.argv.append("--merge_weights")
        test_finetune_main()
    
    elif args.command == "test_flops":
        from scripts.model.test_flops import main as test_flops_main
        sys.argv = [sys.argv[0]] + [
            "--model_path", args.model_path,
            "--attention", args.attention,
            "--input_length", str(args.input_length),
            "--batch_size", str(args.batch_size),
            "--output_length", str(args.output_length)
        ]
        if args.detailed:
            sys.argv.append("--detailed")
        if args.dynamic:
            sys.argv.append("--dynamic")
        if args.monitor:
            sys.argv.append("--monitor")
        test_flops_main()
    
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