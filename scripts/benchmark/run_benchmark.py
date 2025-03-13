"""
基准测试脚本，用于测试模型在不同配置下的性能
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from config import load_config, DEFAULT_MODEL_PATH, LOG_DIR, RESULTS_DIR, SUPPORTED_ATTENTION_TYPES
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer
from src.utils.hardware_monitor import HardwareMonitor
from src.utils.flops_profiler import FlopsProfilerWrapper
from src.attention.attention_utils import replace_attention_mechanism
from src.quantization.quant_utils import load_quantized_model
from src.benchmark.benchmark_utils import run_benchmark, BenchmarkResult

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基准测试脚本")
    
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径或名称")
    parser.add_argument("--quant", type=str, choices=["none", "awq", "gptq", "fp16", "bf16"], default="none",
                        help="量化方法")
    parser.add_argument("--attention", type=str, choices=SUPPORTED_ATTENTION_TYPES, default="standard",
                        help="注意力机制类型")
    parser.add_argument("--sparsity", type=float, default=0.8,
                        help="稀疏注意力的稀疏度")
    parser.add_argument("--kernel_function", type=str, default="elu",
                        help="线性注意力的核函数")
    parser.add_argument("--num_hashes", type=int, default=4,
                        help="Reformer注意力的哈希数")
    parser.add_argument("--k_ratio", type=float, default=0.25,
                        help="Linformer注意力的k比例")
    parser.add_argument("--window_size", type=int, default=128,
                        help="Longformer注意力的窗口大小")
    parser.add_argument("--global_tokens_ratio", type=float, default=0.1,
                        help="Longformer注意力的全局token比例")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批处理大小")
    parser.add_argument("--input_length", type=int, default=512,
                        help="输入长度")
    parser.add_argument("--output_length", type=int, default=128,
                        help="输出长度")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="运行次数")
    parser.add_argument("--warmup_runs", type=int, default=2,
                        help="预热运行次数")
    parser.add_argument("--max_test_cases", type=int, default=-1,
                        help="最大测试用例数量，None或负数表示使用所有测试用例")
    parser.add_argument("--monitor", action="store_true",
                        help="是否监控硬件使用情况")
    parser.add_argument("--flops_profiler", action="store_true",
                        help="是否使用DeepSpeed的Flops Profiler分析FLOPs")
    parser.add_argument("--save_results", action="store_true",
                        help="是否保存结果")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                        help="结果保存目录")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    logger = setup_logger(
        name="run_benchmark",
        log_dir=LOG_DIR,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info(f"开始基准测试: {args.model_path}")
    logger.info(f"配置: quant={args.quant}, attention={args.attention}, batch_size={args.batch_size}, input_length={args.input_length}, output_length={args.output_length}")
    
    # 初始化硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=1.0, log_dir=LOG_DIR)
        monitor.start()
        logger.info("硬件监控已启动")
    
    # 初始化FLOPs分析器
    flops_profiler = None
    if args.flops_profiler:
        flops_profiler = FlopsProfilerWrapper(hardware_monitor=monitor)
        logger.info("FLOPs分析器已初始化")
    
    try:
        # 加载模型和tokenizer
        logger.info("加载模型和tokenizer...")
        if args.quant == "none":
            model, tokenizer = load_model_and_tokenizer(args.model_path)
        else:
            model = load_quantized_model(args.model_path, args.quant)
            # 加载tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        # 检查模型数据类型
        param_dtypes = {p.dtype for p in model.parameters()}
        logger.info(f"模型参数数据类型: {param_dtypes}")
        
        # 检查模型是否在GPU上
        device_info = {p.device for p in model.parameters()}
        logger.info(f"模型参数设备: {device_info}")
        
        # 检查GPU显存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} 显存使用情况: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB / {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        
        # 替换注意力机制
        if args.attention != "standard":
            logger.info(f"替换注意力机制为: {args.attention}")
            kwargs = {}
            if args.attention == "sparse":
                kwargs["sparsity"] = args.sparsity
            elif args.attention == "linear":
                kwargs["kernel_function"] = args.kernel_function
            elif args.attention == "reformer":
                kwargs["num_hashes"] = args.num_hashes
            elif args.attention == "linformer":
                kwargs["k_ratio"] = args.k_ratio
                kwargs["max_seq_length"] = args.input_length  # 使用输入长度作为最大序列长度
            elif args.attention == "longformer":
                kwargs["window_size"] = args.window_size
                kwargs["global_tokens_ratio"] = args.global_tokens_ratio
            
            model = replace_attention_mechanism(model, args.attention, **kwargs)
        
        # 如果启用了FLOPs分析器，进行静态分析
        if flops_profiler:
            logger.info("开始进行模型FLOPs静态分析...")
            # 设置适当的输入形状，考虑到batch_size和input_length
            input_shape = (args.batch_size, args.input_length)
            flops, macs, params, results = flops_profiler.profile_model_statistics(
                model=model,
                input_shape=input_shape,
                detailed=True,
                warm_up=2
            )
            logger.info(f"模型静态分析结果: FLOPs={flops}, MACs={macs}, 参数量={params}")
        
        # 创建基准测试配置
        benchmark_config = {
            "model_path": args.model_path,
            "quant": args.quant,
            "attention": args.attention,
            "batch_size": args.batch_size,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "num_runs": args.num_runs,
            "warmup_runs": args.warmup_runs,
            "max_test_cases": args.max_test_cases,
            "prompts": ["你好，请介绍一下自己。"],  # 添加默认提示词
            "max_new_tokens": args.output_length,  # 使用output_length作为max_new_tokens
            "model_config": {
                "quantization": args.quant,
                "attention_type": args.attention
            }
        }
        
        # 运行基准测试
        logger.info("运行基准测试...")
        # 如果启用了FLOPs分析器，在运行基准测试之前启动分析
        if flops_profiler:
            flops_profiler.start_profiling(model)
            
        result = run_benchmark(model, tokenizer, benchmark_config, hardware_monitor=monitor)
        
        # 如果启用了FLOPs分析器，停止分析并获取结果
        if flops_profiler:
            profiling_results = flops_profiler.stop_profiling(print_results=True)
            if profiling_results:
                # 将FLOPs分析结果添加到基准测试结果中
                for key, value in profiling_results["numeric"].items():
                    result.add_metric(f"flops_{key}", value)
                
                # 将可读的结果打印到日志
                logger.info("FLOPs分析结果:")
                for key, value in profiling_results["readable"].items():
                    logger.info(f"{key}: {value}")
        
        # 输出结果
        logger.info("基准测试结果:")
        for metric, value in result.summary.items():
            logger.info(f"{metric}: {value}")
        
        # 保存结果
        if args.save_results:
            logger.info("保存结果...")
            
            # 创建结果目录
            results_dir = Path(args.results_dir)
            os.makedirs(results_dir, exist_ok=True)
            
            # 创建结果文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"benchmark_{args.quant}_{args.attention}_{args.batch_size}_{args.input_length}_{args.output_length}_{timestamp}"
            
            # 保存JSON结果
            json_path = results_dir / f"{result_filename}.json"
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"结果已保存到: {json_path}")
            
            # 保存CSV结果
            csv_path = results_dir / f"{result_filename}.csv"
            df = pd.DataFrame({
                "metric": list(result.summary.keys()),
                "value": list(result.summary.values())
            })
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            logger.info(f"摘要已保存到: {csv_path}")
            
            # 如果有FLOPs分析结果，单独保存
            if flops_profiler and hasattr(flops_profiler, 'model_stats') and flops_profiler.model_stats:
                flops_path = results_dir / f"{result_filename}_flops.json"
                with open(flops_path, "w", encoding='utf-8') as f:
                    json.dump(flops_profiler.model_stats, f, indent=2, ensure_ascii=False)
                logger.info(f"FLOPs分析结果已保存到: {flops_path}")
        
        logger.info("基准测试成功")
    
    except Exception as e:
        logger.error(f"基准测试失败: {str(e)}")
        # 导入traceback来打印完整的错误堆栈
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            
            # 保存监控数据
            if args.save_results:
                monitor_filename = f"monitor_{args.quant}_{args.attention}_{args.batch_size}_{args.input_length}_{args.output_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                monitor.save_to_csv(monitor_filename)
            
            logger.info("硬件监控已停止")
        
        # 确保FLOPs分析器已停止
        if flops_profiler and hasattr(flops_profiler, 'flops_profiler') and flops_profiler.flops_profiler:
            try:
                flops_profiler.stop_profiling(print_results=False)
            except Exception as e:
                logger.error(f"停止FLOPs分析器出错: {str(e)}")
    
    logger.info("基准测试完成")

if __name__ == "__main__":
    main() 