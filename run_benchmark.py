"""
基准测试脚本
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, ATTENTION_CONFIG, QUANTIZATION_CONFIG, EXPERIMENT_CONFIG, LOGGING_CONFIG
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer, get_model_info
from src.utils.hardware_monitor import HardwareMonitor
from src.attention.attention_utils import replace_attention_mechanism, get_attention_info
from src.quantization.quant_utils import load_quantized_model, get_quantized_model_info
from src.benchmark.benchmark_utils import run_benchmark

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基准测试脚本")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default=MODEL_CONFIG["model_name_or_path"],
                        help="模型路径或名称")
    
    # 量化相关参数
    parser.add_argument("--quant", type=str, default="none", choices=["none", "awq", "gptq"],
                        help="量化方式：none(FP16), awq, gptq")
    
    # 注意力机制相关参数
    parser.add_argument("--attention", type=str, default="standard", 
                        choices=list(ATTENTION_CONFIG.keys()),
                        help="注意力机制类型")
    parser.add_argument("--sparsity", type=float, default=0.8,
                        help="稀疏注意力的稀疏度")
    parser.add_argument("--kernel_function", type=str, default="elu",
                        choices=["elu", "relu", "softmax"],
                        help="线性注意力的核函数")
    
    # 基准测试相关参数
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批处理大小")
    parser.add_argument("--input_length", type=int, default=512,
                        help="输入长度")
    parser.add_argument("--output_length", type=int, default=128,
                        help="输出长度")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="运行次数")
    parser.add_argument("--warmup_runs", type=int, default=2,
                        help="预热运行次数")
    
    # 其他参数
    parser.add_argument("--monitor", action="store_true",
                        help="是否监控硬件使用情况")
    parser.add_argument("--save_results", action="store_true",
                        help="是否保存结果")
    parser.add_argument("--results_dir", type=str, default=str(EXPERIMENT_CONFIG["results_dir"]),
                        help="结果保存目录")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    os.makedirs(LOGGING_CONFIG["log_dir"], exist_ok=True)
    logger = setup_logger(
        name="benchmark",
        log_dir=LOGGING_CONFIG["log_dir"],
        log_level=LOGGING_CONFIG["log_level"],
        log_to_file=LOGGING_CONFIG["log_to_file"],
        log_to_console=LOGGING_CONFIG["log_to_console"]
    )
    
    logger.info(f"开始基准测试")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"量化方式: {args.quant}")
    logger.info(f"注意力机制: {args.attention}")
    
    # 创建结果目录
    results_dir = Path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # 启动硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=0.5, log_dir=LOGGING_CONFIG["log_dir"])
        monitor.start()
    
    try:
        # 加载模型和分词器
        if args.quant == "none":
            # 加载原始模型
            model, tokenizer = load_model_and_tokenizer(
                model_path=args.model_path,
                dtype="float16"
            )
        else:
            # 加载量化模型
            model = load_quantized_model(
                model_path=args.model_path,
                quant_type=args.quant
            )
            
            # 加载分词器
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                trust_remote_code=True
            )
        
        # 获取模型信息
        if args.quant == "none":
            model_info = get_model_info(model)
        else:
            model_info = get_quantized_model_info(model, args.quant)
        
        logger.info(f"模型信息: {model_info}")
        
        # 替换注意力机制
        if args.attention != "standard":
            kwargs = {}
            if args.attention == "sparse":
                kwargs["sparsity"] = args.sparsity
            elif args.attention == "linear":
                kwargs["kernel_function"] = args.kernel_function
            
            model = replace_attention_mechanism(
                model=model,
                attn_type=args.attention,
                **kwargs
            )
        
        # 获取注意力机制信息
        attn_info = get_attention_info(model, args.attention)
        logger.info(f"注意力机制信息: {attn_info}")
        
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
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # 添加特定类型的信息
        if args.attention == "sparse":
            benchmark_config["sparsity"] = args.sparsity
        elif args.attention == "linear":
            benchmark_config["kernel_function"] = args.kernel_function
        
        # 创建测试提示
        prompts = [
            "你好，请介绍一下自己。",
            "请解释一下量子力学的基本原理。",
            "如何使用Python进行数据分析？"
        ]
        
        # 创建困惑度测试文本
        perplexity_texts = [
            "人工智能（AI）是计算机科学的一个分支，它致力于创造能够模拟人类智能的机器。这些机器可以学习、推理、感知、规划和解决问题。人工智能的发展已经对社会产生了深远的影响，从自动驾驶汽车到语音助手，再到医疗诊断系统。",
            "量子计算是一种利用量子力学原理进行计算的技术。与传统计算机使用比特（0或1）不同，量子计算机使用量子比特，可以同时处于多种状态。这种特性使得量子计算机在某些特定问题上比传统计算机更加高效。"
        ]
        
        # 更新基准测试配置
        benchmark_config["prompts"] = prompts
        benchmark_config["perplexity_texts"] = perplexity_texts
        benchmark_config["max_new_tokens"] = args.output_length
        
        # 运行基准测试
        result = run_benchmark(
            model=model,
            tokenizer=tokenizer,
            config=benchmark_config
        )
        
        # 保存结果
        if args.save_results:
            # 创建结果文件名
            result_filename = f"{args.quant}_{args.attention}_{args.batch_size}_{args.input_length}_{args.output_length}_{benchmark_config['timestamp']}"
            
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
    
    except Exception as e:
        logger.error(f"基准测试过程中出错: {str(e)}")
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            
            # 保存监控数据
            if args.save_results:
                monitor_filename = f"monitor_{args.quant}_{args.attention}_{args.batch_size}_{args.input_length}_{args.output_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                monitor.save_to_csv(monitor_filename)
            
            # 输出监控摘要
            summary = monitor.get_summary()
            logger.info(f"硬件监控摘要: {summary}")
    
    logger.info("基准测试完成")

if __name__ == "__main__":
    main() 