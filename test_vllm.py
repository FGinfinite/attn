"""
vLLM测试脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, VLLM_CONFIG, LOGGING_CONFIG
from src.utils.logger import setup_logger
from src.utils.hardware_monitor import HardwareMonitor
from src.utils.vllm_utils import (
    check_vllm_available, load_vllm_model, generate_with_vllm, 
    measure_vllm_latency, get_vllm_model_info
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="vLLM测试脚本")
    
    parser.add_argument("--model_path", type=str, default=MODEL_CONFIG["model_name_or_path"],
                        help="模型路径或名称")
    parser.add_argument("--quant", type=str, default="none", choices=["none", "awq", "gptq"],
                        help="量化方式：none(FP16), awq, gptq")
    parser.add_argument("--tensor_parallel", type=int, 
                        default=VLLM_CONFIG["tensor_parallel_size"],
                        help="张量并行大小")
    parser.add_argument("--gpu_memory_utilization", type=float, 
                        default=VLLM_CONFIG["gpu_memory_utilization"],
                        help="GPU内存利用率")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="是否强制使用eager模式（兼容自定义注意力）")
    parser.add_argument("--prompt", type=str, default="你好，请用一句话介绍自己。",
                        help="测试提示文本")
    parser.add_argument("--max_tokens", type=int, default=20,
                        help="最大生成token数")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="运行次数")
    parser.add_argument("--warmup_runs", type=int, default=2,
                        help="预热运行次数")
    parser.add_argument("--monitor", action="store_true",
                        help="是否监控硬件使用情况")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 检查vLLM是否可用
    if not check_vllm_available():
        print("未安装vllm库，无法使用vLLM加速")
        return
    
    # 设置日志
    os.makedirs(LOGGING_CONFIG["log_dir"], exist_ok=True)
    logger = setup_logger(
        name="vllm_test",
        log_dir=LOGGING_CONFIG["log_dir"],
        log_level=LOGGING_CONFIG["log_level"],
        log_to_file=LOGGING_CONFIG["log_to_file"],
        log_to_console=LOGGING_CONFIG["log_to_console"]
    )
    
    logger.info(f"开始测试vLLM")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"量化方式: {args.quant}")
    logger.info(f"张量并行大小: {args.tensor_parallel}")
    logger.info(f"GPU内存利用率: {args.gpu_memory_utilization}")
    logger.info(f"强制使用eager模式: {args.enforce_eager}")
    
    # 启动硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=0.5, log_dir=LOGGING_CONFIG["log_dir"])
        monitor.start()
    
    try:
        # 设置量化方式
        quantization = args.quant if args.quant != "none" else None
        
        # 加载vLLM模型
        vllm_model = load_vllm_model(
            model_path=args.model_path,
            quantization=quantization,
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager
        )
        
        # 获取模型信息
        model_info = get_vllm_model_info(vllm_model)
        logger.info(f"vLLM模型信息: {model_info}")
        
        # 生成文本
        prompts = [args.prompt]
        generated_texts = generate_with_vllm(
            vllm_model=vllm_model,
            prompts=prompts,
            max_tokens=args.max_tokens
        )
        
        logger.info(f"输入: {args.prompt}")
        logger.info(f"输出: {generated_texts[0]}")
        
        # 测量延迟
        latency, tokens_per_second = measure_vllm_latency(
            vllm_model=vllm_model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs
        )
        
        logger.info(f"平均延迟: {latency:.2f} ms")
        logger.info(f"平均生成速度: {tokens_per_second:.2f} token/s")
    
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            monitor.save_to_csv(f"vllm_{args.quant}_metrics.csv")
            
            # 输出监控摘要
            summary = monitor.get_summary()
            logger.info(f"硬件监控摘要: {summary}")
    
    logger.info("测试完成")

if __name__ == "__main__":
    main() 