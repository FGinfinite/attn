"""
模型量化脚本，用于将模型量化为AWQ、GPTQ或FP16格式
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from config import load_config, DEFAULT_MODEL_PATH, LOG_DIR, AWQ_CONFIG, GPTQ_CONFIG
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer, verify_model
from src.utils.hardware_monitor import HardwareMonitor
from src.quantization.quant_utils import quantize_model, load_quantized_model, check_quantization_error

def log_gpu_memory_usage(logger, stage=""):
    """记录GPU显存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            logger.info(f"{stage} - GPU {i} 显存使用情况: {allocated:.2f} MB / {reserved:.2f} MB")
    else:
        logger.info(f"{stage} - 没有可用的GPU")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型量化脚本")
    
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径或名称")
    parser.add_argument("--quant", type=str, choices=["awq", "gptq", "fp16", "bf16"], default="awq",
                        help="量化方法")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="量化模型输出目录，默认为model_path-{quant}")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下自己。",
                        help="测试提示词")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成的最大token数量")
    parser.add_argument("--monitor", action="store_true",
                        help="是否监控硬件使用情况")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    logger = setup_logger(
        name="quantize_model",
        log_dir=LOG_DIR,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info(f"开始量化模型: {args.model_path} -> {args.quant}")
    
    # 设置输出目录
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path)
        args.output_dir = f"{model_name}-{args.quant}"
    
    logger.info(f"量化模型输出目录: {args.output_dir}")
    
    # 初始化硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=1.0, log_dir=LOG_DIR)
        monitor.start()
        logger.info("硬件监控已启动")
    
    try:
        # 记录初始GPU显存使用情况
        log_gpu_memory_usage(logger, "初始状态")
        
        # 加载原始模型和tokenizer
        logger.info("加载原始模型和tokenizer...")
        original_model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # 记录原始模型加载后的GPU显存使用情况
        log_gpu_memory_usage(logger, "原始模型加载后")
        
        # 检查原始模型的数据类型
        original_dtypes = {p.dtype for p in original_model.parameters()}
        logger.info(f"原始模型参数数据类型: {original_dtypes}")
        
        # 验证原始模型
        logger.info("验证原始模型...")
        original_output = verify_model(original_model, tokenizer, args.prompt, args.max_new_tokens)
        logger.info(f"原始模型输出: {original_output}")
        
        # 设置量化配置
        logger.info(f"设置{args.quant}量化配置...")
        if args.quant == "awq":
            quant_config = AWQ_CONFIG
        elif args.quant == "gptq":
            quant_config = GPTQ_CONFIG
        else:  # fp16
            quant_config = None
        
        # 量化模型
        logger.info(f"开始{args.quant}量化...")
        quantized_model = quantize_model(original_model, tokenizer, args.quant, quant_config, args.output_dir)
        
        # 记录量化后的GPU显存使用情况
        log_gpu_memory_usage(logger, "量化后")
        
        # 检查量化后模型的数据类型
        quantized_dtypes = {p.dtype for p in quantized_model.parameters()}
        logger.info(f"量化后模型参数数据类型: {quantized_dtypes}")
        
        # 释放原始模型内存
        logger.info("释放原始模型内存...")
        del original_model
        torch.cuda.empty_cache()
        
        # 记录释放原始模型后的GPU显存使用情况
        log_gpu_memory_usage(logger, "释放原始模型后")
        
        # 验证量化后的模型
        logger.info("验证量化后的模型...")
        quantized_output = verify_model(quantized_model, tokenizer, args.prompt, args.max_new_tokens)
        logger.info(f"量化后模型输出: {quantized_output}")
        
        # 检查量化误差
        logger.info("检查量化误差...")
        
        # 重新加载原始模型用于比较
        logger.info("重新加载原始模型用于比较...")
        original_model, _ = load_model_and_tokenizer(args.model_path)
        
        error_metrics = check_quantization_error(original_model, quantized_model, tokenizer, args.prompt)
        logger.info(f"量化误差指标: {error_metrics}")
        
        # 最终GPU显存使用情况
        log_gpu_memory_usage(logger, "最终状态")
        
        logger.info("模型量化成功")
    
    except Exception as e:
        logger.error(f"模型量化失败: {str(e)}")
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            monitor.save_to_csv(f"quant_{args.quant}_metrics.csv")
            logger.info("硬件监控已停止")
    
    logger.info("量化完成")

if __name__ == "__main__":
    main() 