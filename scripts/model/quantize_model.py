"""
模型量化脚本，用于将模型量化为AWQ或GPTQ格式
"""

import os
import sys
import argparse
import logging
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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型量化脚本")
    
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径或名称")
    parser.add_argument("--quant", type=str, choices=["awq", "gptq"], default="awq",
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
        # 加载原始模型和tokenizer
        logger.info("加载原始模型和tokenizer...")
        original_model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # 验证原始模型
        logger.info("验证原始模型...")
        original_output = verify_model(original_model, tokenizer, args.prompt, args.max_new_tokens)
        logger.info(f"原始模型输出: {original_output}")
        
        # 设置量化配置
        logger.info(f"设置{args.quant}量化配置...")
        if args.quant == "awq":
            quant_config = AWQ_CONFIG
        else:  # gptq
            quant_config = GPTQ_CONFIG
        
        # 量化模型
        logger.info(f"开始{args.quant}量化...")
        quantize_model(original_model, tokenizer, args.quant, quant_config, args.output_dir)
        
        # 加载量化后的模型
        logger.info("加载量化后的模型...")
        quantized_model = load_quantized_model(args.output_dir, args.quant)
        
        # 验证量化后的模型
        logger.info("验证量化后的模型...")
        quantized_output = verify_model(quantized_model, tokenizer, args.prompt, args.max_new_tokens)
        logger.info(f"量化后模型输出: {quantized_output}")
        
        # 检查量化误差
        logger.info("检查量化误差...")
        error_metrics = check_quantization_error(original_model, quantized_model, tokenizer, args.prompt)
        logger.info(f"量化误差指标: {error_metrics}")
        
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