"""
量化模型验证脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, QUANTIZATION_CONFIG, LOGGING_CONFIG
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer, verify_model, get_model_info
from src.utils.hardware_monitor import HardwareMonitor
from src.quantization.quant_utils import (
    quantize_model, load_quantized_model, get_quantized_model_info, check_quantization_error
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="量化模型验证脚本")
    
    parser.add_argument("--model_path", type=str, default=MODEL_CONFIG["model_name_or_path"],
                        help="模型路径或名称")
    parser.add_argument("--quant", type=str, default="awq", choices=["awq", "gptq"],
                        help="量化方式：awq, gptq")
    parser.add_argument("--output_dir", type=str, default="./quantized_models",
                        help="量化模型输出目录")
    parser.add_argument("--prompt", type=str, default="你好，请用一句话介绍自己。",
                        help="测试提示文本")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="最大生成token数")
    parser.add_argument("--monitor", action="store_true",
                        help="是否监控硬件使用情况")
    parser.add_argument("--skip_quantize", action="store_true",
                        help="是否跳过量化过程，直接加载已量化的模型")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    os.makedirs(LOGGING_CONFIG["log_dir"], exist_ok=True)
    logger = setup_logger(
        name="quantize_model",
        log_dir=LOGGING_CONFIG["log_dir"],
        log_level=LOGGING_CONFIG["log_level"],
        log_to_file=LOGGING_CONFIG["log_to_file"],
        log_to_console=LOGGING_CONFIG["log_to_console"]
    )
    
    logger.info(f"开始量化模型: {args.model_path}")
    logger.info(f"量化方式: {args.quant}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / f"{args.quant}_models" / Path(args.model_path).name
    os.makedirs(output_dir, exist_ok=True)
    
    # 启动硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=0.5, log_dir=LOGGING_CONFIG["log_dir"])
        monitor.start()
    
    try:
        # 加载原始模型和分词器
        original_model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            dtype="float16"
        )
        
        # 获取原始模型信息
        original_model_info = get_model_info(original_model)
        logger.info(f"原始模型信息: {original_model_info}")
        
        # 验证原始模型
        success, original_output = verify_model(
            model=original_model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens
        )
        
        if not success:
            logger.error(f"原始模型验证失败: {original_output}")
            return
        
        # 量化模型
        if not args.skip_quantize:
            logger.info(f"开始量化模型")
            
            # 获取量化配置
            quant_config = QUANTIZATION_CONFIG[args.quant]
            
            # 量化模型
            quantized_model = quantize_model(
                model=original_model,
                tokenizer=tokenizer,
                quant_type=args.quant,
                quant_config=quant_config,
                output_dir=output_dir
            )
        else:
            logger.info(f"跳过量化过程，直接加载已量化的模型")
            
            # 加载已量化的模型
            quantized_model = load_quantized_model(
                model_path=output_dir,
                quant_type=args.quant
            )
        
        # 获取量化模型信息
        quantized_model_info = get_quantized_model_info(quantized_model, args.quant)
        logger.info(f"量化模型信息: {quantized_model_info}")
        
        # 验证量化模型
        success, quantized_output = verify_model(
            model=quantized_model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens
        )
        
        if not success:
            logger.error(f"量化模型验证失败: {quantized_output}")
            return
        
        # 检查量化误差
        similarity = check_quantization_error(
            original_model=original_model,
            quantized_model=quantized_model,
            tokenizer=tokenizer,
            prompt=args.prompt
        )
        
        logger.info(f"量化误差（余弦相似度）: {similarity}")
        logger.info(f"原始模型输出: {original_output}")
        logger.info(f"量化模型输出: {quantized_output}")
    
    except Exception as e:
        logger.error(f"量化过程中出错: {str(e)}")
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            monitor.save_to_csv(f"quant_{args.quant}_metrics.csv")
            
            # 输出监控摘要
            summary = monitor.get_summary()
            logger.info(f"硬件监控摘要: {summary}")
    
    logger.info("量化完成")

if __name__ == "__main__":
    main() 