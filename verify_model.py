"""
模型验证脚本，用于验证模型是否能正常加载和运行
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, LOGGING_CONFIG
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer, verify_model, get_model_info
from src.utils.hardware_monitor import HardwareMonitor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型验证脚本")
    
    parser.add_argument("--model_path", type=str, default=MODEL_CONFIG["model_name_or_path"],
                        help="模型路径或名称")
    parser.add_argument("--dtype", type=str, default=MODEL_CONFIG["dtype"],
                        help="数据类型")
    parser.add_argument("--prompt", type=str, default="你好，请用一句话介绍自己。",
                        help="测试提示文本")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="最大生成token数")
    parser.add_argument("--monitor", action="store_true",
                        help="是否监控硬件使用情况")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    os.makedirs(LOGGING_CONFIG["log_dir"], exist_ok=True)
    logger = setup_logger(
        name="model_verification",
        log_dir=LOGGING_CONFIG["log_dir"],
        log_level=LOGGING_CONFIG["log_level"],
        log_to_file=LOGGING_CONFIG["log_to_file"],
        log_to_console=LOGGING_CONFIG["log_to_console"]
    )
    
    logger.info(f"开始验证模型: {args.model_path}")
    
    # 启动硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=0.5, log_dir=LOGGING_CONFIG["log_dir"])
        monitor.start()
    
    try:
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            dtype=args.dtype
        )
        
        # 获取模型信息
        model_info = get_model_info(model)
        logger.info(f"模型信息: {model_info}")
        
        # 验证模型
        success, output = verify_model(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens
        )
        
        if success:
            logger.info("模型验证成功")
            logger.info(f"输入: {args.prompt}")
            logger.info(f"输出: {output}")
        else:
            logger.error(f"模型验证失败: {output}")
    
    except Exception as e:
        logger.error(f"验证过程中出错: {str(e)}")
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            monitor.save_to_csv()
            
            # 输出监控摘要
            summary = monitor.get_summary()
            logger.info(f"硬件监控摘要: {summary}")
    
    logger.info("验证完成")

if __name__ == "__main__":
    main() 