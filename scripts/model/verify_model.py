"""
模型验证脚本，用于验证模型的基本功能
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

from config import load_config, DEFAULT_MODEL_PATH, LOG_DIR
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer, verify_model
from src.utils.hardware_monitor import HardwareMonitor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型验证脚本")
    
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径或名称")
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
        name="verify_model",
        log_dir=LOG_DIR,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info(f"开始验证模型: {args.model_path}")
    
    # 初始化硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=1.0, log_dir=LOG_DIR)
        monitor.start()
        logger.info("硬件监控已启动")
    
    try:
        # 加载模型和tokenizer
        logger.info("加载模型和tokenizer...")
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # 验证模型
        logger.info(f"验证模型，提示词: {args.prompt}")
        output = verify_model(model, tokenizer, args.prompt, args.max_new_tokens)
        
        logger.info(f"模型输出: {output}")
        logger.info("模型验证成功")
    
    except Exception as e:
        logger.error(f"模型验证失败: {str(e)}")
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            monitor.save_to_csv()
            logger.info("硬件监控已停止")
    
    logger.info("验证完成")

if __name__ == "__main__":
    main() 