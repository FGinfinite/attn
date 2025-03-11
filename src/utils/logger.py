"""
日志工具模块
"""

import os
import logging
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_dir=None, log_level="INFO", log_to_file=True, log_to_console=True):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        log_level: 日志级别
        log_to_file: 是否记录到文件
        log_to_console: 是否输出到控制台
    
    Returns:
        logger: 日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)
    
    # 清除已有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    # 添加控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_to_file and log_dir:
        # 创建日志目录
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        # 创建文件处理器，指定编码为UTF-8
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 