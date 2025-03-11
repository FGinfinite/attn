"""
注意力机制测试脚本，用于测试不同注意力机制的效果
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

from config import load_config, DEFAULT_MODEL_PATH, LOG_DIR, SUPPORTED_ATTENTION_TYPES
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer, verify_model
from src.utils.hardware_monitor import HardwareMonitor
from src.attention.attention_utils import replace_attention_mechanism, get_attention_info

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="注意力机制测试脚本")
    
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径或名称")
    parser.add_argument("--attention", type=str, choices=SUPPORTED_ATTENTION_TYPES, default="standard",
                        help="注意力机制类型")
    parser.add_argument("--sparsity", type=float, default=0.8,
                        help="稀疏注意力的稀疏度")
    parser.add_argument("--kernel_function", type=str, default="elu",
                        help="线性注意力的核函数")
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
        name="test_attention",
        log_dir=LOG_DIR,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info(f"开始测试注意力机制: {args.attention}")
    
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
        
        # 替换注意力机制
        logger.info(f"替换注意力机制为: {args.attention}")
        kwargs = {}
        if args.attention == "sparse":
            kwargs["sparsity"] = args.sparsity
        elif args.attention == "linear":
            kwargs["kernel_function"] = args.kernel_function
        
        model = replace_attention_mechanism(model, args.attention, **kwargs)
        
        # 获取注意力机制信息
        attention_info = get_attention_info(model, args.attention)
        logger.info(f"注意力机制信息: {attention_info}")
        
        # 验证模型
        logger.info(f"验证模型，提示词: {args.prompt}")
        output = verify_model(model, tokenizer, args.prompt, args.max_new_tokens)
        
        logger.info(f"模型输出: {output}")
        logger.info("注意力机制测试成功")
    
    except Exception as e:
        logger.error(f"注意力机制测试失败: {str(e)}")
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            monitor.save_to_csv(f"attn_{args.attention}_metrics.csv")
            logger.info("硬件监控已停止")
    
    logger.info("测试完成")

if __name__ == "__main__":
    main() 