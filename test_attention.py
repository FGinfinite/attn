"""
注意力机制验证脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, ATTENTION_CONFIG, LOGGING_CONFIG
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer, verify_model, get_model_info
from src.utils.hardware_monitor import HardwareMonitor
from src.attention.attention_utils import (
    get_attention_config, replace_attention_mechanism, get_attention_info
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="注意力机制验证脚本")
    
    parser.add_argument("--model_path", type=str, default=MODEL_CONFIG["model_name_or_path"],
                        help="模型路径或名称")
    parser.add_argument("--attention", type=str, default="standard",
                        choices=list(ATTENTION_CONFIG.keys()),
                        help="注意力机制类型")
    parser.add_argument("--sparsity", type=float, default=0.8,
                        help="稀疏注意力的稀疏度")
    parser.add_argument("--kernel_function", type=str, default="elu",
                        choices=["elu", "relu", "softmax"],
                        help="线性注意力的核函数")
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
        name="attention_test",
        log_dir=LOGGING_CONFIG["log_dir"],
        log_level=LOGGING_CONFIG["log_level"],
        log_to_file=LOGGING_CONFIG["log_to_file"],
        log_to_console=LOGGING_CONFIG["log_to_console"]
    )
    
    logger.info(f"开始验证注意力机制: {args.attention}")
    
    # 启动硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=0.5, log_dir=LOGGING_CONFIG["log_dir"])
        monitor.start()
    
    try:
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            dtype="float16"
        )
        
        # 获取模型信息
        model_info = get_model_info(model)
        logger.info(f"模型信息: {model_info}")
        
        # 替换注意力机制
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
            monitor.save_to_csv(f"attn_{args.attention}_metrics.csv")
            
            # 输出监控摘要
            summary = monitor.get_summary()
            logger.info(f"硬件监控摘要: {summary}")
    
    logger.info("验证完成")

if __name__ == "__main__":
    main() 