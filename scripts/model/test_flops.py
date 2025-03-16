#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSpeed Flops Profiler测试脚本
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from config import load_config, LOG_DIR, DEFAULT_MODEL_PATH
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer
from src.utils.hardware_monitor import HardwareMonitor
from src.utils.flops_profiler import FlopsProfilerWrapper
from src.attention.attention_utils import replace_attention_mechanism

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FLOPs分析测试脚本")
    
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径或名称")
    parser.add_argument("--attention", type=str, choices=["standard", "sparse", "linear", "reformer", "linformer", "longformer", "realformer", "low_rank"], default="standard",
                        help="注意力机制类型")
    parser.add_argument("--input_length", type=int, default=512,
                        help="输入长度")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批处理大小")
    parser.add_argument("--output_length", type=int, default=128,
                        help="输出长度")
    parser.add_argument("--detailed", action="store_true",
                        help="是否输出详细的FLOPs分析信息")
    parser.add_argument("--dynamic", action="store_true",
                        help="是否进行动态FLOPs分析")
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
        name="test_flops",
        log_dir=LOG_DIR,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info(f"开始FLOPs分析测试: {args.model_path}")
    logger.info(f"配置: attention={args.attention}, input_length={args.input_length}, batch_size={args.batch_size}, output_length={args.output_length}")
    
    # 初始化硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=1.0, log_dir=LOG_DIR)
        monitor.start()
        logger.info("硬件监控已启动")
    
    # 初始化FLOPs分析器
    flops_profiler = FlopsProfilerWrapper(hardware_monitor=monitor)
    
    try:
        # 加载模型和tokenizer
        logger.info("加载模型和tokenizer...")
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # 检查模型数据类型
        param_dtypes = {p.dtype for p in model.parameters()}
        logger.info(f"模型参数数据类型: {param_dtypes}")
        
        # 检查模型是否在GPU上
        device_info = {p.device for p in model.parameters()}
        logger.info(f"模型参数设备: {device_info}")
        
        # 检查GPU显存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} 显存使用情况: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB / {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        
        # 替换注意力机制
        if args.attention != "standard":
            logger.info(f"替换注意力机制为: {args.attention}")
            
            # 根据注意力机制类型设置参数
            kwargs = {}
            if args.attention == "sparse":
                kwargs["sparsity"] = 0.8
            elif args.attention == "linear":
                kwargs["kernel_function"] = "elu"
            elif args.attention == "reformer":
                kwargs["num_hashes"] = 4
            elif args.attention == "linformer":
                kwargs["k_ratio"] = 0.25
                kwargs["max_seq_length"] = args.input_length
            elif args.attention == "longformer":
                kwargs["window_size"] = 128
                kwargs["global_tokens_ratio"] = 0.1
            elif args.attention == "low_rank":
                kwargs["rank_ratio"] = 0.5
            
            model = replace_attention_mechanism(model, args.attention, **kwargs)
        
        # 进行静态FLOPs分析
        logger.info("进行静态FLOPs分析...")
        input_shape = (args.batch_size, args.input_length)
        flops, macs, params, results = flops_profiler.profile_model_statistics(
            model=model,
            input_shape=input_shape,
            detailed=args.detailed,
            warm_up=2
        )
        
        logger.info(f"静态FLOPs分析结果: ")
        logger.info(f"FLOPs: {flops}")
        logger.info(f"MACs: {macs}")
        logger.info(f"参数量: {params}")
        
        # 如果需要进行动态FLOPs分析
        if args.dynamic:
            logger.info("进行动态FLOPs分析...")
            
            # 准备输入数据
            input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.input_length), device=model.device)
            attention_mask = torch.ones_like(input_ids)
            
            # 开始分析
            flops_profiler.start_profiling(model)
            
            # 前向传播
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 停止分析
            profiling_results = flops_profiler.stop_profiling(print_results=True)
            
            if profiling_results:
                logger.info("动态FLOPs分析结果:")
                for key, value in profiling_results["readable"].items():
                    logger.info(f"{key}: {value}")
        
        logger.info("FLOPs分析测试成功完成")
    
    except Exception as e:
        logger.error(f"FLOPs分析测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            logger.info("硬件监控已停止")
    
    logger.info("FLOPs分析测试结束")

if __name__ == "__main__":
    main() 