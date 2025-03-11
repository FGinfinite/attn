"""
vLLM测试脚本，用于测试vLLM加速效果
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
from src.utils.vllm_utils import check_vllm_available, load_vllm_model, generate_with_vllm, measure_vllm_latency, get_vllm_model_info
from src.utils.hardware_monitor import HardwareMonitor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="vLLM测试脚本")
    
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径或名称")
    parser.add_argument("--quant", type=str, choices=["none", "awq", "gptq"], default="none",
                        help="量化方式")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下自己。",
                        help="测试提示词")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="生成的最大token数量")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="张量并行大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU显存利用率")
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
        name="test_vllm",
        log_dir=LOG_DIR,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info(f"开始测试vLLM: {args.model_path}")
    
    # 检查vLLM是否可用
    if not check_vllm_available():
        logger.error("vLLM不可用，请安装vLLM库")
        return
    
    # 初始化硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=1.0, log_dir=LOG_DIR)
        monitor.start()
        logger.info("硬件监控已启动")
    
    try:
        # 加载vLLM模型
        logger.info("加载vLLM模型...")
        quantization = None if args.quant == "none" else args.quant
        vllm_model = load_vllm_model(
            model_path=args.model_path,
            quantization=quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        
        # 获取模型信息
        model_info = get_vllm_model_info(vllm_model)
        logger.info(f"vLLM模型信息: {model_info}")
        
        # 生成文本
        logger.info(f"生成文本，提示词: {args.prompt}")
        outputs = generate_with_vllm(
            vllm_model=vllm_model,
            prompts=[args.prompt],
            max_tokens=args.max_tokens
        )
        
        logger.info(f"生成结果: {outputs[0]}")
        
        # 测量延迟
        logger.info("测量延迟...")
        latency_results = measure_vllm_latency(
            vllm_model=vllm_model,
            prompts=[args.prompt] * 5,  # 使用5个相同的提示词
            max_tokens=args.max_tokens
        )
        
        logger.info(f"延迟结果: {latency_results}")
        
        logger.info("vLLM测试成功")
    
    except Exception as e:
        logger.error(f"vLLM测试失败: {str(e)}")
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            monitor.save_to_csv(f"vllm_{args.quant}_metrics.csv")
            logger.info("硬件监控已停止")
    
    logger.info("测试完成")

if __name__ == "__main__":
    main() 