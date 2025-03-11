"""
自动化测试脚本，用于运行所有组合的测试
"""

import os
import sys
import time
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, ATTENTION_CONFIG, EXPERIMENT_CONFIG, LOGGING_CONFIG
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="自动化测试脚本")
    
    parser.add_argument("--model_path", type=str, default=MODEL_CONFIG["model_name_or_path"],
                        help="模型路径或名称")
    parser.add_argument("--quant_types", type=str, default="none,awq,gptq",
                        help="量化方式列表，用逗号分隔")
    parser.add_argument("--attention_types", type=str, default="standard,sparse,linear",
                        help="注意力机制类型列表，用逗号分隔")
    parser.add_argument("--batch_sizes", type=str, default="1",
                        help="批处理大小列表，用逗号分隔")
    parser.add_argument("--input_lengths", type=str, default="512,1024,2048",
                        help="输入长度列表，用逗号分隔")
    parser.add_argument("--output_lengths", type=str, default="128",
                        help="输出长度列表，用逗号分隔")
    parser.add_argument("--use_vllm", action="store_true",
                        help="是否使用vLLM加速")
    parser.add_argument("--monitor", action="store_true",
                        help="是否监控硬件使用情况")
    parser.add_argument("--save_results", action="store_true",
                        help="是否保存结果")
    parser.add_argument("--results_dir", type=str, default=str(EXPERIMENT_CONFIG["results_dir"]),
                        help="结果保存目录")
    
    return parser.parse_args()

def run_command(command):
    """运行命令"""
    print(f"运行命令: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"命令执行失败，返回码: {process.returncode}")
            print(f"错误信息: {stderr}")
            return False
        
        return True
    
    except Exception as e:
        print(f"命令执行出错: {str(e)}")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    os.makedirs(LOGGING_CONFIG["log_dir"], exist_ok=True)
    logger = setup_logger(
        name="run_all_tests",
        log_dir=LOGGING_CONFIG["log_dir"],
        log_level=LOGGING_CONFIG["log_level"],
        log_to_file=LOGGING_CONFIG["log_to_file"],
        log_to_console=LOGGING_CONFIG["log_to_console"]
    )
    
    logger.info(f"开始自动化测试")
    
    # 解析参数
    quant_types = args.quant_types.split(",")
    attention_types = args.attention_types.split(",")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    input_lengths = [int(x) for x in args.input_lengths.split(",")]
    output_lengths = [int(x) for x in args.output_lengths.split(",")]
    
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"量化方式: {quant_types}")
    logger.info(f"注意力机制: {attention_types}")
    logger.info(f"批处理大小: {batch_sizes}")
    logger.info(f"输入长度: {input_lengths}")
    logger.info(f"输出长度: {output_lengths}")
    logger.info(f"使用vLLM: {args.use_vllm}")
    
    # 创建结果目录
    results_dir = Path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建测试记录文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_record_file = os.path.join(args.results_dir, f"test_record_{timestamp}.txt")
    with open(test_record_file, "w", encoding='utf-8') as f:
        f.write(f"自动化测试记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {args.model_path}\n")
        f.write(f"量化类型: {args.quant_types}\n")
        f.write(f"注意力类型: {args.attention_types}\n")
        f.write(f"批处理大小: {args.batch_sizes}\n")
        f.write(f"输入长度: {args.input_lengths}\n")
        f.write(f"输出长度: {args.output_lengths}\n")
        f.write("-" * 50 + "\n\n")
    
    # 运行测试
    total_tests = len(quant_types) * len(attention_types) * len(batch_sizes) * len(input_lengths) * len(output_lengths)
    completed_tests = 0
    
    logger.info(f"总测试数量: {total_tests}")
    
    for quant in quant_types:
        for attention in attention_types:
            for batch_size in batch_sizes:
                for input_length in input_lengths:
                    for output_length in output_lengths:
                        # 构建命令
                        if args.use_vllm:
                            command = f"python test_vllm.py --model_path {args.model_path} --quant {quant} --prompt \"请介绍一下自己\" --max_tokens {output_length}"
                            
                            if args.monitor:
                                command += " --monitor"
                        else:
                            command = f"python run_benchmark.py --model_path {args.model_path} --quant {quant} --attention {attention} --batch_size {batch_size} --input_length {input_length} --output_length {output_length}"
                            
                            if args.monitor:
                                command += " --monitor"
                            
                            if args.save_results:
                                command += " --save_results"
                            
                            if args.results_dir:
                                command += f" --results_dir {args.results_dir}"
                        
                        # 记录测试信息
                        test_info = f"测试 {completed_tests + 1}/{total_tests}: quant={quant}, attention={attention}, batch_size={batch_size}, input_length={input_length}, output_length={output_length}"
                        logger.info(test_info)
                        
                        with open(test_record_file, "a", encoding='utf-8') as f:
                            f.write(f"{test_info}\n")
                            f.write(f"命令: {command}\n")
                        
                        # 运行命令
                        start_time = time.time()
                        success = run_command(command)
                        end_time = time.time()
                        
                        # 记录结果
                        elapsed_time = end_time - start_time
                        result = "成功" if success else "失败"
                        
                        with open(test_record_file, "a", encoding='utf-8') as f:
                            f.write(f"结果: {result}\n")
                            f.write(f"耗时: {elapsed_time:.2f}秒\n")
                            f.write(f"\n")
                        
                        completed_tests += 1
                        logger.info(f"完成测试 {completed_tests}/{total_tests}")
                        
                        # 等待一段时间，避免GPU过热
                        time.sleep(5)
    
    # 记录测试完成
    with open(test_record_file, "a", encoding='utf-8') as f:
        f.write(f"\n所有测试完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总测试数: {total_tests}\n")
        f.write(f"成功: {total_tests - sum(not success for success in [run_command(command) for quant in quant_types for attention in attention_types for batch_size in batch_sizes for input_length in input_lengths for output_length in output_lengths])}\n")
        f.write(f"失败: {sum(not success for success in [run_command(command) for quant in quant_types for attention in attention_types for batch_size in batch_sizes for input_length in input_lengths for output_length in output_lengths])}\n")
    
    logger.info(f"自动化测试完成，总测试数量: {total_tests}，测试记录已保存到: {test_record_file}")

if __name__ == "__main__":
    main() 