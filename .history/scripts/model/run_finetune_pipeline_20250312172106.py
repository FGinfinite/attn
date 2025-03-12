#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行完整的微调流程：生成数据集 -> 微调模型
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import load_config
from src.utils.logger import setup_logger

logger = logging.getLogger("attn_experiment")

def run_finetune_pipeline(
    model_path: str = "Qwen/Qwen2.5-3B-Instruct",
    num_samples: int = 100,
    dataset_path: str = "data/finetune/instruction_dataset.json",
    output_dir: str = "models/finetune",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 1024,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_fp16: bool = True,
    monitor: bool = True,
    monitor_interval: float = 1.0,
    save_steps: int = 10,
    logging_steps: int = 1,
    warmup_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    seed: int = 42
):
    """
    运行完整的微调流程
    
    Args:
        model_path: 模型路径
        num_samples: 生成的样本数量
        dataset_path: 数据集路径
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批处理大小
        learning_rate: 学习率
        max_length: 最大序列长度
        lora_r: LoRA秩
        lora_alpha: LoRA缩放因子
        lora_dropout: LoRA dropout
        use_fp16: 是否使用半精度训练
        monitor: 是否监控硬件
        monitor_interval: 监控间隔（秒）
        save_steps: 保存步数
        logging_steps: 日志步数
        warmup_steps: 预热步数
        gradient_accumulation_steps: 梯度累积步数
        seed: 随机种子
    """
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 构建命令
    python_executable = sys.executable
    main_script = os.path.join(project_root, "main.py")
    
    # 1. 生成数据集
    logger.info("第1步：生成微调数据集")
    generate_cmd = [
        python_executable, main_script, "generate_dataset",
        "--num_samples", str(num_samples),
        "--output_file", dataset_path,
        "--seed", str(seed)
    ]
    
    logger.info(f"执行命令: {' '.join(generate_cmd)}")
    result = subprocess.run(generate_cmd, check=True)
    
    if result.returncode != 0:
        logger.error("生成数据集失败")
        return
    
    # 2. 微调模型
    logger.info("第2步：微调模型")
    finetune_cmd = [
        python_executable, main_script, "finetune",
        "--model_path", model_path,
        "--dataset_path", dataset_path,
        "--output_dir", output_dir,
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--max_length", str(max_length),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--save_steps", str(save_steps),
        "--logging_steps", str(logging_steps),
        "--warmup_steps", str(warmup_steps),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--seed", str(seed)
    ]
    
    if use_fp16:
        finetune_cmd.append("--use_fp16")
    
    if monitor:
        finetune_cmd.append("--monitor")
        finetune_cmd.extend(["--monitor_interval", str(monitor_interval)])
    
    logger.info(f"执行命令: {' '.join(finetune_cmd)}")
    result = subprocess.run(finetune_cmd, check=True)
    
    if result.returncode != 0:
        logger.error("微调模型失败")
        return
    
    logger.info("微调流程完成")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行完整的微调流程")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    parser.add_argument("--num_samples", type=int, default=100, help="生成的样本数量")
    parser.add_argument("--dataset_path", type=str, default="data/finetune/instruction_dataset.json", help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="models/finetune", help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_fp16", action="store_true", help="是否使用半精度训练")
    parser.add_argument("--monitor", action="store_true", help="是否监控硬件")
    parser.add_argument("--monitor_interval", type=float, default=1.0, help="监控间隔（秒）")
    parser.add_argument("--save_steps", type=int, default=10, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=1, help="日志步数")
    parser.add_argument("--warmup_steps", type=int, default=0, help="预热步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    logger = setup_logger(
        name="attn_experiment",
        log_dir=config["logging"]["log_dir"],
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info("开始运行微调流程")
    
    # 运行微调流程
    run_finetune_pipeline(
        model_path=args.model_path,
        num_samples=args.num_samples,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_fp16=args.use_fp16,
        monitor=args.monitor,
        monitor_interval=args.monitor_interval,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed
    )
    
    logger.info("微调流程完成")

if __name__ == "__main__":
    main() 