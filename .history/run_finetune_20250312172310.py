#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速运行微调流程的脚本
"""

import os
import sys
import subprocess

def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建命令
    python_executable = sys.executable
    main_script = os.path.join(current_dir, "main.py")
    
    # 运行微调流程
    cmd = [
        python_executable, main_script, "finetune_pipeline",
        "--model_path", "Qwen/Qwen2.5-3B-Instruct",
        "--num_samples", "100",
        "--dataset_path", "data/finetune/instruction_dataset.json",
        "--output_dir", "models/finetune",
        "--num_epochs", "3",
        "--batch_size", "4",
        "--learning_rate", "2e-5",
        "--max_length", "1024",
        "--lora_r", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--save_steps", "10",
        "--logging_steps", "1",
        "--use_fp16",
        "--monitor",
        "--monitor_interval", "1.0"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("微调流程完成")

if __name__ == "__main__":
    main() 