#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微调模型脚本，支持半精度训练和硬件监控
"""

import os
import sys
import json
import time
import torch
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import load_config
from src.utils.logger import setup_logger
from src.utils.hardware_monitor import HardwareMonitor

logger = logging.getLogger("attn_experiment")

class InstructionDataset(Dataset):
    """指令微调数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        """
        初始化数据集
        
        Args:
            data_path: 数据集路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        logger.info(f"加载了{len(self.data)}个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        response = item["response"]
        
        # 构建提示模板 (使用Qwen2.5的格式)
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        
        # 编码
        encodings = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length")
        
        # 创建标签，将非回复部分的标签设为-100（在计算损失时会被忽略）
        input_ids = encodings["input_ids"]
        labels = input_ids.copy()
        
        # 找到助手回复的开始位置
        assistant_start_str = "<|im_start|>assistant\n"
        assistant_start_tokens = self.tokenizer.encode(assistant_start_str, add_special_tokens=False)
        
        # 找到助手回复的开始位置
        for i in range(len(input_ids) - len(assistant_start_tokens)):
            if input_ids[i:i+len(assistant_start_tokens)] == assistant_start_tokens:
                assistant_start_idx = i + len(assistant_start_tokens)
                break
        else:
            # 如果找不到，则将所有标签设为-100
            assistant_start_idx = 0
        
        # 将非助手回复部分的标签设为-100
        for i in range(assistant_start_idx):
            labels[i] = -100
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(encodings["attention_mask"]),
            "labels": torch.tensor(labels)
        }

def finetune_model(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 1024,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_fp16: bool = True,
    monitor_hardware: bool = False,
    monitor_interval: float = 1.0,
    save_steps: int = 10,
    logging_steps: int = 1,
    warmup_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    seed: int = 42
):
    """
    微调模型
    
    Args:
        model_path: 模型路径
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
        monitor_hardware: 是否监控硬件
        monitor_interval: 监控间隔（秒）
        save_steps: 保存步数
        logging_steps: 日志步数
        warmup_steps: 预热步数
        gradient_accumulation_steps: 梯度累积步数
        seed: 随机种子
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 配置LoRA
    logger.info("配置LoRA")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 准备模型
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} ({trainable_params / all_params:.2%})")
    
    # 加载数据集
    logger.info(f"加载数据集: {dataset_path}")
    dataset = InstructionDataset(dataset_path, tokenizer, max_length=max_length)
    
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        fp16=use_fp16,
        report_to="none",
        save_total_limit=3,
        remove_unused_columns=False,
        seed=seed
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 启动硬件监控
    if monitor_hardware:
        logger.info("启动硬件监控")
        hardware_monitor = HardwareMonitor(
            log_dir=os.path.join(output_dir, "hardware_logs"),
            interval=monitor_interval
        )
        hardware_monitor.start()
    
    try:
        # 开始训练
        logger.info("开始训练")
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        # 计算训练时间
        training_time = end_time - start_time
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 保存模型
        logger.info(f"保存模型到: {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 保存训练配置
        with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_path": model_path,
                "dataset_path": dataset_path,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "use_fp16": use_fp16,
                "training_time": training_time,
                "trainable_params": trainable_params,
                "all_params": all_params
            }, f, ensure_ascii=False, indent=2)
    
    finally:
        # 停止硬件监控
        if monitor_hardware:
            logger.info("停止硬件监控")
            hardware_monitor.stop()
            hardware_monitor.save_stats(os.path.join(output_dir, "hardware_stats.json"))
            hardware_monitor.plot_stats(os.path.join(output_dir, "hardware_plots"))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="微调模型")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
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
    
    logger.info("开始微调模型")
    
    # 微调模型
    finetune_model(
        model_path=args.model_path,
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
        monitor_hardware=args.monitor,
        monitor_interval=args.monitor_interval,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed
    )
    
    logger.info("模型微调完成")

if __name__ == "__main__":
    main() 