#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微调Qwen2.5模型脚本
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
import numpy as np
from datasets import load_dataset
import psutil
import GPUtil
import threading
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger
from scripts.model.generate_finetune_data import generate_dataset

# 设置日志记录器
logger = logging.getLogger("attn_experiment")

class ResourceMonitor:
    """
    资源监控器，监控CPU、GPU、内存使用情况
    """
    def __init__(self, interval=1.0, log_dir="logs/monitor"):
        self.interval = interval
        self.log_dir = log_dir
        self.running = False
        self.thread = None
        self.cpu_usages = []
        self.memory_usages = []
        self.gpu_usages = []
        self.gpu_memory_usages = []
        self.timestamps = []
        self.start_time = None
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
    
    def _monitor(self):
        """监控线程的主函数"""
        self.start_time = time.time()
        while self.running:
            timestamp = time.time() - self.start_time
            
            # 获取CPU使用情况
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # 获取内存使用情况
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # 获取GPU使用情况
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 获取第一个GPU
                    gpu_load = gpu.load * 100.0
                    gpu_memory_percent = gpu.memoryUtil * 100.0
                else:
                    gpu_load = 0.0
                    gpu_memory_percent = 0.0
            except Exception as e:
                logger.warning(f"获取GPU信息时出错: {e}")
                gpu_load = 0.0
                gpu_memory_percent = 0.0
            
            # 记录数据
            self.timestamps.append(timestamp)
            self.cpu_usages.append(cpu_percent)
            self.memory_usages.append(memory_percent)
            self.gpu_usages.append(gpu_load)
            self.gpu_memory_usages.append(gpu_memory_percent)
            
            # 每隔一段时间记录一次日志
            if len(self.timestamps) % 10 == 0:
                logger.info(f"资源监控 - CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%, "
                           f"GPU负载: {gpu_load:.1f}%, GPU内存: {gpu_memory_percent:.1f}%")
            
            # 休眠一段时间
            time.sleep(self.interval)
    
    def start(self):
        """开始监控"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor)
            self.thread.daemon = True
            self.thread.start()
            logger.info("资源监控已启动")
    
    def stop(self):
        """停止监控"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)
            logger.info("资源监控已停止")
            
            # 生成监控报告
            self._generate_report()
    
    def _generate_report(self):
        """生成资源使用报告"""
        # 保存监控数据到CSV文件
        data_file = os.path.join(self.log_dir, "resource_usage.csv")
        with open(data_file, "w", encoding="utf-8") as f:
            f.write("timestamp,cpu_percent,memory_percent,gpu_load,gpu_memory_percent\n")
            for i in range(len(self.timestamps)):
                f.write(f"{self.timestamps[i]:.2f},{self.cpu_usages[i]:.2f},{self.memory_usages[i]:.2f},"
                       f"{self.gpu_usages[i]:.2f},{self.gpu_memory_usages[i]:.2f}\n")
        
        # 绘制监控图表
        plt.figure(figsize=(12, 10))
        
        # CPU使用率
        plt.subplot(2, 2, 1)
        plt.plot(self.timestamps, self.cpu_usages)
        plt.title("CPU使用率")
        plt.xlabel("时间(秒)")
        plt.ylabel("使用率(%)")
        plt.grid(True)
        
        # 内存使用率
        plt.subplot(2, 2, 2)
        plt.plot(self.timestamps, self.memory_usages)
        plt.title("内存使用率")
        plt.xlabel("时间(秒)")
        plt.ylabel("使用率(%)")
        plt.grid(True)
        
        # GPU使用率
        plt.subplot(2, 2, 3)
        plt.plot(self.timestamps, self.gpu_usages)
        plt.title("GPU使用率")
        plt.xlabel("时间(秒)")
        plt.ylabel("使用率(%)")
        plt.grid(True)
        
        # GPU内存使用率
        plt.subplot(2, 2, 4)
        plt.plot(self.timestamps, self.gpu_memory_usages)
        plt.title("GPU内存使用率")
        plt.xlabel("时间(秒)")
        plt.ylabel("使用率(%)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "resource_usage.png"))
        plt.close()
        
        # 计算平均值和最大值
        avg_cpu = np.mean(self.cpu_usages)
        max_cpu = np.max(self.cpu_usages)
        avg_memory = np.mean(self.memory_usages)
        max_memory = np.max(self.memory_usages)
        avg_gpu = np.mean(self.gpu_usages)
        max_gpu = np.max(self.gpu_usages)
        avg_gpu_memory = np.mean(self.gpu_memory_usages)
        max_gpu_memory = np.max(self.gpu_memory_usages)
        
        # 保存摘要报告
        summary_file = os.path.join(self.log_dir, "resource_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("资源使用摘要报告\n")
            f.write("=================\n\n")
            f.write(f"监控持续时间: {self.timestamps[-1]:.2f} 秒\n\n")
            f.write("CPU使用率:\n")
            f.write(f"  平均: {avg_cpu:.2f}%\n")
            f.write(f"  最大: {max_cpu:.2f}%\n\n")
            f.write("内存使用率:\n")
            f.write(f"  平均: {avg_memory:.2f}%\n")
            f.write(f"  最大: {max_memory:.2f}%\n\n")
            f.write("GPU使用率:\n")
            f.write(f"  平均: {avg_gpu:.2f}%\n")
            f.write(f"  最大: {max_gpu:.2f}%\n\n")
            f.write("GPU内存使用率:\n")
            f.write(f"  平均: {avg_gpu_memory:.2f}%\n")
            f.write(f"  最大: {max_gpu_memory:.2f}%\n")
        
        logger.info(f"资源监控报告已保存到 {self.log_dir}")
        
        # 记录摘要到日志
        logger.info(f"资源使用摘要 - CPU: 平均 {avg_cpu:.2f}%, 最大 {max_cpu:.2f}%; "
                   f"内存: 平均 {avg_memory:.2f}%, 最大 {max_memory:.2f}%; "
                   f"GPU: 平均 {avg_gpu:.2f}%, 最大 {max_gpu:.2f}%; "
                   f"GPU内存: 平均 {avg_gpu_memory:.2f}%, 最大 {max_gpu_memory:.2f}%")

class QwenFinetuneDataset(Dataset):
    """
    Qwen微调数据集类
    """
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_format = self.get_chat_format()
        
        # 加载数据
        logger.info(f"从 {data_path} 加载数据集")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"加载了 {len(self.data)} 个训练样本")
    
    def get_chat_format(self):
        """获取Qwen2的对话格式"""
        # 确保bos_token不为None
        bos_token = self.tokenizer.bos_token or ""
        eos_token = self.tokenizer.eos_token or ""
        
        chat_format = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "system_token": "<|im_start|>system\n",
            "system_token_end": "<|im_end|>\n",
            "user_token": "<|im_start|>user\n",
            "user_token_end": "<|im_end|>\n",
            "assistant_token": "<|im_start|>assistant\n",
            "assistant_token_end": "<|im_end|>\n",
        }
        return chat_format
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        
        # 构建对话格式
        if input_text:
            prompt = f"{self.chat_format['user_token']}{instruction}\n\n{input_text}{self.chat_format['user_token_end']}"
        else:
            prompt = f"{self.chat_format['user_token']}{instruction}{self.chat_format['user_token_end']}"
        
        completion = f"{self.chat_format['assistant_token']}{output}{self.chat_format['assistant_token_end']}"
        
        # 完整文本
        full_text = self.chat_format["bos_token"] + prompt + completion
        
        # 分词
        tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # 构建标签，仅对assistant部分计算损失
        # 对于输入部分（prompt），将标签设为-100，这样不会计算损失
        prompt_tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_len = prompt_tokenized["input_ids"].shape[1]
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def setup_qlora_training(model, tokenizer, args):
    """
    设置QLoRA训练参数
    """
    # LoRA配置
    peft_config = LoraConfig(
        r=16,                    # LoRA秩
        lora_alpha=32,           # LoRA缩放因子
        lora_dropout=0.05,       # LoRA dropout概率
        bias="none",             # 是否为偏置添加LoRA
        task_type="CAUSAL_LM",   # 任务类型
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # 要微调的模块
    )
    
    # 准备模型
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def train_custom(model, tokenizer, dataset, args, monitor=None):
    """
    使用自定义训练循环进行微调
    """
    logger.info("开始自定义训练循环")
    
    # 数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 学习率调度器
    num_training_steps = args.max_steps
    num_warmup_steps = max(int(num_training_steps * 0.1), 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    # 训练循环
    model.train()
    global_step = 0
    epoch = 0
    total_loss = 0
    
    progress_bar = tqdm(total=args.max_steps, desc=f"训练")
    
    while global_step < args.max_steps:
        epoch += 1
        logger.info(f"开始第 {epoch} 轮训练")
        
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= args.max_steps:
                break
            
            # 将数据移到设备上
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 优化器步进
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 更新进度条
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            
            # 每隔一定步数记录日志
            if global_step % 5 == 0:
                avg_loss = total_loss / 5
                logger.info(f"步骤 {global_step}/{args.max_steps}, 损失: {avg_loss:.4f}")
                total_loss = 0
            
            # 每隔一定步数保存检查点
            if args.output_dir and global_step % 20 == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                logger.info(f"保存检查点到 {checkpoint_dir}")
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
    
    progress_bar.close()
    
    # 保存最终模型
    if args.output_dir:
        final_model_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_model_dir, exist_ok=True)
        
        logger.info(f"保存最终模型到 {final_model_dir}")
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
    
    logger.info("训练完成")
    return model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="微调Qwen2.5模型")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型路径")
    parser.add_argument("--dataset_path", type=str, default="data/finetune/dataset.json", help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="models/finetuned", help="输出目录")
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="fp16", help="训练精度")
    parser.add_argument("--max_steps", type=int, default=50, help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--generate_dataset", action="store_true", help="是否生成示例数据集")
    parser.add_argument("--dataset_size", type=int, default=100, help="生成的数据集大小")
    parser.add_argument("--monitor", action="store_true", help="是否监控硬件使用情况")
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成数据集（如果需要）
    if args.generate_dataset or not os.path.exists(args.dataset_path):
        logger.info(f"生成示例数据集，大小: {args.dataset_size}")
        args.dataset_path = generate_dataset(args.dataset_size, args.dataset_path)
    
    # 启动资源监控（如果需要）
    monitor = None
    if args.monitor:
        monitor_log_dir = os.path.join("logs/monitor", time.strftime("%Y%m%d-%H%M%S"))
        monitor = ResourceMonitor(interval=1.0, log_dir=monitor_log_dir)
        monitor.start()
    
    try:
        # 加载模型和分词器
        logger.info(f"加载模型: {args.model_path}")
        
        # 根据精度设置torch_dtype
        if args.precision == "fp16":
            torch_dtype = torch.float16
        elif args.precision == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # 使用QLoRA进行训练
        logger.info(f"使用 {args.precision} 精度加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )
        
        # 准备模型进行QLoRA微调
        logger.info("准备模型进行QLoRA微调")
        model = prepare_model_for_kbit_training(model)
        model = setup_qlora_training(model, tokenizer, args)
        
        # 创建数据集
        logger.info("创建数据集")
        dataset = QwenFinetuneDataset(args.dataset_path, tokenizer)
        
        # 开始训练
        logger.info(f"开始微调训练，最大步数: {args.max_steps}")
        train_custom(model, tokenizer, dataset, args, monitor)
        
        logger.info("微调完成")
        
    finally:
        # 停止资源监控
        if monitor:
            monitor.stop()

if __name__ == "__main__":
    main() 