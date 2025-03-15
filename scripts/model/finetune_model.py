#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning script for Qwen2.5 model
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
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger
from scripts.model.generate_finetune_data import generate_dataset
from src.utils.flops_profiler import FlopsProfilerWrapper
# 导入注意力机制相关工具
from src.attention.attention_utils import replace_attention_mechanism, get_attention_info, get_attention_config

# 设置日志记录器
logger = logging.getLogger("attn_experiment")

class ResourceMonitor:
    """
    Resource monitor to track CPU, GPU, and memory usage
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
        
        # Support for model performance metrics
        self.latency = []  # Latency (ms)
        self.tokens_per_second = []  # Generation speed (tokens/s)
        self.perplexity = []  # Perplexity
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
    
    def _monitor(self):
        """Main function for the monitoring thread"""
        self.start_time = time.time()
        while self.running:
            timestamp = time.time() - self.start_time
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Get memory usage
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # Get GPU usage
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get the first GPU
                    gpu_load = gpu.load * 100.0
                    gpu_memory_percent = gpu.memoryUtil * 100.0
                else:
                    gpu_load = 0.0
                    gpu_memory_percent = 0.0
            except Exception as e:
                logger.warning(f"Error getting GPU information: {e}")
                gpu_load = 0.0
                gpu_memory_percent = 0.0
            
            # Record data
            self.timestamps.append(timestamp)
            self.cpu_usages.append(cpu_percent)
            self.memory_usages.append(memory_percent)
            self.gpu_usages.append(gpu_load)
            self.gpu_memory_usages.append(gpu_memory_percent)
            
            # Log periodically
            if len(self.timestamps) % 10 == 0:
                logger.info(f"Resource monitoring - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, "
                           f"GPU Load: {gpu_load:.1f}%, GPU Memory: {gpu_memory_percent:.1f}%")
            
            # Sleep for a while
            time.sleep(self.interval)
    
    def add_model_metric(self, metric_name, value):
        """
        Add a model performance metric
        
        Args:
            metric_name: Metric name, e.g., 'latency', 'tokens_per_second', 'perplexity', 'train_loss'
            value: Metric value
        """
        if not self.running:
            logger.warning(f"Monitor is not running, cannot add metric: {metric_name}={value}")
            return
            
        if metric_name == "latency":
            self.latency.append(value)
        elif metric_name == "tokens_per_second":
            self.tokens_per_second.append(value)
        elif metric_name == "perplexity":
            self.perplexity.append(value)
        elif metric_name == "train_loss":
            # Support for training loss
            if not hasattr(self, 'train_loss'):
                self.train_loss = []
            self.train_loss.append(value)
        # Support for FLOPs analyzer performance metrics
        elif metric_name in ["dynamic_flops", "dynamic_macs", "dynamic_params", 
                            "forward_elapsed_time", "flops_per_second"]:
            # Dynamically create lists for each metric type
            if not hasattr(self, metric_name):
                setattr(self, metric_name, [])
            # Get the list for this metric and add the value
            metric_list = getattr(self, metric_name)
            metric_list.append(value)
        else:
            logger.warning(f"Unknown performance metric: {metric_name}")
    
    def start(self):
        """Start monitoring"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)
            logger.info("Resource monitoring stopped")
            
            # Generate monitoring report
            self._generate_report()
    
    def _generate_report(self):
        """Generate resource usage report"""
        # Save monitoring data to CSV file
        data_file = os.path.join(self.log_dir, "resource_usage.csv")
        with open(data_file, "w", encoding="utf-8") as f:
            # Add headers, including model performance metrics
            headers = ["timestamp", "cpu_percent", "memory_percent", "gpu_load", "gpu_memory_percent"]
            
            # If we have model performance metrics, add corresponding columns
            if self.latency:
                headers.append("latency")
            if self.tokens_per_second:
                headers.append("tokens_per_second")
            if self.perplexity:
                headers.append("perplexity")
            if hasattr(self, 'train_loss') and self.train_loss:
                headers.append("train_loss")
            
            # Add FLOPs analyzer performance metrics
            flops_metrics = ["dynamic_flops", "dynamic_macs", "dynamic_params", 
                            "forward_elapsed_time", "flops_per_second"]
            for metric in flops_metrics:
                if hasattr(self, metric) and getattr(self, metric):
                    headers.append(metric)
                
            f.write(",".join(headers) + "\n")
            
            # Write all data
            for i in range(len(self.timestamps)):
                line = f"{self.timestamps[i]:.2f},{self.cpu_usages[i]:.2f},{self.memory_usages[i]:.2f}," \
                      f"{self.gpu_usages[i]:.2f},{self.gpu_memory_usages[i]:.2f}"
                
                # Add model performance metrics (if available)
                if self.latency and i < len(self.latency):
                    line += f",{self.latency[i]:.2f}"
                elif self.latency:
                    line += ","
                    
                if self.tokens_per_second and i < len(self.tokens_per_second):
                    line += f",{self.tokens_per_second[i]:.2f}"
                elif self.tokens_per_second:
                    line += ","
                    
                if self.perplexity and i < len(self.perplexity):
                    line += f",{self.perplexity[i]:.2f}"
                elif self.perplexity:
                    line += ","
                
                if hasattr(self, 'train_loss') and self.train_loss:
                    if i < len(self.train_loss):
                        line += f",{self.train_loss[i]:.4f}"
                    else:
                        line += ","
                
                # Add FLOPs analyzer performance metrics
                for metric in flops_metrics:
                    if hasattr(self, metric) and getattr(self, metric):
                        metric_list = getattr(self, metric)
                        if i < len(metric_list):
                            line += f",{metric_list[i]}"
                        else:
                            line += ","
                
                f.write(line + "\n")
        
        # Plot monitoring charts
        num_plots = 4  # Base number of plots (CPU, memory, GPU, GPU memory)
        
        # Determine the number of additional plots needed
        if self.latency:
            num_plots += 1
        if self.tokens_per_second:
            num_plots += 1
        if self.perplexity:
            num_plots += 1
        
        # Add FLOPs analyzer metrics
        flops_metrics = ["dynamic_flops", "dynamic_macs", "dynamic_params", 
                        "forward_elapsed_time", "flops_per_second"]
        for metric in flops_metrics:
            if hasattr(self, metric) and getattr(self, metric) and len(getattr(self, metric)) > 0:
                num_plots += 1
        
        # Add training loss
        if hasattr(self, 'train_loss') and self.train_loss and len(self.train_loss) > 0:
            num_plots += 1
        
        # 创建训练步骤索引 - 用于统一所有图表的横坐标
        train_steps = None
        if hasattr(self, 'train_loss') and self.train_loss:
            # 如果有训练损失数据，使用其长度作为训练步数的参考
            train_steps = len(self.train_loss)
        
        # 如果需要调整FLOPs分析数据的采样频率
        flops_data_resampled = {}
        for metric in flops_metrics:
            if hasattr(self, metric) and getattr(self, metric) and len(getattr(self, metric)) > 0:
                metric_data = getattr(self, metric)
                if train_steps and len(metric_data) != train_steps:
                    # 如果FLOPs数据点数与训练步数不一致，进行重采样
                    # 方法1：保留前train_steps个点
                    if len(metric_data) > train_steps:
                        flops_data_resampled[metric] = metric_data[:train_steps]
                    # 方法2：如果数据点不足，则重复最后一个值
                    else:
                        resampled = list(metric_data)
                        while len(resampled) < train_steps:
                            resampled.append(resampled[-1])
                        flops_data_resampled[metric] = resampled
        
        # Create appropriately sized figure
        rows = (num_plots + 1) // 2  # Round up
        plt.figure(figsize=(14, 5 * rows))
        
        # CPU usage
        plt.subplot(rows, 2, 1)
        plt.plot(self.timestamps, self.cpu_usages)
        plt.title("CPU Usage")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Usage (%)")
        plt.grid(True)
        
        # Memory usage
        plt.subplot(rows, 2, 2)
        plt.plot(self.timestamps, self.memory_usages)
        plt.title("Memory Usage")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Usage (%)")
        plt.grid(True)
        
        # GPU usage
        plt.subplot(rows, 2, 3)
        plt.plot(self.timestamps, self.gpu_usages)
        plt.title("GPU Usage")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Usage (%)")
        plt.grid(True)
        
        # GPU memory usage
        plt.subplot(rows, 2, 4)
        plt.plot(self.timestamps, self.gpu_memory_usages)
        plt.title("GPU Memory Usage")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Usage (%)")
        plt.grid(True)
        
        # 生成用于步骤图表的X轴
        if train_steps:
            steps_x = np.arange(train_steps)
        
        # Model latency
        plot_idx = 5
        if self.latency:
            plt.subplot(rows, 2, plot_idx)
            if train_steps and len(self.latency) > train_steps:
                # 如果数据点数超过训练步数，只绘制前train_steps个点
                plt.plot(steps_x, self.latency[:train_steps])
            else:
                plt.plot(range(len(self.latency)), self.latency)
            plt.title("Model Latency")
            plt.xlabel("Training Steps")
            plt.ylabel("Latency (ms)")
            plt.grid(True)
            plot_idx += 1
        
        # Model generation speed
        if self.tokens_per_second:
            plt.subplot(rows, 2, plot_idx)
            if train_steps and len(self.tokens_per_second) > train_steps:
                plt.plot(steps_x, self.tokens_per_second[:train_steps])
            else:
                plt.plot(range(len(self.tokens_per_second)), self.tokens_per_second)
            plt.title("Generation Speed")
            plt.xlabel("Training Steps")
            plt.ylabel("Tokens/sec")
            plt.grid(True)
            plot_idx += 1
        
        # Model perplexity
        if self.perplexity:
            plt.subplot(rows, 2, plot_idx)
            if train_steps and len(self.perplexity) > train_steps:
                plt.plot(steps_x, self.perplexity[:train_steps])
            else:
                plt.plot(range(len(self.perplexity)), self.perplexity)
            plt.title("Model Perplexity")
            plt.xlabel("Training Steps")
            plt.ylabel("Perplexity")
            plt.grid(True)
            plot_idx += 1
        
        # Training loss
        if hasattr(self, 'train_loss') and self.train_loss and len(self.train_loss) > 0:
            plt.subplot(rows, 2, plot_idx)
            plt.plot(steps_x, self.train_loss)
            plt.title("Training Loss")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss Value")
            plt.grid(True)
            plot_idx += 1
        
        # FLOPs analyzer metrics
        for metric in flops_metrics:
            if hasattr(self, metric) and getattr(self, metric) and len(getattr(self, metric)) > 0:
                plt.subplot(rows, 2, plot_idx)
                
                # 使用重采样的数据或原始数据
                if metric in flops_data_resampled:
                    metric_data = flops_data_resampled[metric]
                    plt.plot(steps_x, metric_data)
                else:
                    metric_data = getattr(self, metric)
                    if train_steps and len(metric_data) > train_steps:
                        plt.plot(steps_x, metric_data[:train_steps])
                    else:
                        plt.plot(range(len(metric_data)), metric_data)
                
                if metric == "dynamic_flops":
                    plt.title("Dynamic FLOPs")
                    plt.ylabel("FLOPs")
                elif metric == "dynamic_macs":
                    plt.title("Dynamic MACs")
                    plt.ylabel("MACs")
                elif metric == "dynamic_params":
                    plt.title("Dynamic Parameters")
                    plt.ylabel("Parameters")
                elif metric == "forward_elapsed_time":
                    plt.title("Forward Time")
                    plt.ylabel("Time (sec)")
                elif metric == "flops_per_second":
                    plt.title("FLOPs per Second")
                    plt.ylabel("FLOPs/sec")
                else:
                    plt.title(f"{metric}")
                    plt.ylabel("Value")
                
                plt.xlabel("Training Steps")
                plt.grid(True)
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "resource_usage.png"))
        plt.close()
        
        # Calculate averages and maximums
        avg_cpu = np.mean(self.cpu_usages)
        max_cpu = np.max(self.cpu_usages)
        avg_memory = np.mean(self.memory_usages)
        max_memory = np.max(self.memory_usages)
        avg_gpu = np.mean(self.gpu_usages)
        max_gpu = np.max(self.gpu_usages)
        avg_gpu_memory = np.mean(self.gpu_memory_usages)
        max_gpu_memory = np.max(self.gpu_memory_usages)
        
        # Calculate model performance metrics statistics
        avg_latency = np.mean(self.latency) if self.latency else None
        max_latency = np.max(self.latency) if self.latency else None
        avg_tokens_per_second = np.mean(self.tokens_per_second) if self.tokens_per_second else None
        min_tokens_per_second = np.min(self.tokens_per_second) if self.tokens_per_second else None
        avg_perplexity = np.mean(self.perplexity) if self.perplexity else None
        min_perplexity = np.min(self.perplexity) if self.perplexity else None
        
        # Calculate FLOPs analyzer performance metrics statistics
        flops_metrics_stats = {}
        flops_metrics = ["dynamic_flops", "dynamic_macs", "dynamic_params", 
                         "forward_elapsed_time", "flops_per_second"]
        for metric in flops_metrics:
            if hasattr(self, metric) and getattr(self, metric):
                metric_list = getattr(self, metric)
                if metric_list:
                    flops_metrics_stats[metric] = {
                        "avg": np.mean(metric_list),
                        "max": np.max(metric_list),
                        "min": np.min(metric_list)
                    }
        
        # Save summary report
        summary_file = os.path.join(self.log_dir, "resource_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Resource Usage Summary Report\n")
            f.write("============================\n\n")
            f.write(f"Monitoring Duration: {self.timestamps[-1]:.2f} seconds\n\n")
            f.write("CPU Usage:\n")
            f.write(f"  Average: {avg_cpu:.2f}%\n")
            f.write(f"  Maximum: {max_cpu:.2f}%\n\n")
            f.write("Memory Usage:\n")
            f.write(f"  Average: {avg_memory:.2f}%\n")
            f.write(f"  Maximum: {max_memory:.2f}%\n\n")
            f.write("GPU Usage:\n")
            f.write(f"  Average: {avg_gpu:.2f}%\n")
            f.write(f"  Maximum: {max_gpu:.2f}%\n\n")
            f.write("GPU Memory Usage:\n")
            f.write(f"  Average: {avg_gpu_memory:.2f}%\n")
            f.write(f"  Maximum: {max_gpu_memory:.2f}%\n\n")
            
            # Add model performance metrics
            if avg_latency is not None:
                f.write("Model Latency:\n")
                f.write(f"  Average: {avg_latency:.2f} ms\n")
                f.write(f"  Maximum: {max_latency:.2f} ms\n\n")
            
            if avg_tokens_per_second is not None:
                f.write("Generation Speed:\n")
                f.write(f"  Average: {avg_tokens_per_second:.2f} tokens/sec\n")
                f.write(f"  Minimum: {min_tokens_per_second:.2f} tokens/sec\n\n")
            
            if avg_perplexity is not None:
                f.write("Model Perplexity:\n")
                f.write(f"  Average: {avg_perplexity:.2f}\n")
                f.write(f"  Minimum: {min_perplexity:.2f}\n\n")
            
            # Add FLOPs analyzer performance metrics
            for metric, stats in flops_metrics_stats.items():
                if metric == "dynamic_flops":
                    f.write("Dynamic FLOPs:\n")
                elif metric == "dynamic_macs":
                    f.write("Dynamic MACs:\n")
                elif metric == "dynamic_params":
                    f.write("Dynamic Parameters:\n")
                elif metric == "forward_elapsed_time":
                    f.write("Forward Time:\n")
                elif metric == "flops_per_second":
                    f.write("FLOPs per Second:\n")
                else:
                    f.write(f"{metric}:\n")
                
                f.write(f"  Average: {stats['avg']:.2f}\n")
                f.write(f"  Maximum: {stats['max']:.2f}\n")
                f.write(f"  Minimum: {stats['min']:.2f}\n\n")
            
            # Add training loss metrics
            if hasattr(self, 'train_loss') and self.train_loss:
                avg_train_loss = np.mean(self.train_loss)
                min_train_loss = np.min(self.train_loss)
                max_train_loss = np.max(self.train_loss)
                f.write("Training Loss:\n")
                f.write(f"  Average: {avg_train_loss:.4f}\n")
                f.write(f"  Minimum: {min_train_loss:.4f}\n")
                f.write(f"  Maximum: {max_train_loss:.4f}\n\n")
        
        logger.info(f"Monitoring report generated to {self.log_dir}")
        
        return data_file

class QwenFinetuneDataset(Dataset):
    """
    Dataset class for Qwen fine-tuning
    """
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_format = self.get_chat_format()
        
        # Load data
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def get_chat_format(self):
        """Get Qwen2 chat format"""
        # Ensure bos_token is not None
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
        
        # Build chat format
        if input_text:
            prompt = f"{self.chat_format['user_token']}{instruction}\n\n{input_text}{self.chat_format['user_token_end']}"
        else:
            prompt = f"{self.chat_format['user_token']}{instruction}{self.chat_format['user_token_end']}"
        
        completion = f"{self.chat_format['assistant_token']}{output}{self.chat_format['assistant_token_end']}"
        
        # Full text
        full_text = self.chat_format["bos_token"] + prompt + completion
        
        # Tokenize
        tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # Build labels, only compute loss for assistant part
        # For input part (prompt), set labels to -100, which won't compute loss
        prompt_tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_len = prompt_tokenized["input_ids"].shape[1]
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class CustomDataCollator:
    """
    Custom data collator to resolve warning about creating tensor from list of numpy arrays
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = True
        self.return_tensors = "pt"
    
    def __call__(self, features):
        # Organize input IDs and attention masks
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        
        # Get labels (if any)
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
        else:
            labels = None
        
        # Calculate max length
        max_length = max(len(ids) for ids in input_ids)
        
        # Align lengths (padding)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = [] if labels else None
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        for i, ids in enumerate(input_ids):
            padding_length = max_length - len(ids)
            
            # Pad input IDs
            if padding_length > 0:
                # Check if ids is already a tensor
                if isinstance(ids, torch.Tensor):
                    ids_tensor = ids.clone().detach()
                else:
                    ids_tensor = torch.tensor(ids, dtype=torch.long)
                
                # Create padding tensor
                padding = torch.ones(padding_length, dtype=torch.long) * pad_token_id
                padded_ids = torch.cat([ids_tensor, padding])
            else:
                # If no padding needed, just use clone
                if isinstance(ids, torch.Tensor):
                    padded_ids = ids.clone().detach()
                else:
                    padded_ids = torch.tensor(ids, dtype=torch.long)
            
            padded_input_ids.append(padded_ids)
            
            # Pad attention masks
            if padding_length > 0:
                # Check if attention_mask[i] is already a tensor
                if isinstance(attention_mask[i], torch.Tensor):
                    mask_tensor = attention_mask[i].clone().detach()
                else:
                    mask_tensor = torch.tensor(attention_mask[i], dtype=torch.long)
                
                # Create padding tensor
                padding = torch.zeros(padding_length, dtype=torch.long)
                padded_mask = torch.cat([mask_tensor, padding])
            else:
                # If no padding needed, just use clone
                if isinstance(attention_mask[i], torch.Tensor):
                    padded_mask = attention_mask[i].clone().detach()
                else:
                    padded_mask = torch.tensor(attention_mask[i], dtype=torch.long)
            
            padded_attention_mask.append(padded_mask)
            
            # Pad labels (if any)
            if labels:
                label = labels[i]
                if padding_length > 0:
                    # Check if label is already a tensor
                    if isinstance(label, torch.Tensor):
                        label_tensor = label.clone().detach()
                    else:
                        label_tensor = torch.tensor(label, dtype=torch.long)
                    
                    # Create padding tensor, using -100
                    padding = torch.ones(padding_length, dtype=torch.long) * -100
                    padded_label = torch.cat([label_tensor, padding])
                else:
                    # If no padding needed, just use clone
                    if isinstance(label, torch.Tensor):
                        padded_label = label.clone().detach()
                    else:
                        padded_label = torch.tensor(label, dtype=torch.long)
                
                padded_labels.append(padded_label)
        
        # Convert to batch tensors
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
        }
        
        if padded_labels:
            batch["labels"] = torch.stack(padded_labels)
        
        return batch

def setup_lora_training(model, tokenizer, args):
    """
    Set up LoRA training parameters
    """
    # LoRA configuration
    peft_config = LoraConfig(
        r=16,                    # LoRA rank
        lora_alpha=32,           # LoRA scaling factor
        lora_dropout=0.05,       # LoRA dropout probability
        bias="none",             # Whether to add LoRA to bias
        task_type="CAUSAL_LM",   # Task type
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # Modules to fine-tune
    )
    
    # Prepare model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def train_custom(model, tokenizer, dataset, args, monitor=None, flops_profiler=None):
    """Custom training function"""
    device = model.device
    train_batch_size = args.batch_size
    
    # 初始化用于统计的变量
    total_training_time = 0
    total_tokens_processed = 0
    
    # Use our custom data collator
    data_collator = CustomDataCollator(tokenizer)
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=train_batch_size, 
        collate_fn=data_collator, 
        shuffle=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    total_steps = min(args.max_steps, len(train_dataloader))
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    total_loss = 0
    steps = 0
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    # 记录训练开始时间
    train_start_time = time.time()
    
    # Check if FLOPs profiler should be enabled
    do_profile = flops_profiler is not None
    
    # Initialize a counter for limiting profiling frequency
    profile_counter = 0
    # Only profile every N steps to reduce overhead (e.g. every 5 steps)
    profile_frequency = args.profile_frequency if hasattr(args, 'profile_frequency') else 5
    
    # Only log the first profiling result in detail, subsequent ones will be summarized
    first_profile = True
    
    # Training loop
    for batch in train_dataloader:
        if steps >= total_steps:
            break
        
        # Move data to device
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels" or v is not None}
        
        # Start profiling if enabled and it's time to profile
        should_profile = do_profile and (profile_counter % profile_frequency == 0)
        
        if should_profile:
            logger.info(f"步骤 {steps}: 开始FLOPs分析")
            # Use print_results=first_profile to log detailed info only for the first time
            flops_profiler.start_profiling(model)
        
        # 记录开始时间，用于计算延迟和生成速度
        start_time = time.time()
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # 计算每个批次的延迟(ms)
        latency = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 计算困惑度（从损失值）
        # perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
        
        # 计算生成速度（tokens/秒）
        # 假设每个批次中的tokens数量为batch中labels的非-100值的数量
        num_tokens = (batch["labels"] != -100).sum().item()
        tokens_per_second = num_tokens / (time.time() - start_time)
        
        # Stop profiling and collect results if enabled
        if should_profile:
            try:
                profile_results = flops_profiler.stop_profiling(print_results=first_profile)
                if profile_results and monitor:
                    # Add metrics to monitor
                    monitor.add_model_metric("dynamic_flops", profile_results['numeric']['flops'])
                    monitor.add_model_metric("dynamic_macs", profile_results['numeric']['macs'])
                    monitor.add_model_metric("dynamic_params", profile_results['numeric']['params'])
                    monitor.add_model_metric("forward_elapsed_time", profile_results['numeric']['forward_elapsed_time'])
                    monitor.add_model_metric("flops_per_second", profile_results['numeric']['flops_per_second'])
                    
                    # Output basic summary if not first time
                    if not first_profile:
                        logger.info(f"步骤 {steps} - FLOPs: {profile_results['readable']['flops']}, "
                                    f"前向传播时间: {profile_results['readable']['forward_elapsed_time']}")
                
                # After first profiling, switch to summary mode
                first_profile = False
                
            except Exception as e:
                logger.warning(f"步骤 {steps} FLOPs分析出错: {str(e)}")
        
        # Update profile counter
        profile_counter += 1
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer and scheduler steps
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update progress
        total_loss += loss.item()
        steps += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": total_loss / steps})
        
        # Add metrics to monitor
        if monitor:
            # 确保每个步骤都记录所有性能指标
            monitor.add_model_metric("train_loss", loss.item())
            monitor.add_model_metric("perplexity", perplexity)
            monitor.add_model_metric("latency", latency)
            monitor.add_model_metric("tokens_per_second", tokens_per_second)
            
            # 每个步骤都输出日志信息，而不是每10步一次
            logger.info(f"步骤 {steps} - 损失: {loss.item():.4f}, 困惑度: {perplexity:.4f}, "
                        f"延迟: {latency:.2f}ms, 生成速度: {tokens_per_second:.2f} tokens/s")
        
        # 累加处理的tokens总数
        total_tokens_processed += num_tokens
        
        # If we've reached max steps, break out of the loop
        if steps >= total_steps:
            break
    
    # Close progress bar
    progress_bar.close()
    
    # 计算总训练时间
    total_training_time = time.time() - train_start_time
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    
    # Calculate average loss
    avg_loss = total_loss / steps
    logger.info(f"Training completed, average loss: {avg_loss:.4f}")
    
    # Save trained model
    logger.info(f"Saving model to {args.output_dir}")
    
    # Save full model (excluding base model weights)
    model.save_pretrained(args.output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training parameters
    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "precision": args.precision,
            "avg_loss": avg_loss
        }, f, indent=2, ensure_ascii=False)
    
    # 保存训练结果和统计数据
    with open(os.path.join(args.output_dir, "training_stats.json"), "w") as f:
        json.dump({
            "train_loss": avg_loss,
            "training_time": total_training_time,
            "train_tokens_per_second": round(total_tokens_processed / total_training_time, 2)
        }, f, indent=2, ensure_ascii=False)
    
    # 保存注意力机制信息
    if args.attention != "standard":
        attention_info = {
            "attention_type": args.attention
        }
        if args.attention == "sparse":
            attention_info["sparsity"] = args.sparsity
        elif args.attention == "linear":
            attention_info["kernel_function"] = args.kernel_function
        elif args.attention == "reformer":
            attention_info["num_hashes"] = args.num_hashes
        elif args.attention == "linformer":
            attention_info["k_ratio"] = args.k_ratio
        elif args.attention == "longformer":
            attention_info["window_size"] = args.window_size
            attention_info["global_tokens_ratio"] = args.global_tokens_ratio
        
        with open(os.path.join(args.output_dir, "attention_config.json"), "w") as f:
            json.dump(attention_info, f, indent=2)
        
        logger.info(f"保存注意力机制配置到: {os.path.join(args.output_dir, 'attention_config.json')}")
    
    # 保存最终模型
    logger.info(f"保存最终模型到: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("微调完成！")
    
    return model, avg_loss

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model path")
    parser.add_argument("--dataset_path", type=str, default="data/finetune/dataset.json", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="models/finetuned", help="Output directory")
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="fp16", help="Training precision")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--generate_dataset", action="store_true", help="Whether to generate example dataset")
    parser.add_argument("--dataset_size", type=int, default=100, help="Size of generated dataset")
    parser.add_argument("--monitor", action="store_true", help="Whether to monitor hardware usage")
    parser.add_argument("--flops_profiler", action="store_true", help="Whether to use DeepSpeed's Flops Profiler for FLOPs analysis")
    parser.add_argument("--profile_frequency", type=int, default=5, help="Frequency of FLOPs profiling (every N steps)")
    # 添加注意力机制相关参数
    parser.add_argument("--attention", type=str, choices=["standard", "sparse", "linear", "reformer", "linformer", "longformer", "realformer", "mla"], default="standard", help="Attention mechanism type")
    parser.add_argument("--sparsity", type=float, default=0.8, help="Sparsity for sparse attention")
    parser.add_argument("--kernel_function", type=str, default="elu", help="Kernel function for linear attention")
    parser.add_argument("--num_hashes", type=int, default=4, help="Number of hashes for Reformer attention")
    parser.add_argument("--k_ratio", type=float, default=0.25, help="K ratio for Linformer attention")
    parser.add_argument("--window_size", type=int, default=128, help="Window size for Longformer attention")
    parser.add_argument("--global_tokens_ratio", type=float, default=0.1, help="Global tokens ratio for Longformer attention")
    parser.add_argument("--last_layer_only", action="store_true", help="Whether to replace only the last attention layer")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset (if needed)
    if args.generate_dataset or not os.path.exists(args.dataset_path):
        logger.info(f"Generating example dataset, size: {args.dataset_size}")
        args.dataset_path = generate_dataset(args.dataset_size, args.dataset_path)
    
    # Start resource monitoring (if needed)
    monitor = None
    if args.monitor:
        # 修改监控日志目录名称，添加注意力机制类型
        monitor_log_dir = os.path.join("logs/monitor", 
                                       f"{time.strftime('%Y%m%d-%H%M%S')}_{args.attention}")
        monitor = ResourceMonitor(interval=1.0, log_dir=monitor_log_dir)
        monitor.start()
    
    # Initialize FLOPs profiler (if needed)
    flops_profiler = None
    if args.flops_profiler:
        logger.info("Initializing FLOPs profiler")
        flops_profiler = FlopsProfilerWrapper(hardware_monitor=monitor)
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading model: {args.model_path}")
        
        # Set torch_dtype based on precision
        if args.precision == "fp16":
            torch_dtype = torch.float16
        elif args.precision == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Use standard LoRA for training
        logger.info(f"Using {args.precision} precision to load model")
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
        
        # 先应用LoRA准备
        logger.info("Preparing model for LoRA fine-tuning")
        model = setup_lora_training(model, tokenizer, args)
        
        # 在LoRA之后替换注意力机制（如果指定）
        if args.attention != "standard":
            if args.last_layer_only:
                logger.info(f"Replacing only the last layer attention with {args.attention} attention")
            else:
                logger.info(f"Replacing all standard attention layers with {args.attention} attention")
                
            attention_kwargs = {}
            if args.attention == "sparse":
                attention_kwargs["sparsity"] = args.sparsity
            elif args.attention == "linear":
                attention_kwargs["kernel_function"] = args.kernel_function
            elif args.attention == "reformer":
                attention_kwargs["num_hashes"] = args.num_hashes
            elif args.attention == "linformer":
                attention_kwargs["k_ratio"] = args.k_ratio
            elif args.attention == "longformer":
                attention_kwargs["window_size"] = args.window_size
                attention_kwargs["global_tokens_ratio"] = args.global_tokens_ratio
            
            # 添加last_layer_only参数
            attention_kwargs["last_layer_only"] = args.last_layer_only
            
            # 替换模型中的注意力层
            model = replace_attention_mechanism(model, args.attention, **attention_kwargs)
            
            # 获取注意力机制信息并记录
            attention_config = get_attention_config(args.attention, **attention_kwargs)
            logger.info(f"Attention mechanism config: {attention_config}")
            
            # 验证注意力机制是否真的被替换了
            replaced_count = 0
            attention_modules = []
            last_layer_replaced = False
            last_layer_name = None
            
            logger.info("开始验证注意力机制替换...")
            # 首先记录所有可能的注意力模块
            for name, module in model.named_modules():
                if 'atten' in name.lower() or 'sdpa' in name.lower() or getattr(module, '_attn_type', None) == args.attention:
                    attention_modules.append((name, type(module).__name__))
                    if hasattr(module, '_attn_type') and module._attn_type == args.attention:
                        replaced_count += 1
                        logger.info(f"✓ 确认替换的注意力模块: {name}, 类型: {type(module).__name__}")
                        # 记录最后一个被替换的模块
                        last_layer_name = name
                        last_layer_replaced = True
                        
                        # 打印模块的稀疏度设置
                        if args.attention == "sparse" and hasattr(module, "_sparsity"):
                            logger.info(f"  稀疏度设置: {module._sparsity} (期望值: {args.sparsity})")
            
            # 如果只替换最后一层，显示明确的信息
            if last_layer_replaced and replaced_count == 1:
                logger.info(f"已成功确认最后一层注意力机制({last_layer_name})被替换为{args.attention}注意力")
            else:
                logger.info(f"模型中找到 {len(attention_modules)} 个可能的注意力相关模块，其中 {replaced_count} 个已确认替换")
            
            if replaced_count == 0:
                logger.warning("警告: 没有找到任何带有_attn_type标记的模块！")
                
                # 打印找到的所有注意力相关模块的信息
                logger.info("可能的注意力模块列表:")
                for name, type_name in attention_modules:
                    logger.info(f"- {name}: {type_name}")
                
                # 更进一步的检查：查找是否有函数被替换
                method_replaced_count = 0
                for name, module in model.named_modules():
                    # 检查compute_attention方法
                    if hasattr(module, "compute_attention"):
                        compute_attn_func = module.compute_attention
                        function_source = str(compute_attn_func)
                        if args.attention.lower() in function_source.lower():
                            logger.info(f"✓ 已找到注意力函数替换: {name}.compute_attention")
                            method_replaced_count += 1
                            last_layer_name = name
                    
                    # 检查forward方法
                    if hasattr(module, "forward"):
                        forward_func = module.forward
                        function_source = str(forward_func)
                        if args.attention.lower() in function_source.lower() or 'sparse' in function_source.lower():
                            logger.info(f"✓ 已找到forward函数替换: {name}.forward")
                            method_replaced_count += 1
                            last_layer_name = name
                
                if method_replaced_count > 0:
                    if method_replaced_count == 1:
                        logger.info(f"通过方法替换确认最后一层注意力机制({last_layer_name})被替换")
                    else:
                        logger.info(f"找到 {method_replaced_count} 个替换的方法，继续进行微调")
                    replaced_count = method_replaced_count
                
                # 如果仍未找到替换，尝试通过行为验证
                if replaced_count == 0:
                    logger.warning("尝试通过行为验证注意力机制替换...")
                    try:
                        # 准备测试输入
                        device = next(model.parameters()).device
                        batch_size, seq_len = 1, 16
                        
                        # 构建一个随机输入序列用于测试注意力
                        if hasattr(model.config, "vocab_size"):
                            vocab_size = model.config.vocab_size
                            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                            
                            logger.info(f"使用随机输入测试模型行为...")
                            # 使用输出注意力权重的模式运行前向传递
                            with torch.no_grad():
                                outputs = model(input_ids=input_ids, output_attentions=True)
                                
                                # 分析输出的注意力权重
                                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                                    attentions = outputs.attentions
                                    if isinstance(attentions, tuple) and len(attentions) > 0:
                                        # 获取最后一层的注意力权重（通常是最后一个元素）
                                        attn_weights = attentions[-1]
                                        
                                        # 分析稀疏模式
                                        if args.attention == "sparse":
                                            # 对于稀疏注意力，我们期望大多数值接近0
                                            nonzero_ratio = (attn_weights > 1e-6).float().mean().item()
                                            logger.info(f"最后一层注意力权重中非零元素比例: {nonzero_ratio:.4f}")
                                            
                                            # 稀疏度验证
                                            expected_nonzero = 1.0 - args.sparsity
                                            tolerance = 0.2  # 允许20%的误差
                                            
                                            if abs(nonzero_ratio - expected_nonzero) < tolerance:
                                                logger.info(f"✓ 验证成功! 最后一层非零比例({nonzero_ratio:.4f})接近预期({expected_nonzero:.4f})")
                                                replaced_count = 1
                                            else:
                                                logger.warning(f"× 验证失败! 非零比例({nonzero_ratio:.4f})与预期({expected_nonzero:.4f})相差较大")
                        else:
                            logger.warning("模型配置中没有vocab_size，无法生成随机输入")
                            
                    except Exception as e:
                        logger.warning(f"行为验证失败: {str(e)}")
            
            # 最终决定是否继续微调
            if replaced_count == 0:
                logger.warning("⚠️ 警告! 无法确认注意力机制是否成功替换!")
                logger.warning("微调过程中可能不会使用您指定的注意力机制。是否要继续?")
                
                # 为了不阻塞进程，这里我们仍然继续执行，但日志中会有明确警告
                logger.warning("继续进行微调，但请注意您可能没有实际使用所选的注意力机制...")
            else:
                if replaced_count == 1:
                    logger.info(f"✅ 确认成功! 已替换最后一层注意力模块/函数")
                else:
                    logger.info(f"✅ 确认成功! 已找到 {replaced_count} 个替换的注意力模块/函数")
        
        # Create dataset
        logger.info("Creating dataset")
        dataset = QwenFinetuneDataset(args.dataset_path, tokenizer)
        
        # Start training
        logger.info(f"Starting fine-tuning, max steps: {args.max_steps}")
        train_custom(model, tokenizer, dataset, args, monitor, flops_profiler)
        
        logger.info("Fine-tuning completed")
        
    finally:
        # Stop resource monitoring
        if monitor:
            monitor.stop()

if __name__ == "__main__":
    main() 