#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试微调前后的模型效果差异
"""

import os
import json
import random
import argparse
import logging
import torch
import sys
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger

# 设置日志记录器
logger = logging.getLogger("attn_experiment")

def format_chat_prompt(tokenizer, instruction, input_text=""):
    """
    格式化对话提示
    """
    if input_text:
        prompt = f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    
    return prompt

def load_model(model_path, device="cuda", precision="fp16"):
    """
    加载原始模型
    """
    logger.info(f"加载模型: {model_path}")
    
    # 设置精度
    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    return model, tokenizer

def load_finetuned_model(base_model_path, adapter_path, device="cuda", precision="fp16", merge_weights=False):
    """
    加载微调后的模型
    
    参数:
        base_model_path: 基础模型路径
        adapter_path: 适配器路径
        device: 设备
        precision: 精度
        merge_weights: 是否合并权重
    """
    logger.info(f"加载基础模型: {base_model_path}")
    logger.info(f"加载微调适配器: {adapter_path}")
    
    # 设置精度
    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    # 加载基础模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        torch_dtype=torch_dtype
    )
    
    # 合并LoRA权重以加快推理速度
    if merge_weights:
        logger.info("合并LoRA权重到基础模型...")
        model = model.merge_and_unload()
        logger.info("权重合并完成，应该可以获得更快的推理速度")
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """
    生成模型回答
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 开始计时
    start_time = time.time()
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # 结束计时
    end_time = time.time()
    generation_time = end_time - start_time
    
    # 解码回答
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text.strip(), generation_time

def load_dataset(dataset_path):
    """
    加载微调数据集
    """
    logger.info(f"加载数据集: {dataset_path}")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return []

def sample_from_dataset(dataset, num_samples=5):
    """
    从数据集中随机抽取样本
    """
    if len(dataset) <= num_samples:
        return dataset
    
    return random.sample(dataset, num_samples)

def compare_responses(original_model, finetuned_model, tokenizer, samples):
    """
    比较原始模型和微调模型的回答
    """
    results = []
    
    # 统计生成时间
    total_original_time = 0
    total_finetuned_time = 0
    
    for idx, sample in enumerate(samples):
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        expected_output = sample["output"]
        
        # 格式化提示
        prompt = format_chat_prompt(tokenizer, instruction, input_text)
        
        logger.info(f"测试样本 {idx+1}/{len(samples)}: {instruction}")
        
        # 使用原始模型生成回答
        logger.info("原始模型生成中...")
        original_response, original_time = generate_response(original_model, tokenizer, prompt)
        logger.info(f"原始模型生成耗时: {original_time:.2f}秒")
        total_original_time += original_time
        
        # 使用微调后的模型生成回答
        logger.info("微调模型生成中...")
        finetuned_response, finetuned_time = generate_response(finetuned_model, tokenizer, prompt)
        logger.info(f"微调模型生成耗时: {finetuned_time:.2f}秒")
        total_finetuned_time += finetuned_time
        
        # 收集结果
        result = {
            "instruction": instruction,
            "input": input_text,
            "expected_output": expected_output,
            "original_response": original_response,
            "finetuned_response": finetuned_response,
            "original_time": original_time,
            "finetuned_time": finetuned_time
        }
        
        results.append(result)
    
    # 计算平均生成时间
    avg_original_time = total_original_time / len(samples)
    avg_finetuned_time = total_finetuned_time / len(samples)
    logger.info(f"原始模型平均生成时间: {avg_original_time:.2f}秒")
    logger.info(f"微调模型平均生成时间: {avg_finetuned_time:.2f}秒")
    logger.info(f"微调模型与原始模型的时间比: {avg_finetuned_time/avg_original_time:.2f}x")
    
    return results, avg_original_time, avg_finetuned_time

def print_comparison(results, avg_original_time, avg_finetuned_time, output_file=None):
    """
    打印比较结果
    """
    output_lines = []
    
    # 添加时间统计信息
    output_lines.append(f"性能比较:")
    output_lines.append(f"原始模型平均生成时间: {avg_original_time:.2f}秒")
    output_lines.append(f"微调模型平均生成时间: {avg_finetuned_time:.2f}秒")
    output_lines.append(f"微调模型与原始模型的时间比: {avg_finetuned_time/avg_original_time:.2f}x")
    output_lines.append("\n" + "="*80)
    
    for idx, result in enumerate(results):
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"样本 {idx+1}:")
        output_lines.append(f"{'='*80}")
        
        output_lines.append(f"\n【问题】: {result['instruction']}")
        if result['input']:
            output_lines.append(f"【输入】: {result['input']}")
        
        output_lines.append(f"\n【原始模型回答】(耗时: {result['original_time']:.2f}秒):\n{result['original_response']}")
        
        output_lines.append(f"\n【微调模型回答】(耗时: {result['finetuned_time']:.2f}秒):\n{result['finetuned_response']}")
        
        output_lines.append(f"\n【数据集中的参考答案】:\n{result['expected_output']}")
        
        output_lines.append("\n" + "-"*80)
    
    output_text = "\n".join(output_lines)
    print(output_text)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        logger.info(f"比较结果已保存到 {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试微调前后的模型效果差异")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="基础模型路径")
    parser.add_argument("--finetuned_model", type=str, default="models/finetuned/final", help="微调模型路径")
    parser.add_argument("--dataset_path", type=str, default="data/finetune/dataset.json", help="数据集路径")
    parser.add_argument("--num_samples", type=int, default=3, help="测试样本数量")
    parser.add_argument("--output_file", type=str, default="test_results.txt", help="输出结果文件")
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="fp16", help="模型精度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--merge_weights", action="store_true", help="是否合并LoRA权重以加快推理速度")
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(42)
    
    # 加载数据集
    dataset = load_dataset(args.dataset_path)
    if not dataset:
        logger.error("数据集为空，退出测试")
        return
    
    # 随机抽取样本
    samples = sample_from_dataset(dataset, args.num_samples)
    
    # 加载原始模型
    original_model, tokenizer = load_model(args.base_model, args.device, args.precision)
    
    # 加载微调后的模型
    finetuned_model, _ = load_finetuned_model(
        args.base_model, 
        args.finetuned_model, 
        args.device, 
        args.precision,
        args.merge_weights
    )
    
    # 优化模型以加快推理
    logger.info("设置模型为评估模式...")
    original_model.eval()
    finetuned_model.eval()
    
    # 预热模型
    logger.info("预热模型...")
    warm_prompt = format_chat_prompt(tokenizer, "你好")
    with torch.no_grad():
        _ = generate_response(original_model, tokenizer, warm_prompt, max_new_tokens=10)
        _ = generate_response(finetuned_model, tokenizer, warm_prompt, max_new_tokens=10)
    
    # 比较回答
    logger.info("开始生成并比较回答...")
    results, avg_original_time, avg_finetuned_time = compare_responses(original_model, finetuned_model, tokenizer, samples)
    
    # 打印比较结果
    print_comparison(results, avg_original_time, avg_finetuned_time, args.output_file)
    
    logger.info("测试完成")

if __name__ == "__main__":
    # 设置日志
    setup_logger(
        name="attn_experiment",
        log_dir="logs",
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    main() 