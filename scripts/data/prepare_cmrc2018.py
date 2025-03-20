#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将CMRC2018数据集转换为适合微调的格式
"""

import os
import json
import random
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import sys
import math

# 添加项目根目录到系统路径
# 修正路径: 当脚本在scripts/data下时，需要向上三级才能到达项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger

# 导入tokenizer
from transformers import AutoTokenizer

# 设置日志记录器
logger = logging.getLogger("attn_experiment")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="准备CMRC2018数据集用于微调")
    parser.add_argument("--input_dir", type=str, default="raw/cmrc2018/squad-style-data", help="CMRC2018数据集目录")
    parser.add_argument("--output_dir", type=str, default="data/finetune", help="输出目录")
    parser.add_argument("--train_file", type=str, default="cmrc2018_train.json", help="训练集文件名")
    parser.add_argument("--dev_file", type=str, default="cmrc2018_dev.json", help="开发集文件名")
    parser.add_argument("--test_file", type=str, default="cmrc2018_dev.json", help="测试集文件名（默认使用dev作为测试集）")
    parser.add_argument("--max_samples", type=int, default=-1, help="每个集合中的最大样本数，-1表示使用所有样本")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--include_position", action="store_true", help="是否在助手回复中包含答案位置信息")
    parser.add_argument("--tokenizer", type=str, default="bert-base-chinese", help="使用的tokenizer")
    parser.add_argument("--use_char_position", action="store_true", help="是否使用字符级位置而不是token级位置")
    parser.add_argument("--split_valid", action="store_true", help="是否从训练集中分割出验证集")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="验证集占训练集的比例")
    return parser.parse_args()

def load_squad_format_data(file_path):
    """加载SQuAD格式的数据集"""
    logger.info(f"正在加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    total_questions = 0
    
    for article in data['data']:
        title = article.get('title', '')
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answers = qa['answers']
                answer_text = answers[0]['text']  # 取第一个答案
                answer_start = answers[0]['answer_start']
                
                sample = {
                    'title': title,
                    'context': context,
                    'question': question,
                    'answer': answer_text,
                    'answer_start': answer_start,
                    'id': qa['id']
                }
                samples.append(sample)
                total_questions += 1
    
    logger.info(f"加载了 {len(samples)} 个问答对")
    return samples

def get_token_positions(context, answer, answer_start, tokenizer):
    """计算答案在context中的token级别起始和结束位置
    
    Args:
        context: 原文内容
        answer: 答案文本
        answer_start: 答案在原文中的字符级起始位置
        tokenizer: 使用的tokenizer
        
    Returns:
        token_start, token_end: token级别的起始和结束位置
    """
    # 对context进行tokenize，获取每个token对应的原始文本范围
    encoding = tokenizer(context, return_offsets_mapping=True)
    offsets = encoding.offset_mapping  # 每个token对应的字符级范围
    
    # 计算answer的结束位置（字符级）
    answer_end = answer_start + len(answer) - 1
    
    # 找到包含answer起始和结束位置的token
    token_start = token_end = None
    
    for i, (start, end) in enumerate(offsets):
        # 找到包含答案起始位置的token
        if start <= answer_start < end:
            token_start = i
        
        # 找到包含答案结束位置的token
        if start <= answer_end < end:
            token_end = i
            break
    
    # 如果没有找到，尝试使用text匹配方法
    if token_start is None or token_end is None:
        tokens = tokenizer.tokenize(context)
        answer_tokens = tokenizer.tokenize(answer)
        
        for i in range(len(tokens)):
            if i + len(answer_tokens) <= len(tokens):
                if tokens[i:i + len(answer_tokens)] == answer_tokens:
                    token_start = i
                    token_end = i + len(answer_tokens) - 1
                    break
    
    return token_start, token_end

def convert_to_qa_format(samples, max_samples=-1, include_position=False, tokenizer_name="bert-base-chinese", use_char_position=False):
    """将样本转换为问答微调格式
    
    Args:
        samples: 数据样本
        max_samples: 最大样本数，-1表示使用所有样本
        include_position: 是否在回复中包含答案位置信息
        tokenizer_name: 使用的tokenizer名称
        use_char_position: 是否使用字符级位置而不是token级位置
    """
    logger.info("将样本转换为问答微调格式")
    
    if max_samples > 0 and max_samples < len(samples):
        samples = random.sample(samples, max_samples)
        logger.info(f"随机选择了 {max_samples} 个样本")
    
    # 加载tokenizer
    if include_position and not use_char_position:
        logger.info(f"加载tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 转换为微调格式
    finetune_samples = []
    for sample in tqdm(samples):
        # 构建问答格式
        user_prompt = f"根据以下文章回答问题：\n\n文章：{sample['context']}\n\n问题：{sample['question']}"
        
        # 根据参数决定是否包含位置信息
        if include_position:
            if use_char_position:
                # 使用字符级位置
                assistant_response = f"答案是：{sample['answer']}\n答案在原文中的起始位置是第 {sample['answer_start']} 个字符"
            else:
                # 使用token级位置
                token_start, token_end = get_token_positions(
                    sample['context'], 
                    sample['answer'], 
                    sample['answer_start'], 
                    tokenizer
                )
                
                if token_start is not None and token_end is not None:
                    assistant_response = f"答案是：{sample['answer']}\n答案在原文中的token位置是：起始token {token_start}，结束token {token_end}"
                else:
                    # 如果找不到token位置，退回到使用字符级位置
                    assistant_response = f"答案是：{sample['answer']}\n答案在原文中的起始位置是第 {sample['answer_start']} 个字符"
                    logger.warning(f"无法找到样本 {sample['id']} 的token位置，使用字符级位置代替")
        else:
            assistant_response = f"{sample['answer']}"
        
        finetune_sample = {
            "id": sample["id"],
            "conversations": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
        
        finetune_samples.append(finetune_sample)
    
    return finetune_samples

def split_train_valid(samples, valid_ratio=0.1, seed=42):
    """将训练样本分割为训练集和验证集
    
    Args:
        samples: 训练样本列表
        valid_ratio: 验证集占比，默认0.1
        seed: 随机种子
        
    Returns:
        train_samples, valid_samples: 分割后的训练集和验证集
    """
    if valid_ratio <= 0 or valid_ratio >= 1:
        raise ValueError("验证集比例必须在0到1之间")
    
    # 设置随机种子
    random.seed(seed)
    
    # 打乱样本顺序
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    # 计算验证集大小
    valid_size = math.ceil(len(shuffled_samples) * valid_ratio)
    
    # 分割样本
    valid_samples = shuffled_samples[:valid_size]
    train_samples = shuffled_samples[valid_size:]
    
    logger.info(f"将 {len(samples)} 个样本分割为 {len(train_samples)} 个训练样本和 {len(valid_samples)} 个验证样本")
    
    return train_samples, valid_samples

def save_data(data, output_path):
    """保存数据到JSON文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"数据已保存到 {output_path}")

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 设置日志
    setup_logger(
        name="attn_experiment",
        log_dir="logs",
        log_level="INFO",
        log_to_file=True,
        log_to_console=True,
        log_file_suffix="_prepare_cmrc2018"
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理训练集
    if os.path.exists(os.path.join(args.input_dir, args.train_file)):
        # 加载训练数据
        train_samples = load_squad_format_data(os.path.join(args.input_dir, args.train_file))
        
        # 如果需要分割验证集
        if args.split_valid:
            train_samples, valid_samples = split_train_valid(
                train_samples, 
                valid_ratio=args.valid_ratio, 
                seed=args.seed
            )
            
            # 处理验证集
            valid_finetune_samples = convert_to_qa_format(
                valid_samples, 
                args.max_samples, 
                args.include_position,
                args.tokenizer,
                args.use_char_position
            )
            save_data(valid_finetune_samples, os.path.join(args.output_dir, "cmrc2018_valid.json"))
        
        # 处理训练集
        train_finetune_samples = convert_to_qa_format(
            train_samples, 
            args.max_samples, 
            args.include_position,
            args.tokenizer,
            args.use_char_position
        )
        save_data(train_finetune_samples, os.path.join(args.output_dir, "cmrc2018_train.json"))
    
    # 处理测试集（使用dev数据）
    if os.path.exists(os.path.join(args.input_dir, args.test_file)):
        test_samples = load_squad_format_data(os.path.join(args.input_dir, args.test_file))
        test_finetune_samples = convert_to_qa_format(
            test_samples, 
            args.max_samples, 
            args.include_position,
            args.tokenizer,
            args.use_char_position
        )
        save_data(test_finetune_samples, os.path.join(args.output_dir, "cmrc2018_test.json"))
    
    logger.info("数据集处理完成")

if __name__ == "__main__":
    main() 