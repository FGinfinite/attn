#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成用于微调的数据集
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import load_config
from src.utils.logger import setup_logger

logger = logging.getLogger("attn_experiment")

def generate_instruction_dataset(
    num_samples: int = 100,
    output_file: str = "data/finetune/instruction_dataset.json",
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    生成指令微调数据集
    
    Args:
        num_samples: 样本数量
        output_file: 输出文件路径
        seed: 随机种子
        
    Returns:
        生成的数据集
    """
    random.seed(seed)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 定义一些指令模板
    instruction_templates = [
        "解释{}的概念",
        "如何实现{}",
        "比较{}和{}的区别",
        "分析{}的优缺点",
        "总结{}的主要内容",
        "列举{}的应用场景",
        "描述{}的工作原理",
        "评估{}的性能表现",
        "预测{}的未来发展趋势",
        "提供关于{}的建议"
    ]
    
    # 定义一些主题
    topics = [
        "注意力机制", "Transformer模型", "自回归生成", "模型量化", "知识蒸馏",
        "稀疏注意力", "线性注意力", "FlashAttention", "模型并行", "张量并行",
        "流水线并行", "混合精度训练", "梯度累积", "梯度检查点", "分布式训练",
        "PEFT", "LoRA", "QLoRA", "AdaLoRA", "P-tuning",
        "Prompt-tuning", "In-context Learning", "Few-shot Learning", "Zero-shot Learning", "迁移学习",
        "对比学习", "自监督学习", "强化学习", "PPO", "RLHF",
        "DPO", "KTO", "IPO", "模型对齐", "安全对齐",
        "模型评估", "困惑度", "ROUGE", "BLEU", "BERTScore",
        "推理加速", "KV Cache", "Continuous Batching", "Speculative Decoding", "Medusa",
        "大模型推理", "大模型训练", "大模型部署", "大模型应用", "大模型安全"
    ]
    
    # 生成数据集
    dataset = []
    for _ in range(num_samples):
        # 随机选择指令模板
        template = random.choice(instruction_templates)
        
        # 根据模板格式确定需要的主题数量
        if "{}" in template and "{}" in template[template.index("{}") + 2:]:
            # 需要两个主题
            topic1, topic2 = random.sample(topics, 2)
            instruction = template.format(topic1, topic2)
        else:
            # 需要一个主题
            topic = random.choice(topics)
            instruction = template.format(topic)
        
        # 生成回答
        response = generate_response_for_instruction(instruction)
        
        # 添加到数据集
        dataset.append({
            "instruction": instruction,
            "response": response
        })
    
    # 保存数据集
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已生成{num_samples}个样本，保存到{output_file}")
    
    return dataset

def generate_response_for_instruction(instruction: str) -> str:
    """
    为指令生成回答
    
    Args:
        instruction: 指令
        
    Returns:
        生成的回答
    """
    # 这里简化处理，实际应用中可以使用更复杂的方法生成高质量回答
    # 例如使用现有的大模型生成回答
    
    # 提取指令中的主题
    topics = []
    for topic in [
        "注意力机制", "Transformer模型", "自回归生成", "模型量化", "知识蒸馏",
        "稀疏注意力", "线性注意力", "FlashAttention", "模型并行", "张量并行"
    ]:
        if topic in instruction:
            topics.append(topic)
    
    if "注意力机制" in topics:
        return """注意力机制是深度学习中的一种重要技术，它允许模型在处理序列数据时关注输入的特定部分。

在传统的序列模型中，所有输入元素被平等对待，而注意力机制允许模型对不同输入元素赋予不同的权重，从而"关注"更相关的信息。

注意力机制的核心组成部分：
1. 查询(Query)：当前处理的元素
2. 键(Key)：用于与查询计算相似度的表示
3. 值(Value)：实际被聚合的信息

计算过程：
1. 计算查询与所有键的相似度得分
2. 对得分进行归一化（通常使用softmax函数）
3. 使用归一化后的得分对值进行加权求和

注意力机制的变体：
- 自注意力(Self-Attention)：查询、键和值来自同一序列
- 多头注意力(Multi-head Attention)：并行计算多组注意力，然后合并结果
- 掩码注意力(Masked Attention)：在某些位置上应用掩码，防止信息泄露

注意力机制是Transformer架构的核心组件，为现代大型语言模型的成功奠定了基础。"""
    
    elif "Transformer模型" in topics:
        return """Transformer模型是由Google在2017年提出的一种基于自注意力机制的神经网络架构，论文标题为"Attention is All You Need"。

Transformer的核心创新在于完全依赖注意力机制处理序列数据，摒弃了之前广泛使用的循环神经网络(RNN)和卷积神经网络(CNN)。

Transformer架构的主要组件：
1. 多头自注意力机制(Multi-head Self-attention)：允许模型同时关注不同位置的信息
2. 位置编码(Positional Encoding)：为模型提供序列中位置信息
3. 前馈神经网络(Feed-forward Neural Network)：对每个位置独立应用的全连接层
4. 残差连接(Residual Connection)和层归一化(Layer Normalization)：帮助训练更深的网络

Transformer分为编码器(Encoder)和解码器(Decoder)两部分：
- 编码器：处理输入序列，生成表示
- 解码器：基于编码器输出和之前生成的输出，生成新的输出

Transformer的优势：
1. 并行计算：不依赖序列的顺序处理，可以高度并行化
2. 长距离依赖：能够直接建模序列中任意位置之间的关系
3. 可扩展性：架构可以轻松扩展到更大规模

Transformer是现代大型语言模型(如GPT、BERT、T5等)的基础架构，推动了自然语言处理领域的重大进展。"""
    
    else:
        # 默认回答
        return f"""关于{instruction}的回答：

这是一个与人工智能和深度学习相关的重要主题。在现代大语言模型的发展中，这一技术发挥了关键作用。

主要特点包括：
1. 提高了模型的性能和效率
2. 解决了传统方法面临的挑战
3. 为未来的研究提供了新的方向

应用场景广泛，包括自然语言处理、计算机视觉、语音识别等多个领域。

随着技术的不断发展，我们可以预期在未来会看到更多创新和突破。"""
    
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成微调数据集")
    parser.add_argument("--num_samples", type=int, default=100, help="样本数量")
    parser.add_argument("--output_file", type=str, default="data/finetune/instruction_dataset.json", help="输出文件路径")
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
    
    logger.info("开始生成微调数据集")
    
    # 生成数据集
    generate_instruction_dataset(
        num_samples=args.num_samples,
        output_file=args.output_file,
        seed=args.seed
    )
    
    logger.info("微调数据集生成完成")

if __name__ == "__main__":
    main() 