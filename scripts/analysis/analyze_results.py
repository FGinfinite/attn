"""
结果分析脚本
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EXPERIMENT_CONFIG, LOGGING_CONFIG
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="结果分析脚本")
    
    parser.add_argument("--results_dir", type=str, default=str(EXPERIMENT_CONFIG["results_dir"]),
                        help="结果目录")
    parser.add_argument("--output_dir", type=str, default="./analysis",
                        help="分析结果输出目录")
    parser.add_argument("--metrics", type=str, default="latency,tokens_per_second,memory_usage,perplexity",
                        help="要分析的指标，用逗号分隔")
    
    return parser.parse_args()

def load_results(results_dir):
    """加载结果"""
    results_dir = Path(results_dir)
    
    # 查找所有JSON结果文件
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print(f"在目录 {results_dir} 中未找到JSON结果文件")
        return None
    
    # 加载结果
    results = []
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding='utf-8') as f:
                data = json.load(f)
                
                # 提取配置和摘要
                config = data.get("config", {})
                summary = data.get("summary", {})
                
                # 合并配置和摘要
                result = {**config, **summary}
                results.append(result)
        
        except Exception as e:
            print(f"加载文件 {json_file} 时出错: {str(e)}")
    
    # 转换为DataFrame
    if results:
        df = pd.DataFrame(results)
        return df
    
    return None

def analyze_results(df, metrics, output_dir):
    """分析结果"""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建分析报告
    report_file = output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, "w", encoding='utf-8') as f:
        f.write("# 注意力机制对比实验分析报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 数据概览\n\n")
        f.write("### 1.1 数据集统计\n\n")
        f.write(f"- 总样本数: {len(df)}\n")
        f.write(f"- 量化方式: {', '.join(df['quant'].unique())}\n")
        f.write(f"- 注意力机制: {', '.join(df['attention'].unique())}\n")
        f.write(f"- 批处理大小: {', '.join(map(str, df['batch_size'].unique()))}\n")
        f.write(f"- 输入长度: {', '.join(map(str, df['input_length'].unique()))}\n")
        f.write(f"- 输出长度: {', '.join(map(str, df['output_length'].unique()))}\n\n")
        
        f.write("### 1.2 指标统计\n\n")
        
        # 创建指标统计表格
        f.write("| 指标 | 平均值 | 标准差 | 最小值 | 最大值 |\n")
        f.write("| ---- | ------ | ------ | ------ | ------ |\n")
        
        for metric in metrics:
            if f"{metric}_mean" in df.columns:
                mean = df[f"{metric}_mean"].mean()
                std = df[f"{metric}_mean"].std()
                min_val = df[f"{metric}_mean"].min()
                max_val = df[f"{metric}_mean"].max()
                
                f.write(f"| {metric} | {mean:.2f} | {std:.2f} | {min_val:.2f} | {max_val:.2f} |\n")
        
        f.write("\n## 2. 量化方式对比\n\n")
        
        # 对每个指标进行分析
        for metric in metrics:
            if f"{metric}_mean" in df.columns:
                f.write(f"### 2.1 {metric} 对比\n\n")
                
                # 创建量化方式对比表格
                f.write("| 量化方式 | 平均值 | 标准差 | 最小值 | 最大值 |\n")
                f.write("| -------- | ------ | ------ | ------ | ------ |\n")
                
                for quant in df["quant"].unique():
                    quant_df = df[df["quant"] == quant]
                    
                    mean = quant_df[f"{metric}_mean"].mean()
                    std = quant_df[f"{metric}_mean"].std()
                    min_val = quant_df[f"{metric}_mean"].min()
                    max_val = quant_df[f"{metric}_mean"].max()
                    
                    f.write(f"| {quant} | {mean:.2f} | {std:.2f} | {min_val:.2f} | {max_val:.2f} |\n")
                
                f.write("\n")
        
        f.write("\n## 3. 注意力机制对比\n\n")
        
        # 对每个指标进行分析
        for metric in metrics:
            if f"{metric}_mean" in df.columns:
                f.write(f"### 3.1 {metric} 对比\n\n")
                
                # 创建注意力机制对比表格
                f.write("| 注意力机制 | 平均值 | 标准差 | 最小值 | 最大值 |\n")
                f.write("| ---------- | ------ | ------ | ------ | ------ |\n")
                
                for attention in df["attention"].unique():
                    attention_df = df[df["attention"] == attention]
                    
                    mean = attention_df[f"{metric}_mean"].mean()
                    std = attention_df[f"{metric}_mean"].std()
                    min_val = attention_df[f"{metric}_mean"].min()
                    max_val = attention_df[f"{metric}_mean"].max()
                    
                    f.write(f"| {attention} | {mean:.2f} | {std:.2f} | {min_val:.2f} | {max_val:.2f} |\n")
                
                f.write("\n")
        
        f.write("\n## 4. 量化方式与注意力机制组合对比\n\n")
        
        # 对每个指标进行分析
        for metric in metrics:
            if f"{metric}_mean" in df.columns:
                f.write(f"### 4.1 {metric} 对比\n\n")
                
                # 创建组合对比表格
                f.write("| 量化方式 | 注意力机制 | 平均值 | 标准差 | 最小值 | 最大值 |\n")
                f.write("| -------- | ---------- | ------ | ------ | ------ | ------ |\n")
                
                for quant in df["quant"].unique():
                    for attention in df["attention"].unique():
                        combo_df = df[(df["quant"] == quant) & (df["attention"] == attention)]
                        
                        if len(combo_df) > 0:
                            mean = combo_df[f"{metric}_mean"].mean()
                            std = combo_df[f"{metric}_mean"].std()
                            min_val = combo_df[f"{metric}_mean"].min()
                            max_val = combo_df[f"{metric}_mean"].max()
                            
                            f.write(f"| {quant} | {attention} | {mean:.2f} | {std:.2f} | {min_val:.2f} | {max_val:.2f} |\n")
                
                f.write("\n")
        
        f.write("\n## 5. 结论\n\n")
        f.write("根据以上分析，我们可以得出以下结论：\n\n")
        
        # 生成延迟结论
        if "latency_mean" in df.columns:
            best_latency = df.loc[df["latency_mean"].idxmin()]
            f.write(f"1. 延迟最低的组合是：量化方式={best_latency['quant']}，注意力机制={best_latency['attention']}，平均延迟={best_latency['latency_mean']:.2f}ms\n")
        
        # 生成生成速度结论
        if "tokens_per_second_mean" in df.columns:
            best_speed = df.loc[df["tokens_per_second_mean"].idxmax()]
            f.write(f"2. 生成速度最快的组合是：量化方式={best_speed['quant']}，注意力机制={best_speed['attention']}，平均生成速度={best_speed['tokens_per_second_mean']:.2f}token/s\n")
        
        # 生成显存使用结论
        if "memory_usage_mean" in df.columns:
            best_memory = df.loc[df["memory_usage_mean"].idxmin()]
            f.write(f"3. 显存使用最少的组合是：量化方式={best_memory['quant']}，注意力机制={best_memory['attention']}，平均显存使用={best_memory['memory_usage_mean']:.2f}MB\n")
        
        # 生成困惑度结论
        if "perplexity_mean" in df.columns:
            best_perplexity = df.loc[df["perplexity_mean"].idxmin()]
            f.write(f"4. 困惑度最低的组合是：量化方式={best_perplexity['quant']}，注意力机制={best_perplexity['attention']}，平均困惑度={best_perplexity['perplexity_mean']:.2f}\n")
    
    print(f"分析报告已保存到: {report_file}")
    
    # 创建可视化图表
    for metric in metrics:
        if f"{metric}_mean" in df.columns:
            # 创建量化方式对比图
            plt.figure(figsize=(10, 6))
            sns.barplot(x="quant", y=f"{metric}_mean", data=df)
            plt.title(f"{metric} by Quantization")
            plt.xlabel("Quantization")
            plt.ylabel(metric)
            plt.savefig(output_dir / f"{metric}_by_quant.png")
            plt.close()
            
            # 创建注意力机制对比图
            plt.figure(figsize=(10, 6))
            sns.barplot(x="attention", y=f"{metric}_mean", data=df)
            plt.title(f"{metric} by Attention")
            plt.xlabel("Attention")
            plt.ylabel(metric)
            plt.savefig(output_dir / f"{metric}_by_attention.png")
            plt.close()
            
            # 创建组合对比图
            plt.figure(figsize=(12, 8))
            sns.barplot(x="quant", y=f"{metric}_mean", hue="attention", data=df)
            plt.title(f"{metric} by Quantization and Attention")
            plt.xlabel("Quantization")
            plt.ylabel(metric)
            plt.legend(title="Attention")
            plt.savefig(output_dir / f"{metric}_by_quant_attention.png")
            plt.close()
    
    print(f"可视化图表已保存到: {output_dir}")
    
    return report_file

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    os.makedirs(LOGGING_CONFIG["log_dir"], exist_ok=True)
    logger = setup_logger(
        name="analyze_results",
        log_dir=LOGGING_CONFIG["log_dir"],
        log_level=LOGGING_CONFIG["log_level"],
        log_to_file=LOGGING_CONFIG["log_to_file"],
        log_to_console=LOGGING_CONFIG["log_to_console"]
    )
    
    logger.info(f"开始分析结果")
    logger.info(f"结果目录: {args.results_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 解析指标
    metrics = args.metrics.split(",")
    logger.info(f"要分析的指标: {metrics}")
    
    # 加载结果
    df = load_results(args.results_dir)
    
    if df is None:
        logger.error(f"未找到结果数据")
        return
    
    logger.info(f"加载了 {len(df)} 条结果数据")
    
    # 分析结果
    report_file = analyze_results(df, metrics, args.output_dir)
    
    logger.info(f"分析完成，报告已保存到: {report_file}")

if __name__ == "__main__":
    main() 