#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成用于微调的示例数据集
"""

import os
import json
import argparse
import random
from pathlib import Path
import logging

logger = logging.getLogger("attn_experiment")

# 示例指令模板
INSTRUCTION_TEMPLATES = [
    "解释{topic}的概念",
    "总结{topic}的主要观点",
    "比较{topic}和{topic2}的区别",
    "分析{topic}的优缺点",
    "描述{topic}的实际应用场景",
    "如何在实践中应用{topic}",
    "详细介绍{topic}的历史发展",
    "为初学者解释{topic}",
    "{topic}有哪些常见的误解",
    "如何评估{topic}的效果",
]

# 示例主题
TOPICS = [
    "深度学习", "自然语言处理", "计算机视觉", "强化学习", "神经网络",
    "注意力机制", "Transformer模型", "卷积神经网络", "循环神经网络", 
    "迁移学习", "对比学习", "自监督学习", "图神经网络", "生成对抗网络",
    "知识图谱", "语义分割", "目标检测", "机器翻译", "情感分析", 
    "问答系统", "推荐系统", "时间序列分析", "图像生成", "元学习",
    "联邦学习", "量子计算", "边缘计算", "雾计算", "区块链技术",
]

# 第二主题（用于比较类指令）
TOPICS2 = [
    "传统机器学习", "规则系统", "统计模型", "决策树", "支持向量机",
    "朴素贝叶斯", "K近邻算法", "线性回归", "逻辑回归", "聚类算法",
    "主成分分析", "遗传算法", "粒子群优化", "蚁群算法", "模糊逻辑",
    "马尔可夫链", "隐马尔可夫模型", "贝叶斯网络", "随机森林", "梯度提升树",
]

# 回答模板
RESPONSE_TEMPLATES = [
    "作为一个AI助手，我会尽力解答您的问题。\n\n{content}",
    "感谢您的提问，以下是我的回答：\n\n{content}",
    "关于这个问题，我可以提供以下信息：\n\n{content}",
    "我很乐意回答这个问题。\n\n{content}",
    "这是一个很好的问题，让我来解答：\n\n{content}",
]

# 针对不同指令类型的回答内容
CONTENT_TEMPLATES = {
    "解释": [
        "{topic}是{field}领域中的一个重要概念，它主要指{definition}。{topic}的核心思想是{core_idea}。在实际应用中，{topic}可以解决{problem}等问题。",
        "{topic}是指{definition}。这一概念最早由{originator}在{year}年提出，目的是解决{problem}。{topic}的基本原理是{principle}。",
        "{topic}可以被定义为{definition}。它在{field}领域有着广泛的应用，特别是在解决{problem}方面表现出色。{topic}的关键特点包括{features}。",
    ],
    "总结": [
        "{topic}的主要观点可以概括为以下几点：\n1. {point1}\n2. {point2}\n3. {point3}\n4. {point4}\n\n总的来说，{topic}强调{emphasis}。",
        "关于{topic}的核心观点包括：\n- {point1}\n- {point2}\n- {point3}\n- {point4}\n\n这些观点共同构成了{topic}的理论框架。",
        "{topic}的关键思想可以总结如下：\n首先，{point1}。\n其次，{point2}。\n再次，{point3}。\n最后，{point4}。\n这些思想相互关联，形成了{topic}的完整体系。",
    ],
    "比较": [
        "{topic}和{topic2}是{field}领域中两个不同的概念。\n\n{topic}的特点是{feature1}，而{topic2}的特点是{feature2}。\n\n在应用场景方面，{topic}更适合{scenario1}，而{topic2}更适合{scenario2}。\n\n从效果来看，{topic}在{advantage1}方面表现更好，而{topic2}在{advantage2}方面具有优势。",
        "比较{topic}和{topic2}：\n\n1. 原理不同：{topic}基于{principle1}，而{topic2}基于{principle2}。\n2. 适用范围不同：{topic}主要用于{scope1}，而{topic2}主要用于{scope2}。\n3. 复杂度不同：{topic}的计算复杂度为{complexity1}，而{topic2}的计算复杂度为{complexity2}。\n4. 效果不同：{topic}在{metric1}上表现更好，而{topic2}在{metric2}上更占优势。",
        "{topic}与{topic2}的区别主要体现在以下几个方面：\n\n- 历史发展：{topic}起源于{origin1}，而{topic2}起源于{origin2}。\n- 核心思想：{topic}强调{core1}，而{topic2}注重{core2}。\n- 技术实现：{topic}通常采用{tech1}实现，而{topic2}常用{tech2}实现。\n- 未来趋势：{topic}的发展方向是{trend1}，而{topic2}正朝着{trend2}发展。",
    ],
    "分析": [
        "{topic}的优点包括：\n1. {advantage1}\n2. {advantage2}\n3. {advantage3}\n\n缺点包括：\n1. {disadvantage1}\n2. {disadvantage2}\n3. {disadvantage3}\n\n总体而言，{topic}在{context}场景下能发挥最大价值，但在{limitation}情况下可能效果有限。",
        "对{topic}的优缺点分析：\n\n优势方面：\n- {advantage1}\n- {advantage2}\n- {advantage3}\n\n劣势方面：\n- {disadvantage1}\n- {disadvantage2}\n- {disadvantage3}\n\n在实际应用中，需要根据具体场景权衡这些因素。",
        "{topic}的优缺点分析如下：\n\n【优点】\n* {advantage1}\n* {advantage2}\n* {advantage3}\n\n【缺点】\n* {disadvantage1}\n* {disadvantage2}\n* {disadvantage3}\n\n【使用建议】\n在{condition1}条件下优先考虑使用{topic}，而在{condition2}条件下可能需要考虑其他替代方案。",
    ],
    "描述": [
        "{topic}在实际应用中有多种场景：\n\n1. {scenario1}：{description1}\n2. {scenario2}：{description2}\n3. {scenario3}：{description3}\n4. {scenario4}：{description4}\n\n其中，{scenario1}是最为常见和成熟的应用场景。",
        "{topic}的实际应用场景包括：\n\n- 在{industry1}行业：{application1}\n- 在{industry2}行业：{application2}\n- 在{industry3}行业：{application3}\n- 在{industry4}行业：{application4}\n\n这些应用显示了{topic}的广泛适用性和实用价值。",
        "{topic}可以应用于多个现实场景：\n\n【场景一】{scenario1}\n{description1}\n\n【场景二】{scenario2}\n{description2}\n\n【场景三】{scenario3}\n{description3}\n\n【场景四】{scenario4}\n{description4}\n\n通过这些应用，{topic}已经证明了其在解决实际问题方面的强大能力。",
    ],
}

def fill_template(template, topic=None, topic2=None):
    """
    填充模板中的占位符
    """
    # 随机生成一些填充内容
    fillers = {
        "field": random.choice(["人工智能", "机器学习", "数据科学", "计算机科学"]),
        "definition": f"一种用于{random.choice(['处理', '分析', '理解', '生成'])}{random.choice(['数据', '信息', '知识', '模式'])}的技术",
        "core_idea": f"通过{random.choice(['优化', '学习', '推理', '识别'])}来{random.choice(['提高效率', '增强性能', '解决复杂问题', '自动化决策'])}",
        "problem": f"{random.choice(['数据稀疏', '过拟合', '梯度消失', '特征提取'])}问题",
        "originator": random.choice(["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng", "Ian Goodfellow"]),
        "year": random.choice(["2012", "2014", "2016", "2018", "2020"]),
        "principle": f"通过{random.choice(['反向传播', '梯度下降', '注意力机制', '自监督学习'])}实现{random.choice(['特征学习', '模式识别', '决策优化', '预测分析'])}",
        "features": f"{random.choice(['高效性', '可扩展性', '鲁棒性', '适应性'])}和{random.choice(['泛化能力', '表示学习', '端到端训练', '多任务处理'])}",
        
        "point1": f"强调{random.choice(['数据驱动', '模型架构', '优化算法', '评估方法'])}的重要性",
        "point2": f"提出了{random.choice(['新的训练策略', '创新的网络结构', '改进的损失函数', '高效的推理方法'])}",
        "point3": f"解决了{random.choice(['长期依赖', '稀疏表示', '多模态融合', '知识迁移'])}问题",
        "point4": f"为{random.choice(['自动化系统', '智能应用', '人机交互', '决策支持'])}提供了理论基础",
        "emphasis": f"{random.choice(['模型性能与效率的平衡', '理论与实践的结合', '通用性与专用性的权衡', '创新与稳定性的协调'])}",
        
        "feature1": f"基于{random.choice(['深度学习', '神经网络', '概率模型', '规则系统'])}",
        "feature2": f"依赖于{random.choice(['传统算法', '统计方法', '数学模型', '逻辑推理'])}",
        "scenario1": f"{random.choice(['大规模数据处理', '复杂模式识别', '实时预测分析', '多模态任务'])}",
        "scenario2": f"{random.choice(['小数据场景', '可解释性要求高的任务', '计算资源有限的环境', '严格约束条件下的应用'])}",
        "advantage1": f"{random.choice(['准确性', '效率', '泛化能力', '鲁棒性'])}",
        "advantage2": f"{random.choice(['可解释性', '计算效率', '数据效率', '易用性'])}",
        
        "principle1": f"{random.choice(['表示学习', '深度架构', '端到端优化', '数据驱动'])}",
        "principle2": f"{random.choice(['逻辑规则', '统计假设', '数学推导', '专家知识'])}",
        "scope1": f"{random.choice(['复杂非线性问题', '大规模数据集', '多模态任务', '序列预测'])}",
        "scope2": f"{random.choice(['线性问题', '小数据集', '特定领域任务', '静态分类'])}",
        "complexity1": f"{random.choice(['O(n²)', 'O(n log n)', 'O(n)', 'O(n³)'])}",
        "complexity2": f"{random.choice(['O(log n)', 'O(1)', 'O(n)', 'O(n log n)'])}",
        "metric1": f"{random.choice(['准确率', '召回率', 'F1得分', 'AUC'])}",
        "metric2": f"{random.choice(['训练时间', '推理速度', '内存占用', '可解释性'])}",
        
        "origin1": f"{random.choice(['2010年代', '2000年代', '1990年代', '2020年代'])}",
        "origin2": f"{random.choice(['1980年代', '1970年代', '1960年代', '1950年代'])}",
        "core1": f"{random.choice(['非线性表达', '端到端学习', '表示学习', '分层抽象'])}",
        "core2": f"{random.choice(['线性模型', '特征工程', '统计推断', '规则设计'])}",
        "tech1": f"{random.choice(['深度神经网络', '复杂网络架构', '分布式训练', '自动微分'])}",
        "tech2": f"{random.choice(['传统算法', '数学优化', '统计方法', '规则引擎'])}",
        "trend1": f"{random.choice(['多模态融合', '自监督学习', '高效小型模型', '可解释AI'])}",
        "trend2": f"{random.choice(['集成方法', '混合模型', '专家系统', '人机协作'])}",
        
        "advantage1": f"在{random.choice(['复杂任务', '大规模数据', '非结构化信息', '多模态场景'])}中表现出色",
        "advantage2": f"能够{random.choice(['自动学习特征', '端到端优化', '处理高维数据', '适应新环境'])}",
        "advantage3": f"具有{random.choice(['较强的泛化能力', '良好的可扩展性', '持续改进的潜力', '处理不确定性的能力'])}",
        "disadvantage1": f"需要{random.choice(['大量训练数据', '高性能计算资源', '专业调参经验', '长时间训练'])}",
        "disadvantage2": f"{random.choice(['可解释性差', '泛化到新领域困难', '对数据质量敏感', '训练不稳定'])}",
        "disadvantage3": f"在{random.choice(['小数据场景', '高度可解释性要求', '严格实时约束', '资源受限环境'])}中表现不佳",
        "context": f"{random.choice(['数据丰富', '计算资源充足', '非实时要求', '复杂模式识别'])}",
        "limitation": f"{random.choice(['数据稀少', '实时性要求高', '解释性要求严格', '计算资源有限'])}",
        
        "condition1": f"{random.choice(['数据量大', '任务复杂', '非线性关系明显', '人工特征困难'])}",
        "condition2": f"{random.choice(['数据量小', '任务简单', '线性关系足够', '需要高度可解释性'])}",
        
        "scenario1": f"{random.choice(['医疗诊断', '金融预测', '自然语言处理', '计算机视觉'])}",
        "description1": f"通过{random.choice(['分析病历数据', '处理金融时序', '理解文本语义', '识别图像内容'])}来{random.choice(['辅助诊断疾病', '预测市场走势', '实现智能对话', '完成目标检测'])}",
        "scenario2": f"{random.choice(['智能制造', '智慧城市', '推荐系统', '自动驾驶'])}",
        "description2": f"利用{random.choice(['工业传感器数据', '城市监控信息', '用户行为数据', '多传感器融合'])}实现{random.choice(['生产优化', '交通管理', '个性化推荐', '路径规划'])}",
        "scenario3": f"{random.choice(['教育评估', '能源管理', '安防监控', '环境监测'])}",
        "description3": f"基于{random.choice(['学习行为分析', '能耗数据', '异常行为检测', '环境参数'])}进行{random.choice(['个性化学习', '能源优化', '风险预警', '污染预测'])}",
        "scenario4": f"{random.choice(['娱乐创作', '农业管理', '零售分析', '客户服务'])}",
        "description4": f"结合{random.choice(['创意生成', '农作物监测', '购物行为分析', '客户反馈'])}提供{random.choice(['内容创作', '产量预测', '销售策略', '智能客服'])}",
        
        "industry1": f"{random.choice(['医疗', '金融', '教育', '制造'])}",
        "application1": f"{random.choice(['疾病诊断', '风险评估', '个性化学习', '质量控制'])}",
        "industry2": f"{random.choice(['零售', '交通', '能源', '农业'])}",
        "application2": f"{random.choice(['需求预测', '路况分析', '用量优化', '作物管理'])}",
        "industry3": f"{random.choice(['安防', '娱乐', '通信', '房地产'])}",
        "application3": f"{random.choice(['监控分析', '内容推荐', '网络优化', '价格预测'])}",
        "industry4": f"{random.choice(['公共服务', '环保', '法律', '体育'])}",
        "application4": f"{random.choice(['市民服务', '污染监测', '案例分析', '表现评估'])}",
    }
    
    # 替换模板中的主题
    if "{topic}" in template:
        template = template.replace("{topic}", topic)
    if "{topic2}" in template:
        template = template.replace("{topic2}", topic2 or random.choice(TOPICS2))
    
    # 替换其他占位符
    for key, value in fillers.items():
        placeholder = "{" + key + "}"
        if placeholder in template:
            template = template.replace(placeholder, value)
    
    return template

def generate_example(instruction_template, topic=None):
    """
    生成单个示例数据
    """
    # 随机选择主题（如果未提供）
    if topic is None:
        topic = random.choice(TOPICS)
    
    # 随机选择第二主题（用于比较类指令）
    topic2 = random.choice(TOPICS2)
    
    # 生成指令
    instruction = instruction_template.replace("{topic}", topic)
    if "{topic2}" in instruction_template:
        instruction = instruction.replace("{topic2}", topic2)
    
    # 确定指令类型
    instruction_type = "描述"  # 默认类型
    for key in CONTENT_TEMPLATES.keys():
        if key in instruction_template:
            instruction_type = key
            break
    
    # 生成回答内容
    content_template = random.choice(CONTENT_TEMPLATES.get(instruction_type, CONTENT_TEMPLATES["描述"]))
    content = fill_template(content_template, topic, topic2)
    
    # 生成完整回答
    response_template = random.choice(RESPONSE_TEMPLATES)
    response = response_template.replace("{content}", content)
    
    return {
        "instruction": instruction,
        "input": "",  # 大多数指令微调不需要额外输入
        "output": response
    }

def generate_dataset(size=100, output_path="data/finetune/dataset.json"):
    """
    生成微调数据集
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 生成示例数据
    dataset = []
    for _ in range(size):
        # 随机选择指令模板
        instruction_template = random.choice(INSTRUCTION_TEMPLATES)
        # 生成示例
        example = generate_example(instruction_template)
        dataset.append(example)
    
    # 保存数据集
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"生成了包含 {size} 个示例的数据集，保存至 {output_path}")
    return output_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成微调数据集")
    parser.add_argument("--size", type=int, default=100, help="数据集大小")
    parser.add_argument("--output", type=str, default="data/finetune/dataset.json", help="输出文件路径")
    args = parser.parse_args()
    
    generate_dataset(args.size, args.output)

if __name__ == "__main__":
    main() 