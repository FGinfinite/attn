"""
测试用例模块
"""

TEST_CASES = [
    {
        "name": "基础测试",
        "prompt": "请介绍一下你自己。",
        "max_new_tokens": 50,
        "description": "基础自我介绍测试"
    },
    {
        "name": "复杂推理测试",
        "prompt": "在一个有100个房间的酒店里，每个房间都有编号1-100。现在有100个客人，每个客人都有自己的房间号。第一个客人打开所有房间的门，第二个客人关上所有2的倍数的门，第三个客人改变所有3的倍数的门的状态（如果开着就关上，如果关着就打开），以此类推。请问第100个客人操作后，哪些房间的门是开着的？请详细解释推理过程。",
        "max_new_tokens": 200,
        "description": "测试模型的逻辑推理能力"
    },
    {
        "name": "多轮对话测试",
        "prompt": "让我们进行一个关于量子计算的对话。首先，请解释一下量子计算的基本原理。",
        "max_new_tokens": 150,
        "description": "测试模型在专业领域的知识储备"
    },
    {
        "name": "长文本生成测试",
        "prompt": "请写一个关于人工智能未来发展的科幻故事，包含以下要素：1. 时间背景设定在2050年 2. 涉及量子计算机 3. 包含人机协作 4. 有情感冲突",
        "max_new_tokens": 300,
        "description": "测试模型的长文本生成能力"
    },
    {
        "name": "多语言混合测试",
        "prompt": "请用中文、英文和日文分别写一句关于春天的诗句，并解释每句诗的含义。",
        "max_new_tokens": 200,
        "description": "测试模型的多语言能力"
    },
    {
        "name": "代码生成测试",
        "prompt": "请用Python实现一个支持以下功能的神经网络：1. 使用PyTorch框架 2. 包含注意力机制 3. 支持批处理 4. 包含dropout层 5. 使用Adam优化器",
        "max_new_tokens": 250,
        "description": "测试模型的代码生成能力"
    },
    {
        "name": "数学问题测试",
        "prompt": "请解决以下数学问题并详细说明解题思路：设f(x)是一个连续函数，满足f(x+y)=f(x)+f(y)对所有实数x,y都成立。证明f(x)一定是线性函数。",
        "max_new_tokens": 200,
        "description": "测试模型的数学推理能力"
    },
    {
        "name": "创意写作测试",
        "prompt": "请创作一首关于人工智能的现代诗，要求：1. 使用意象派手法 2. 包含科技元素 3. 体现人文关怀 4. 结构完整",
        "max_new_tokens": 150,
        "description": "测试模型的创意写作能力"
    }
]

def get_test_case(name):
    """
    获取指定名称的测试用例
    
    Args:
        name: 测试用例名称
    
    Returns:
        dict: 测试用例配置
    """
    for case in TEST_CASES:
        if case["name"] == name:
            return case
    return None

def get_all_test_cases(max_cases=None):
    """
    获取测试用例
    
    Args:
        max_cases: 最大测试用例数量，None表示返回所有测试用例
    
    Returns:
        list: 测试用例列表
    """
    if max_cases is None or max_cases >= len(TEST_CASES):
        return TEST_CASES
    else:
        return TEST_CASES[:max_cases] 