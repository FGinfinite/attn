"""
模型验证脚本，用于验证模型的基本功能
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from config import load_config, DEFAULT_MODEL_PATH, LOG_DIR
from src.utils.logger import setup_logger
from src.utils.model_utils import load_model_and_tokenizer, verify_model
from src.utils.hardware_monitor import HardwareMonitor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型验证脚本")
    
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径或名称")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下自己。",
                        help="测试提示词")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成的最大token数量")
    parser.add_argument("--monitor", action="store_true",
                        help="是否监控硬件使用情况")
    parser.add_argument("--print_structure", action="store_true", default=True,
                        help="是否打印模型结构")
    parser.add_argument("--attention_only", action="store_true", default=False,
                        help="是否只打印注意力层结构")
    parser.add_argument("--no_console_output", action="store_true", default=False,
                        help="禁用控制台输出，适用于重定向日志到文件")
    parser.add_argument("--output_file", type=str, default=None,
                        help="指定输出日志文件路径，不使用时间戳")
    
    return parser.parse_args()

def print_model_structure(model, logger, attention_only=False):
    """
    打印模型结构，特别关注自注意力层
    
    Args:
        model: 模型对象
        logger: 日志记录器
        attention_only: 是否只打印注意力层
    """
    logger.info("=" * 50)
    logger.info("模型结构详情:")
    logger.info("=" * 50)
    
    # 打印模型基本信息
    logger.info(f"模型类型: {model.__class__.__name__}")
    logger.info(f"模型参数总量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数总量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 打印模型配置
    if hasattr(model, 'config'):
        logger.info("-" * 50)
        logger.info("模型配置:")
        for key, value in model.config.to_dict().items():
            if isinstance(value, (int, float, str, bool, type(None))):
                logger.info(f"  {key}: {value}")
        
        # 打印模型架构相关配置
        logger.info("-" * 50)
        logger.info("模型架构配置:")
        arch_keys = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 
                    'intermediate_size', 'hidden_act', 'max_position_embeddings']
        for key in arch_keys:
            if hasattr(model.config, key):
                logger.info(f"  {key}: {getattr(model.config, key)}")
    
    # 递归打印模型结构，特别关注自注意力层
    logger.info("-" * 50)
    logger.info("模型层次结构:")
    
    def print_module(module, prefix='', depth=0):
        """递归打印模块结构"""
        if attention_only and 'attention' not in module.__class__.__name__.lower():
            # 如果只打印注意力层且当前模块不是注意力相关，则跳过
            for name, child in module.named_children():
                print_module(child, prefix + '.' + name if prefix else name, depth + 1)
            return
        
        # 打印当前模块
        indent = '  ' * depth
        module_name = module.__class__.__name__
        params = sum(p.numel() for p in module.parameters())
        
        # 特别标记注意力层
        is_attention = 'attention' in module_name.lower()
        marker = '【注意力层】' if is_attention else ''
        
        logger.info(f"{indent}{prefix}: {module_name} ({params:,} 参数) {marker}")
        
        # 如果是注意力层，打印更详细的信息
        if is_attention:
            logger.info(f"{indent}  - 详细结构:")
            for attr_name, attr_value in module.__dict__.items():
                if attr_name.startswith('_'):
                    continue
                if isinstance(attr_value, (int, float, str, bool, type(None))):
                    logger.info(f"{indent}    {attr_name}: {attr_value}")
            
            # 分析注意力层的内部结构
            analyze_attention_layer(module, logger, indent + "  ")
        
        # 递归处理子模块
        for name, child in module.named_children():
            print_module(child, prefix + '.' + name if prefix else name, depth + 1)
    
    # 从模型的顶层开始打印
    for name, module in model.named_children():
        print_module(module, name, 0)
    
    # 尝试找出所有注意力层的路径
    logger.info("-" * 50)
    logger.info("注意力层路径汇总:")
    
    def find_attention_layers(module, path=''):
        """查找所有注意力层的路径"""
        attention_paths = []
        
        if 'attention' in module.__class__.__name__.lower():
            attention_paths.append(path)
        
        for name, child in module.named_children():
            child_path = path + '.' + name if path else name
            attention_paths.extend(find_attention_layers(child, child_path))
        
        return attention_paths
    
    attention_paths = []
    for name, module in model.named_children():
        attention_paths.extend(find_attention_layers(module, name))
    
    for path in attention_paths:
        logger.info(f"  {path}")
    
    # 提取一个示例注意力层进行详细分析
    if attention_paths:
        logger.info("-" * 50)
        logger.info("示例注意力层详细分析:")
        
        # 获取第一个注意力层的路径
        example_path = attention_paths[0]
        parts = example_path.split('.')
        
        # 递归获取该模块
        current_module = model
        for part in parts:
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
            else:
                logger.warning(f"无法找到模块: {part} 在 {example_path}")
                break
        
        if 'attention' in current_module.__class__.__name__.lower():
            analyze_attention_layer(current_module, logger, "  ", detailed=True)
    
    logger.info("=" * 50)

def analyze_attention_layer(attention_module, logger, indent="", detailed=False):
    """
    分析自注意力层的内部结构
    
    Args:
        attention_module: 注意力模块
        logger: 日志记录器
        indent: 缩进字符串
        detailed: 是否打印详细信息
    """
    logger.info(f"{indent}注意力层分析 ({attention_module.__class__.__name__}):")
    
    # 打印子模块
    logger.info(f"{indent}- 子模块:")
    for name, child in attention_module.named_children():
        child_type = child.__class__.__name__
        child_params = sum(p.numel() for p in child.parameters())
        logger.info(f"{indent}  {name}: {child_type} ({child_params:,} 参数)")
    
    # 打印参数
    if detailed:
        logger.info(f"{indent}- 参数:")
        for name, param in attention_module.named_parameters():
            logger.info(f"{indent}  {name}: 形状={param.shape}, 类型={param.dtype}")
    
    # 尝试识别注意力层的关键组件
    logger.info(f"{indent}- 关键组件识别:")
    
    # 查找查询、键、值投影矩阵
    q_proj = None
    k_proj = None
    v_proj = None
    qkv_proj = None
    out_proj = None
    
    for name, child in attention_module.named_children():
        name_lower = name.lower()
        if 'q_proj' in name_lower or 'query' in name_lower:
            q_proj = (name, child)
        elif 'k_proj' in name_lower or 'key' in name_lower:
            k_proj = (name, child)
        elif 'v_proj' in name_lower or 'value' in name_lower:
            v_proj = (name, child)
        elif 'qkv' in name_lower:
            qkv_proj = (name, child)
        elif 'out_proj' in name_lower or 'output' in name_lower or 'o_proj' in name_lower:
            out_proj = (name, child)
    
    # 打印找到的组件
    if q_proj:
        logger.info(f"{indent}  查询投影: {q_proj[0]} ({q_proj[1].__class__.__name__})")
    if k_proj:
        logger.info(f"{indent}  键投影: {k_proj[0]} ({k_proj[1].__class__.__name__})")
    if v_proj:
        logger.info(f"{indent}  值投影: {v_proj[0]} ({v_proj[1].__class__.__name__})")
    if qkv_proj:
        logger.info(f"{indent}  合并QKV投影: {qkv_proj[0]} ({qkv_proj[1].__class__.__name__})")
    if out_proj:
        logger.info(f"{indent}  输出投影: {out_proj[0]} ({out_proj[1].__class__.__name__})")
    
    # 尝试识别注意力计算方法
    attention_fn = None
    for name, child in attention_module.named_children():
        if 'score' in name.lower() or 'compute' in name.lower() or 'attn' in name.lower():
            attention_fn = (name, child)
    
    if attention_fn:
        logger.info(f"{indent}  注意力计算: {attention_fn[0]} ({attention_fn[1].__class__.__name__})")
    
    # 尝试识别位置编码
    pos_encoding = None
    for name, child in attention_module.named_children():
        if 'pos' in name.lower() or 'embed' in name.lower() or 'rope' in name.lower() or 'rotary' in name.lower():
            pos_encoding = (name, child)
    
    if pos_encoding:
        logger.info(f"{indent}  位置编码: {pos_encoding[0]} ({pos_encoding[1].__class__.__name__})")
    
    # 打印注意力层的前向传播方法源代码（如果可用）
    if detailed and hasattr(attention_module, 'forward'):
        try:
            import inspect
            forward_code = inspect.getsource(attention_module.forward)
            logger.info(f"{indent}- 前向传播方法源代码:")
            for line in forward_code.split('\n'):
                logger.info(f"{indent}  {line}")
        except Exception as e:
            logger.info(f"{indent}  无法获取前向传播方法源代码: {str(e)}")
    
    # 提供替换建议
    logger.info(f"{indent}- 替换建议:")
    logger.info(f"{indent}  1. 确定替换点: 通常是在forward方法中的注意力计算部分")
    logger.info(f"{indent}  2. 保留输入输出接口: 确保替换后的自注意力层与原始层有相同的输入输出格式")
    logger.info(f"{indent}  3. 注意位置编码: 如果使用了RoPE等位置编码，需要在新的注意力机制中保留或适配")
    logger.info(f"{indent}  4. 替换方法: 可以继承原始注意力类并重写forward方法，或创建新的注意力类并在模型中替换")
    

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    logger = setup_logger(
        name="verify_model",
        log_dir=LOG_DIR,
        log_level="INFO",
        log_to_file=True,
        log_to_console=not args.no_console_output,
        log_file_suffix="_custom" if args.output_file else ""
    )
    
    # 如果指定了输出文件，则添加一个专门的文件处理器
    if args.output_file:
        # 创建格式化器
        formatter = logging.Formatter(
            '%(message)s',  # 只输出消息内容，不包含时间戳等信息
        )
        
        # 创建文件处理器
        file_handler = logging.FileHandler(args.output_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"开始验证模型: {args.model_path}")
    
    # 初始化硬件监控
    monitor = None
    if args.monitor:
        monitor = HardwareMonitor(interval=1.0, log_dir=LOG_DIR)
        monitor.start()
        logger.info("硬件监控已启动")
    
    try:
        # 加载模型和tokenizer
        logger.info("加载模型和tokenizer...")
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # 打印模型结构
        if args.print_structure:
            logger.info("打印模型结构...")
            print_model_structure(model, logger, args.attention_only)
        
        # 验证模型
        logger.info(f"验证模型，提示词: {args.prompt}")
        output = verify_model(model, tokenizer, args.prompt, args.max_new_tokens)
        
        logger.info(f"模型输出: {output}")
        logger.info("模型验证成功")
    
    except Exception as e:
        logger.error(f"模型验证失败: {str(e)}")
    
    finally:
        # 停止硬件监控
        if monitor:
            monitor.stop()
            monitor.save_to_csv()
            logger.info("硬件监控已停止")
    
    logger.info("验证完成")

if __name__ == "__main__":
    main() 