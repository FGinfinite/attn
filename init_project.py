#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目初始化脚本
用于创建必要的目录结构和初始化配置
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import json
import yaml

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="初始化项目目录和配置")
    parser.add_argument("--force", action="store_true", help="强制重新初始化项目")
    return parser.parse_args()

def create_directory_structure():
    """创建项目目录结构"""
    logger.info("创建项目目录结构...")
    
    # 创建主要目录
    directories = [
        "src/attention",
        "src/quantization",
        "src/utils",
        "src/benchmark",
        "data/results",
        "logs",
        "analysis"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")
    
    # 创建空的__init__.py文件
    init_files = [
        "src/__init__.py",
        "src/attention/__init__.py",
        "src/quantization/__init__.py",
        "src/utils/__init__.py",
        "src/benchmark/__init__.py"
    ]
    
    for init_file in init_files:
        if not Path(init_file).exists():
            Path(init_file).touch()
            logger.info(f"创建文件: {init_file}")

def create_config_file():
    """创建配置文件"""
    logger.info("创建配置文件...")
    
    config = {
        "model": {
            "default_model_path": "Qwen/Qwen2.5-3B-Instruct",
            "supported_models": ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
        },
        "quantization": {
            "supported_types": ["none", "awq", "gptq"],
            "awq": {
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "version": "gemm"
            },
            "gptq": {
                "bits": 4,
                "group_size": 128,
                "desc_act": False
            }
        },
        "attention": {
            "supported_types": ["standard", "sparse", "linear"],
            "sparse": {
                "default_sparsity": 0.8,
                "sparsity_range": [0.5, 0.9]
            },
            "linear": {
                "default_kernel": "elu",
                "supported_kernels": ["elu", "relu", "gelu"]
            }
        },
        "benchmark": {
            "default_batch_size": 1,
            "default_input_length": 512,
            "default_output_length": 128,
            "default_num_runs": 5,
            "default_warmup_runs": 2
        },
        "logging": {
            "log_dir": "logs",
            "results_dir": "data/results",
            "analysis_dir": "analysis"
        }
    }
    
    # 保存为YAML格式
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info("配置文件已创建: config.yaml")
    
    # 同时创建Python配置文件
    with open("config.py", "w", encoding="utf-8") as f:
        f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
配置文件
包含项目的所有配置参数
\"\"\"

import os
import yaml
from pathlib import Path

# 加载YAML配置
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# 模型配置
DEFAULT_MODEL_PATH = CONFIG["model"]["default_model_path"]
SUPPORTED_MODELS = CONFIG["model"]["supported_models"]

# 量化配置
SUPPORTED_QUANT_TYPES = CONFIG["quantization"]["supported_types"]
AWQ_CONFIG = CONFIG["quantization"]["awq"]
GPTQ_CONFIG = CONFIG["quantization"]["gptq"]

# 注意力机制配置
SUPPORTED_ATTENTION_TYPES = CONFIG["attention"]["supported_types"]
SPARSE_ATTENTION_CONFIG = CONFIG["attention"]["sparse"]
LINEAR_ATTENTION_CONFIG = CONFIG["attention"]["linear"]

# 基准测试配置
DEFAULT_BATCH_SIZE = CONFIG["benchmark"]["default_batch_size"]
DEFAULT_INPUT_LENGTH = CONFIG["benchmark"]["default_input_length"]
DEFAULT_OUTPUT_LENGTH = CONFIG["benchmark"]["default_output_length"]
DEFAULT_NUM_RUNS = CONFIG["benchmark"]["default_num_runs"]
DEFAULT_WARMUP_RUNS = CONFIG["benchmark"]["default_warmup_runs"]

# 日志配置
LOG_DIR = CONFIG["logging"]["log_dir"]
RESULTS_DIR = CONFIG["logging"]["results_dir"]
ANALYSIS_DIR = CONFIG["logging"]["analysis_dir"]
""")
    logger.info("Python配置文件已创建: config.py")

def create_gitignore():
    """创建.gitignore文件"""
    logger.info("创建.gitignore文件...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
ENV/
env/

# 日志和结果
logs/
data/results/
analysis/

# 模型文件
*.bin
*.pt
*.pth
*.safetensors
*.gguf

# 编辑器
.idea/
.vscode/
*.swp
*.swo

# 操作系统
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    logger.info(".gitignore文件已创建")

def main():
    """主函数"""
    args = parse_args()
    
    # 检查项目是否已初始化
    if Path("config.yaml").exists() and not args.force:
        logger.warning("项目似乎已经初始化。如需重新初始化，请使用 --force 参数。")
        return
    
    logger.info("开始初始化项目...")
    
    # 创建目录结构
    create_directory_structure()
    
    # 创建配置文件
    create_config_file()
    
    # 创建.gitignore
    create_gitignore()
    
    logger.info("项目初始化完成！")
    logger.info("请运行 'pip install -r requirements.txt' 安装依赖。")

if __name__ == "__main__":
    main() 