# 注意力机制对比实验项目

本项目基于 **Qwen2.5-3B-Instruct**，旨在对比多种注意力机制在性能、推理速度、资源占用等方面的表现。

## 项目结构

```
.
├── config.py                 # 配置文件
├── config.yaml               # YAML配置文件
├── main.py                   # 主入口文件
├── init_project.py           # 项目初始化脚本
├── requirements.txt          # 依赖库列表
├── README.md                 # 英文说明文档
├── README_CN.md              # 中文说明文档
├── scripts/                  # 脚本目录
│   ├── model/                # 模型相关脚本
│   │   ├── verify_model.py   # 模型验证脚本
│   │   ├── quantize_model.py # 模型量化脚本
│   │   ├── test_attention.py # 注意力机制测试脚本
│   │   └── test_vllm.py      # vLLM测试脚本
│   ├── benchmark/            # 基准测试脚本
│   │   ├── run_benchmark.py  # 基准测试脚本
│   │   └── run_all_tests.py  # 自动化测试脚本
│   └── analysis/             # 分析脚本
│       └── analyze_results.py # 结果分析脚本
├── src/                      # 源代码目录
│   ├── attention/            # 注意力机制模块
│   ├── quantization/         # 量化方法模块
│   ├── utils/                # 工具模块
│   └── benchmark/            # 基准测试模块
├── data/                     # 数据目录
│   └── results/              # 结果目录
├── logs/                     # 日志目录
└── analysis/                 # 分析输出目录
```

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/attn.git
cd attn
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 初始化项目：

```bash
python main.py init
```

## 使用方法

本项目通过`main.py`脚本提供统一的命令行接口：

### 1. 验证模型

```bash
python main.py verify --model_path Qwen/Qwen2.5-3B-Instruct --monitor
```

### 2. 量化模型

```bash
# AWQ量化
python main.py quantize --model_path Qwen/Qwen2.5-3B-Instruct --quant awq --monitor

# GPTQ量化
python main.py quantize --model_path Qwen/Qwen2.5-3B-Instruct --quant gptq --monitor
```

### 3. 测试注意力机制

```bash
# 标准注意力
python main.py test_attention --model_path Qwen/Qwen2.5-3B-Instruct --attention standard --monitor

# 稀疏注意力
python main.py test_attention --model_path Qwen/Qwen2.5-3B-Instruct --attention sparse --sparsity 0.8 --monitor

# 线性注意力
python main.py test_attention --model_path Qwen/Qwen2.5-3B-Instruct --attention linear --kernel_function elu --monitor
```

### 4. 使用vLLM加速

```bash
python main.py test_vllm --model_path Qwen/Qwen2.5-3B-Instruct --quant none --monitor
```

### 5. 运行基准测试

```bash
python main.py benchmark --model_path Qwen/Qwen2.5-3B-Instruct --quant none --attention standard --batch_size 16 --input_length 512 --output_length 128 --monitor --save_results
```

### 6. 运行自动化测试

```bash
python main.py auto_test --model_path Qwen/Qwen2.5-3B-Instruct --quant_types none --attention_types standard,sparse,linear --batch_sizes 32 --input_lengths 512,1024,2048 --output_lengths 128 --monitor --save_results
```

### 7. 分析结果

```bash
python main.py analyze --results_dir data/results --output_dir analysis --metrics latency,tokens_per_second,memory_usage,perplexity
```

## 注意事项

1. 确保有足够的GPU显存（至少12GB）
2. AWQ量化需要安装autoawq库
3. GPTQ量化需要安装auto-gptq库
4. vLLM加速需要安装vllm库
5. 使用vLLM时，AWQ需要vLLM>=0.3.3，GPTQ需要安装vllm-gptq扩展包

## 实验结果

实验结果将保存在`data/results`目录中，包括：
- JSON格式的详细结果
- CSV格式的摘要结果
- 硬件监控数据

分析结果将保存在`analysis`目录中，包括：
- Markdown格式的分析报告
- 可视化图表 