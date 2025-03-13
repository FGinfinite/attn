# Attention Mechanism Comparison Project

This project is based on **Qwen2.5-3B-Instruct** and aims to compare various attention mechanisms in terms of performance, inference speed, resource usage, and other metrics.

## Project Structure

```
.
├── config.py                 # Configuration file
├── config.yaml               # YAML configuration file
├── main.py                   # Main entry point
├── init_project.py           # Project initialization script
├── requirements.txt          # Dependencies list
├── README.md                 # English README
├── README_CN.md              # Chinese README
├── scripts/                  # Scripts directory
│   ├── model/                # Model-related scripts
│   │   ├── verify_model.py   # Model verification script
│   │   ├── quantize_model.py # Model quantization script
│   │   ├── test_attention.py # Attention mechanism test script
│   │   └── test_vllm.py      # vLLM test script
│   ├── benchmark/            # Benchmark scripts
│   │   ├── run_benchmark.py  # Benchmark script
│   │   └── run_all_tests.py  # Automated testing script
│   └── analysis/             # Analysis scripts
│       └── analyze_results.py # Results analysis script
├── src/                      # Source code directory
│   ├── attention/            # Attention mechanism modules
│   ├── quantization/         # Quantization method modules
│   ├── utils/                # Utility modules
│   └── benchmark/            # Benchmark modules
├── data/                     # Data directory
│   └── results/              # Results directory
├── logs/                     # Logs directory
└── analysis/                 # Analysis output directory
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/attn.git
cd attn
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Initialize the project:

```bash
python main.py init
```

## Usage

The project provides a unified command-line interface through the `main.py` script:

### 1. Verify Model

```bash
python main.py verify --model_path Qwen/Qwen2.5-3B-Instruct --monitor
```

### 2. Quantize Model

```bash
# AWQ quantization
python main.py quantize --model_path Qwen/Qwen2.5-3B-Instruct --quant awq --monitor

# GPTQ quantization
python main.py quantize --model_path Qwen/Qwen2.5-3B-Instruct --quant gptq --monitor
```

### 3. Test Attention Mechanisms

```bash
# Standard attention
python main.py test_attention --model_path Qwen/Qwen2.5-3B-Instruct --attention standard --monitor

# Sparse attention
python main.py test_attention --model_path Qwen/Qwen2.5-3B-Instruct --attention sparse --sparsity 0.8 --monitor

# Linear attention
python main.py test_attention --model_path Qwen/Qwen2.5-3B-Instruct --attention linear --kernel_function elu --monitor
```

### 4. Use vLLM Acceleration

```bash
python main.py test_vllm --model_path Qwen/Qwen2.5-3B-Instruct --quant none --monitor
```

### 5. Run Benchmark

```bash
python main.py benchmark --model_path Qwen/Qwen2.5-3B-Instruct --quant none --attention standard --batch_size 16 --input_length 512 --output_length 128 --monitor --save_results
```

### 6. Run Automated Tests

```bash
python main.py auto_test --model_path Qwen/Qwen2.5-3B-Instruct --quant_types none --attention_types standard,sparse,linear --batch_sizes 1 --input_lengths 512,1024,2048 --output_lengths 128 --monitor --save_results
```

### 7. Analyze Results

```bash
python main.py analyze --results_dir data/results --output_dir analysis --metrics latency,tokens_per_second,memory_usage,perplexity
```

## Notes

1. Ensure you have sufficient GPU memory (at least 12GB)
2. AWQ quantization requires the autoawq library
3. GPTQ quantization requires the auto-gptq library
4. vLLM acceleration requires the vllm library
5. When using vLLM, AWQ requires vLLM>=0.3.3, and GPTQ requires the vllm-gptq extension

## Experimental Results

Experimental results will be saved in the `data/results` directory, including:
- Detailed results in JSON format
- Summary results in CSV format
- Hardware monitoring data

Analysis results will be saved in the `analysis` directory, including:
- Analysis report in Markdown format
- Visualization charts