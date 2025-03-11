# Attention Mechanism Comparison Project

This project is based on **Qwen2.5-3B-Instruct** and aims to compare various attention mechanisms in terms of performance, inference speed, resource usage, and other metrics.

## Project Structure

```
.
├── config.py                 # Configuration file
├── main.py                   # Main entry point
├── verify_model.py           # Model verification script
├── quantize_model.py         # Model quantization script
├── test_attention.py         # Attention mechanism test script
├── test_vllm.py              # vLLM test script
├── run_benchmark.py          # Benchmark script
├── run_all_tests.py          # Automated testing script
├── analyze_results.py        # Results analysis script
├── requirements.txt          # Dependencies list
├── src/                      # Source code directory
│   ├── attention/            # Attention mechanism modules
│   ├── quantization/         # Quantization method modules
│   ├── utils/                # Utility modules
│   └── benchmark/            # Benchmark modules
├── data/                     # Data directory
│   └── results/              # Results directory
└── logs/                     # Logs directory
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
python init_project.py
```

## Usage

### 1. Verify Model

```bash
python verify_model.py --model_path Qwen/Qwen2.5-3B-Instruct --monitor
```

### 2. Quantize Model

```bash
# AWQ quantization
python quantize_model.py --model_path Qwen/Qwen2.5-3B-Instruct --quant awq --monitor

# GPTQ quantization
python quantize_model.py --model_path Qwen/Qwen2.5-3B-Instruct --quant gptq --monitor
```

### 3. Test Attention Mechanisms

```bash
# Standard attention
python test_attention.py --model_path Qwen/Qwen2.5-3B-Instruct --attention standard --monitor

# Sparse attention
python test_attention.py --model_path Qwen/Qwen2.5-3B-Instruct --attention sparse --sparsity 0.8 --monitor

# Linear attention
python test_attention.py --model_path Qwen/Qwen2.5-3B-Instruct --attention linear --kernel_function elu --monitor
```

### 4. Use vLLM Acceleration

```bash
python test_vllm.py --model_path Qwen/Qwen2.5-3B-Instruct --quant none --monitor
```

### 5. Run Benchmark

```bash
python run_benchmark.py --model_path Qwen/Qwen2.5-3B-Instruct --quant none --attention standard --batch_size 1 --input_length 512 --output_length 128 --monitor --save_results
```

### 6. Run Automated Tests

```bash
python run_all_tests.py --model_path Qwen/Qwen2.5-3B-Instruct --quant_types none,awq,gptq --attention_types standard,sparse,linear --batch_sizes 1 --input_lengths 512,1024,2048 --output_lengths 128 --monitor --save_results
```

### 7. Analyze Results

```bash
python analyze_results.py --results_dir data/results --output_dir analysis --metrics latency,tokens_per_second,memory_usage,perplexity
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