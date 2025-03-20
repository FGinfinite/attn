#!/bin/bash
# 用于CMRC2018数据集处理和微调的脚本

# 设置变量
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"  # 模型路径，可以根据需要修改
OUTPUT_DIR="raw/finetuned-cmrc2018"  # 微调模型输出目录
DATA_DIR="raw/cmrc2018/finetune"  # 处理后数据的目录
MAX_SAMPLES=-1 # 每个数据集的最大样本数，-1表示使用所有样本
BATCH_SIZE=1  # 训练批次大小
LEARNING_RATE=2e-5  # 学习率
MAX_STEPS=200  # 最大训练步数
PRECISION="fp16"  # 训练精度，可选fp16, bf16, fp32
ATTENTION="standard"  # 注意力机制类型
INCLUDE_POSITION=true  # 是否在助手回复中包含答案位置信息，设置为true或false
USE_CHAR_POSITION=true # 是否使用字符级位置信息，false表示使用token级位置
TOKENIZER="Qwen/Qwen2.5-3B-Instruct"  # 用于计算token位置的tokenizer
USE_VALID=true  # 是否使用验证集
VALID_RATIO=0.1  # 从训练集中分割出的验证集比例（0.1表示10%）

# 创建必要的目录
mkdir -p ${DATA_DIR}
mkdir -p ${OUTPUT_DIR}

# 第1步：处理CMRC2018数据集
echo "第1步：处理CMRC2018数据集..."
POSITION_ARGS=""
if [ "$INCLUDE_POSITION" = true ]; then
  POSITION_ARGS="--include_position"
  echo "将在回复中包含答案位置信息"
  
  if [ "$USE_CHAR_POSITION" = true ]; then
    POSITION_ARGS="${POSITION_ARGS} --use_char_position"
    echo "使用字符级位置信息"
  else
    echo "使用token级位置信息，tokenizer: ${TOKENIZER}"
    POSITION_ARGS="${POSITION_ARGS} --tokenizer ${TOKENIZER}"
  fi
fi

# 设置验证集参数
VALID_ARGS=""
if [ "$USE_VALID" = true ]; then
  VALID_ARGS="--split_valid --valid_ratio ${VALID_RATIO}"
  echo "将从训练集中分割 ${VALID_RATIO} 比例的数据作为验证集"
fi

# 注意：更新了脚本路径，从scripts/data目录运行
python scripts/data/prepare_cmrc2018.py \
  --input_dir raw/cmrc2018/squad-style-data \
  --output_dir ${DATA_DIR} \
  --max_samples ${MAX_SAMPLES} \
  --train_file cmrc2018_train.json \
  --test_file cmrc2018_dev.json \
  ${POSITION_ARGS} \
  ${VALID_ARGS}

# 第2步：执行模型微调
echo "第2步：执行模型微调..."
# 需要回到项目根目录才能正确运行main.py
cd $(dirname $(dirname $(dirname $0)))

# 根据是否使用验证集选择微调命令参数
TRAIN_ARGS=""
if [ "$USE_VALID" = true ]; then
  TRAIN_ARGS="--dataset_path ${DATA_DIR}/cmrc2018_train.json --valid_dataset_path ${DATA_DIR}/cmrc2018_valid.json"
  echo "使用训练集和验证集进行微调"
else
  TRAIN_ARGS="--dataset_path ${DATA_DIR}/cmrc2018_train.json"
  echo "仅使用训练集进行微调，无验证集"
fi

python main.py finetune \
  --model_path ${MODEL_PATH} \
  ${TRAIN_ARGS} \
  --output_dir ${OUTPUT_DIR} \
  --precision ${PRECISION} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --max_steps ${MAX_STEPS} \
  --attention ${ATTENTION}

# 第3步：测试微调后的模型
echo "第3步：测试微调后的模型..."
python main.py test_finetune \
  --base_model ${MODEL_PATH} \
  --finetuned_model ${OUTPUT_DIR}/final \
  --dataset_path ${DATA_DIR}/cmrc2018_test.json \
  --num_samples 5 \
  --output_file cmrc2018_test_results.txt \
  --precision ${PRECISION}

echo "CMRC2018微调完成！" 