#!/bin/bash

# 定义默认源文件夹
# DEFAULT_DIRS=("$HOME/attn/raw" "$HOME/attn/grads" "$HOME/attn/selected_data")
DEFAULT_DIRS=("$HOME/attn/raw")

# 定义目标服务器
TARGET_SERVERS=(
    "222.20.97.104"
    "222.20.97.138"
    "222.20.97.217"
    "222.20.97.235"
    "192.168.1.110"
)
TARGET_BASE_DIR="~/attn"

# 显示彩色输出
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
NC="\033[0m" # 无颜色

# 设置参数
MAX_PARALLEL_JOBS=10  # 最大并行任务数

# 捕获CTRL+C
trap 'echo -e "${RED}中断执行!${NC}"; exit 1' INT

# 处理命令行参数
if [ $# -eq 0 ]; then
    # 没有参数时使用默认文件夹
    SOURCE_DIRS=("${DEFAULT_DIRS[@]}")
else
    # 使用提供的参数作为源文件夹
    SOURCE_DIRS=()
    for dir in "$@"; do
        # 转换为绝对路径
        if [[ "$dir" != /* ]]; then
            dir="$HOME/$dir"
        fi
        SOURCE_DIRS+=("$dir")
    done
fi

# 检查源文件夹是否存在
for SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
    if [ ! -d "$SOURCE_DIR" ]; then
        echo -e "${RED}错误: 源文件夹 $SOURCE_DIR 不存在${NC}"
        exit 1
    fi
done

echo -e "${YELLOW}开始并行同步文件夹到多个目标服务器...${NC}"
echo "要同步的文件夹: ${SOURCE_DIRS[*]}"
echo "最大并行任务数: $MAX_PARALLEL_JOBS"
echo ""

# 函数：执行单个同步任务
sync_task() {
    local source_dir="$1"
    local server="$2"
    local dir_name=$(basename "$source_dir")
    local target_dir="$TARGET_BASE_DIR/$dir_name"
    local start_time=$(date +%s)
    
    # 尝试连接服务器检查是否可达
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$server" exit &> /dev/null; then
        echo -e "${RED}失败: $source_dir -> $server (无法连接到服务器)${NC}"
        return
    fi
    
    # 确保目标目录存在
    ssh "$server" "mkdir -p $target_dir" &> /dev/null
    
    # 使用rsync进行增量同步，捕获统计信息
    local rsync_output=$(rsync -avz --delete --stats --human-readable "$source_dir/" "$server:$target_dir/" 2>&1)
    local rsync_status=$?
    
    # 提取传输的数据量
    local transferred_data=$(echo "$rsync_output" | grep "Total transferred file size" | awk '{print $5, $6}')
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $rsync_status -eq 0 ]; then
        # 输出到控制台
        echo -e "${GREEN}成功: $source_dir -> $server (${transferred_data}, ${duration}秒)${NC}"
    else
        # 输出到控制台
        echo -e "${RED}失败: $source_dir -> $server (${duration}秒)${NC}"
    fi
}

# 计数器和作业数组
job_count=0
declare -a pids

# 对每个源文件夹和每个目标服务器执行同步，并行处理
for SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
    DIR_NAME=$(basename "$SOURCE_DIR")
    echo -e "${YELLOW}处理文件夹: $SOURCE_DIR${NC}"
    
    for server in "${TARGET_SERVERS[@]}"; do
        # 检查是否达到最大并行任务数
        if [ $job_count -ge $MAX_PARALLEL_JOBS ]; then
            # 等待一个任务完成
            wait -n
            ((job_count--))
        fi
        
        # 启动新的同步任务
        echo -e "${BLUE}启动任务: $SOURCE_DIR -> $server${NC}"
        sync_task "$SOURCE_DIR" "$server" &
        pids+=($!)
        ((job_count++))
    done
done

# 等待所有剩余任务完成
echo -e "${YELLOW}等待所有同步任务完成...${NC}"
wait

echo -e "${GREEN}所有同步过程完成!${NC}"
