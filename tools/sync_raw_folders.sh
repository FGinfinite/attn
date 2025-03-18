#!/bin/bash

# 定义源文件夹和目标服务器
SOURCE_DIR="$HOME/attn/raw"
TARGET_SERVERS=(
    "222.20.97.104"
    "222.20.97.138"
    "222.20.97.217"
    "222.20.97.235"
    "192.168.1.110"
)
TARGET_DIR="~/attn/raw"

# 显示彩色输出
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m" # 无颜色

# 检查源文件夹是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}错误: 源文件夹 $SOURCE_DIR 不存在${NC}"
    exit 1
fi

echo -e "${YELLOW}开始同步 $SOURCE_DIR 到多个目标服务器...${NC}"

# 对每个目标服务器执行同步
for server in "${TARGET_SERVERS[@]}"; do
    echo -e "${YELLOW}正在同步到服务器 $server...${NC}"
    
    # 首先尝试连接服务器检查是否可达
    ssh -o ConnectTimeout=5 -o BatchMode=yes $server exit &> /dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}无法连接到服务器 $server，跳过此服务器${NC}"
        continue
    fi
    
    # 确保目标目录存在
    ssh $server "mkdir -p $TARGET_DIR" &> /dev/null
    
    # 使用rsync进行增量同步，添加了一些常用选项:
    # -a: 归档模式，保留所有文件属性
    # -v: 详细输出
    # -z: 压缩传输
    # --delete: 删除目标中源中不存在的文件
    # --progress: 显示传输进度
    rsync -avz --delete --progress "$SOURCE_DIR/" "$server:$TARGET_DIR/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}成功同步到服务器 $server${NC}"
    else
        echo -e "${RED}同步到服务器 $server 时出错${NC}"
    fi
    
    echo ""
done

echo -e "${GREEN}同步过程完成!${NC}" 