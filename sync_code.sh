#!/bin/bash

# 定义颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 定义服务器列表
SERVERS=(
    "222.20.97.104"
    "222.20.97.138"
    "222.20.97.217"
    "222.20.97.235"
    "192.168.1.110"
)

# 目标目录
TARGET_DIR="~/attn"

# 打印带颜色的消息函数
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 显示使用帮助
show_help() {
    echo "用法: $0 [选项]"
    echo
    echo "自动同步多个服务器的代码仓库"
    echo
    echo "选项:"
    echo "  -h, --help     显示帮助信息"
    echo "  -i, --interactive  交互模式，遇到冲突时提供手动处理选项"
    echo "  -f, --force    强制模式，使用git reset --hard origin/main解决冲突"
    echo "  -s, --server SERVER  指定单个服务器进行同步"
    echo
}

# 默认参数
INTERACTIVE=true
FORCE=false
SPECIFIC_SERVER=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            INTERACTIVE=false
            shift
            ;;
        -s|--server)
            SPECIFIC_SERVER="$2"
            shift
            shift
            ;;
        *)
            print_message "$RED" "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果指定了特定服务器，则更新服务器列表
if [[ -n "$SPECIFIC_SERVER" ]]; then
    SERVERS=("$SPECIFIC_SERVER")
fi

# 同步单个服务器的函数
sync_server() {
    local server=$1
    print_message "$BLUE" "========================================"
    print_message "$YELLOW" "正在连接服务器: $server"
    
    # 检查服务器是否可达
    ssh -o ConnectTimeout=3 -o BatchMode=yes $server exit &>/dev/null
    if [[ $? -ne 0 ]]; then
        print_message "$RED" "无法连接到服务器 $server，跳过"
        return 1
    fi
    
    print_message "$GREEN" "连接成功，开始同步代码..."
    
    # 进入目标目录并执行git pull
    local output=$(ssh $server "cd $TARGET_DIR && git pull 2>&1")
    local status=$?
    
    echo "$output"
    
    # 检查是否存在冲突或其他错误
    if [[ $status -ne 0 ]] || [[ "$output" == *"CONFLICT"* ]] || [[ "$output" == *"error"* ]]; then
        print_message "$RED" "同步遇到问题！"
        
        if [[ "$FORCE" == true ]]; then
            print_message "$YELLOW" "正在强制重置代码..."
            ssh $server "cd $TARGET_DIR && git reset --hard origin/main"
            if [[ $? -eq 0 ]]; then
                print_message "$GREEN" "强制重置成功"
            else
                print_message "$RED" "强制重置失败"
            fi
        elif [[ "$INTERACTIVE" == true ]]; then
            print_message "$YELLOW" "如何处理这个问题？"
            print_message "$YELLOW" "1) 跳过这个服务器"
            print_message "$YELLOW" "2) 尝试强制重置 (git reset --hard origin/main)"
            print_message "$YELLOW" "3) 登录到服务器手动处理"
            
            read -p "请选择操作 [1/2/3]: " choice
            case $choice in
                1)
                    print_message "$BLUE" "跳过服务器 $server"
                    ;;
                2)
                    print_message "$YELLOW" "正在强制重置代码..."
                    ssh $server "cd $TARGET_DIR && git reset --hard origin/main"
                    if [[ $? -eq 0 ]]; then
                        print_message "$GREEN" "强制重置成功"
                    else
                        print_message "$RED" "强制重置失败"
                    fi
                    ;;
                3)
                    print_message "$BLUE" "正在连接到服务器，请手动处理问题..."
                    print_message "$YELLOW" "执行后请输入 'exit' 返回同步脚本"
                    ssh -t $server "cd $TARGET_DIR && bash"
                    ;;
                *)
                    print_message "$RED" "无效选择，跳过服务器 $server"
                    ;;
            esac
        else
            print_message "$RED" "跳过服务器 $server"
        fi
    else
        print_message "$GREEN" "同步成功！"
    fi
    
    print_message "$BLUE" "========================================"
    echo ""
}

# 主函数
main() {
    print_message "$BLUE" "开始同步代码到多台服务器"
    print_message "$BLUE" "目标目录: $TARGET_DIR"
    echo ""
    
    for server in "${SERVERS[@]}"; do
        sync_server "$server"
    done
    
    print_message "$GREEN" "同步操作完成！"
}

# 执行主函数
main 