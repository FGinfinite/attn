#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 要配置的服务器列表
SERVERS=(
    "222.20.97.104"
    "222.20.97.138"
    "222.20.97.217"
    "222.20.97.235"
    "192.168.1.110"
)

# 默认用户名，可以根据需要修改
DEFAULT_USERNAME="lishiwei"
SSH_KEY_TYPE="ed25519" # 使用更现代的ed25519算法
SSH_KEY_PATH="$HOME/.ssh/id_${SSH_KEY_TYPE}"

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 检查是否已经存在SSH密钥
check_existing_keys() {
    if [ -f "$SSH_KEY_PATH" ]; then
        print_message "$YELLOW" "发现已存在的SSH密钥: $SSH_KEY_PATH"
        read -p "是否使用现有密钥? (y/n): " use_existing
        if [[ $use_existing =~ ^[Yy]$ ]]; then
            return 0
        else
            print_message "$YELLOW" "将生成新的SSH密钥..."
            return 1
        fi
    else
        print_message "$YELLOW" "未找到SSH密钥，将生成新的密钥..."
        return 1
    fi
}

# 生成新的SSH密钥
generate_ssh_key() {
    print_message "$GREEN" "正在生成新的SSH密钥..."
    ssh-keygen -t $SSH_KEY_TYPE -f "$SSH_KEY_PATH" -N ""
    if [ $? -eq 0 ]; then
        print_message "$GREEN" "SSH密钥生成成功！"
    else
        print_message "$RED" "SSH密钥生成失败！"
        exit 1
    fi
}

# 检查服务器上是否已经有我们的公钥
check_server_key() {
    local server=$1
    local username=$2
    
    print_message "$YELLOW" "检查服务器 $server 是否已有我们的公钥..."
    
    # 获取本地公钥内容
    local pubkey=$(cat "$SSH_KEY_PATH.pub")
    
    # 检查远程服务器上是否已有此公钥
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$username@$server" "grep -F \"$pubkey\" ~/.ssh/authorized_keys > /dev/null 2>&1"
    
    return $?
}

# 将公钥复制到远程服务器
copy_key_to_server() {
    local server=$1
    local username=$2
    
    print_message "$YELLOW" "正在将公钥复制到 $username@$server..."
    
    # 确保远程服务器上存在.ssh目录
    ssh -o BatchMode=no "$username@$server" "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
    
    # 复制公钥
    cat "$SSH_KEY_PATH.pub" | ssh "$username@$server" "cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    
    if [ $? -eq 0 ]; then
        print_message "$GREEN" "公钥已成功复制到 $username@$server"
        return 0
    else
        print_message "$RED" "复制公钥到 $username@$server 失败"
        return 1
    fi
}

# 测试SSH连接
test_ssh_connection() {
    local server=$1
    local username=$2
    
    print_message "$YELLOW" "测试SSH连接到 $username@$server..."
    
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$username@$server" "echo '连接成功！'"
    
    if [ $? -eq 0 ]; then
        print_message "$GREEN" "成功连接到 $username@$server"
        return 0
    else
        print_message "$RED" "连接到 $username@$server 失败"
        return 1
    fi
}

# 主函数
main() {
    print_message "$GREEN" "===== SSH密钥设置脚本 ====="
    
    # 检查并可能生成SSH密钥
    if ! check_existing_keys; then
        generate_ssh_key
    fi
    
    # 为每个服务器设置SSH密钥
    for server in "${SERVERS[@]}"; do
        read -p "请输入服务器 $server 的用户名 [默认: $DEFAULT_USERNAME]: " username
        username=${username:-$DEFAULT_USERNAME}
        
        # 检查服务器上是否已有我们的公钥
        if check_server_key "$server" "$username"; then
            print_message "$GREEN" "服务器 $server 已有我们的公钥"
        else
            print_message "$YELLOW" "服务器 $server 没有我们的公钥，正在复制..."
            copy_key_to_server "$server" "$username"
        fi
        
        # 测试连接
        test_ssh_connection "$server" "$username"
    done
    
    print_message "$GREEN" "===== SSH密钥设置完成 ====="
}

# 执行主函数
main 