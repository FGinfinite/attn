#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess
import re
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# 定义目标服务器
TARGET_SERVERS = [
    "222.20.97.104",
    "222.20.97.138",
    "222.20.97.217",
    "222.20.97.235",
    "192.168.1.110"
]

# 定义彩色输出的ANSI转义码
class Colors:
    BLUE = "\033[0;34m"
    GREEN = "\033[0;32m" 
    RED = "\033[0;31m"
    YELLOW = "\033[0;33m"
    GRAY = "\033[0;90m"
    WHITE = "\033[0;37m"
    NC = "\033[0m"  # 无颜色

# 环境名称
ENV_NAME = "moe"

# 监控间隔时间（秒）
INTERVAL = 2

# 创建一个字典，用于存储每个服务器的最新状态
server_status = {server: "正在获取数据..." for server in TARGET_SERVERS}
# 线程锁，用于同步对server_status的访问
status_lock = threading.Lock()

def clear_screen():
    """清除终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """打印监控系统的标题和基本信息"""
    clear_screen()
    print(f"{Colors.BLUE}======================================================{Colors.NC}")
    print(f"{Colors.GREEN}          GPU 监控系统 - 实时状态查看             {Colors.NC}")
    print(f"{Colors.BLUE}======================================================{Colors.NC}")
    print(f"{Colors.YELLOW}环境: {ENV_NAME}  |  刷新间隔: {INTERVAL}秒{Colors.NC}")
    print(f"{Colors.BLUE}======================================================{Colors.NC}")
    print("")

def colorize_gpustat_output(output):
    """对gpustat输出应用彩色格式化"""
    if not output:
        return output
    
    lines = output.split('\n')
    colored_lines = []
    
    # 第一行是主机名和日期信息，保持原样
    if lines:
        colored_lines.append(lines[0])
    
    # 处理剩余行
    for line in lines[1:]:
        if not line.strip():
            colored_lines.append(line)
            continue
        
        # 1. 匹配并着色序号与GPU型号
        line = re.sub(r'(\[\d+\] NVIDIA [^|]+)', f'{Colors.BLUE}\\1{Colors.NC}', line)
        
        # 2. 匹配并着色温度
        line = re.sub(r'(\d+)\'C', f'{Colors.RED}\\1\'C{Colors.NC}', line)
        
        # 3. 匹配并着色使用率百分比
        line = re.sub(r'(\d+ %)', f'{Colors.GREEN}\\1{Colors.NC}', line)
        
        # 4. 匹配并着色显存使用量
        line = re.sub(r'(\d+) / (\d+) MB', f'{Colors.YELLOW}\\1{Colors.NC} / {Colors.YELLOW}\\2{Colors.NC} MB', line)
        
        # 5. 匹配并着色用户名和显存使用
        def color_user_memory(match):
            username = match.group(1)
            memory = match.group(2)
            return f"{Colors.GRAY}{username}{Colors.NC}({Colors.YELLOW}{memory}{Colors.NC})"
        
        line = re.sub(r'([a-zA-Z0-9_]+)\((\d+M)\)', color_user_memory, line)
        
        colored_lines.append(line)
    
    return '\n'.join(colored_lines)

def run_ssh_command(server, command):
    """在指定服务器上通过SSH执行命令"""
    try:
        # 检查服务器连接
        subprocess.run(["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes", server, "exit"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return f"{Colors.RED}无法连接到服务器 {server}{Colors.NC}"
    
    try:
        # 执行实际命令
        result = subprocess.run(["ssh", server, f"source ~/.bashrc && {command}"], 
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               universal_newlines=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"{Colors.RED}执行命令失败: {e.stderr}{Colors.NC}"

def monitor_server(server):
    """监控单个服务器的GPU状态并更新全局状态字典"""
    # 执行gpustat命令
    result = run_ssh_command(server, "gpustat")
    
    # 如果是错误消息，直接保存
    if result.startswith(Colors.RED):
        status = result
    else:
        # 对gpustat输出应用颜色
        status = f"{Colors.YELLOW}服务器: {server}{Colors.NC}\n{colorize_gpustat_output(result)}\n"
    
    # 使用线程锁更新状态字典
    with status_lock:
        server_status[server] = status

def print_all_status():
    """打印所有服务器的当前状态"""
    print_header()
    
    with status_lock:
        for server in TARGET_SERVERS:
            print(server_status[server])
    
    print(f"{Colors.BLUE}======================================================{Colors.NC}")
    print(f"{Colors.YELLOW}按 Ctrl+C 退出监控{Colors.NC}")

def update_servers_status():
    """并行更新所有服务器的状态"""
    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=len(TARGET_SERVERS)) as executor:
        # 为每个服务器创建一个任务
        futures = {executor.submit(monitor_server, server): server for server in TARGET_SERVERS}
        
    # 所有线程完成后，更新显示
    print_all_status()

def main():
    """主程序循环"""
    try:
        while True:
            # 并行获取所有服务器状态
            update_servers_status()
            
            # 等待指定的间隔时间
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n监控已停止")
        sys.exit(0)

if __name__ == "__main__":
    main() 