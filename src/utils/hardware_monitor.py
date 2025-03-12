"""
硬件监控模块，用于监控GPU使用情况
"""

import os
import time
import logging
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# 添加psutil支持CPU监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger("attn_experiment")

class HardwareMonitor:
    """硬件监控类，用于监控GPU使用情况和模型性能指标"""
    
    def __init__(self, interval=1.0, log_dir=None):
        """
        初始化硬件监控器
        
        Args:
            interval: 监控间隔（秒）
            log_dir: 日志目录
        """
        self.interval = interval
        self.log_dir = log_dir
        self.running = False
        self.thread = None
        self.metrics = defaultdict(list)
        self.start_time = None
        
        # 初始化NVML
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.device_count = nvml.nvmlDeviceGetCount()
                logger.info(f"NVML初始化成功，检测到{self.device_count}个GPU设备")
            except Exception as e:
                logger.error(f"NVML初始化失败: {str(e)}")
                self.device_count = 0
        else:
            logger.warning("未安装py3nvml库，无法监控GPU")
            self.device_count = 0
    
    def start(self):
        """开始监控"""
        if self.running:
            logger.warning("监控器已在运行")
            return
        
        self.running = True
        self.start_time = time.time()
        self.metrics.clear()
        
        # 创建并启动监控线程
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("硬件监控器已启动")
    
    def stop(self):
        """停止监控"""
        if not self.running:
            logger.warning("监控器未在运行")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2*self.interval)
        
        logger.info("硬件监控器已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 记录时间戳
                timestamp = time.time() - self.start_time
                self.metrics["timestamp"].append(timestamp)
                
                # 监控CPU使用率
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent()
                    self.metrics["cpu_percent"].append(cpu_percent)
                    
                    # 内存使用率
                    memory_info = psutil.virtual_memory()
                    self.metrics["memory_percent"].append(memory_info.percent)
                
                # 遍历所有GPU设备
                for i in range(self.device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # 获取GPU利用率
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                    self.metrics[f"gpu{i}_util"].append(utilization.gpu)
                    # 保存为gpu_load(与用户当前看到的键名保持一致)
                    self.metrics["gpu_load"].append(utilization.gpu)
                    
                    # 获取GPU内存使用情况
                    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used_mb = memory.used / 1024 / 1024
                    memory_total_mb = memory.total / 1024 / 1024
                    memory_percent = 100 * memory.used / memory.total
                    
                    self.metrics[f"gpu{i}_mem_used_mb"].append(memory_used_mb)
                    self.metrics[f"gpu{i}_mem_total_mb"].append(memory_total_mb)
                    self.metrics[f"gpu{i}_mem_percent"].append(memory_percent)
                    # 保存为gpu_memory_percent(与用户当前看到的键名保持一致)
                    self.metrics["gpu_memory_percent"].append(memory_percent)
                    
                    # 获取GPU温度
                    temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    self.metrics[f"gpu{i}_temp"].append(temperature)
                    
                    # 获取GPU功耗
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                    self.metrics[f"gpu{i}_power_w"].append(power)
            
            except Exception as e:
                logger.error(f"监控过程中出错: {str(e)}")
            
            # 等待下一个监控周期
            time.sleep(self.interval)
    
    def add_model_metric(self, metric_name, value):
        """
        添加模型性能指标
        
        Args:
            metric_name: 指标名称（如latency、tokens_per_second、perplexity）
            value: 指标值
        """
        if self.running:
            self.metrics[metric_name].append(value)
            logger.debug(f"添加模型指标: {metric_name}={value}")
        else:
            logger.warning(f"监控器未运行，无法添加指标: {metric_name}={value}")
    
    def get_metrics(self):
        """获取监控指标"""
        return dict(self.metrics)
    
    def get_summary(self):
        """获取监控摘要"""
        summary = {}
        
        for key, values in self.metrics.items():
            if key == "timestamp":
                continue
            
            if len(values) > 0:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_max"] = np.max(values)
                summary[f"{key}_min"] = np.min(values)
        
        return summary
    
    def save_to_csv(self, filename=None):
        """
        保存监控数据到CSV文件
        
        Args:
            filename: 文件名，如果为None则自动生成
            
        Returns:
            filepath: 保存的文件路径
        """
        if not self.metrics:
            logger.warning("没有监控数据可保存")
            return None
        
        if self.log_dir is None:
            logger.warning("未指定日志目录，无法保存监控数据")
            return None
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gpu_metrics_{timestamp}.csv"
        
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            import pandas as pd
            df = pd.DataFrame(self.metrics)
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"监控数据已保存到: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"保存监控数据失败: {str(e)}")
            return None
    
    def __del__(self):
        """析构函数"""
        self.stop()
        if NVML_AVAILABLE:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass 