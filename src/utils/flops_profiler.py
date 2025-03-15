#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLOPs分析器模块，基于DeepSpeed的Flops Profiler
用于统计模型在运行过程中的FLOPs、参数量等指标
"""

import logging
import torch
import numpy as np
from typing import Dict, Optional, List, Union, Tuple
import io
import sys
import os
from datetime import datetime
import warnings

# 导入DeepSpeed的Flops Profiler
try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

logger = logging.getLogger("attn_experiment")

# 创建自定义LoggerWriter类将print输出重定向到logger
class LoggerWriter:
    def __init__(self, logger):
        self.logger = logger
        self.buffer = ""
    
    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                if line.strip():  # 跳过空行
                    self.logger.info(line)
            self.buffer = lines[-1]
    
    def flush(self):
        if self.buffer:
            self.logger.info(self.buffer)
            self.buffer = ""

def create_file_only_logger(name, log_file_path):
    """
    创建一个只输出到文件的日志记录器
    
    Args:
        name: 日志记录器名称
        log_file_path: 日志文件路径
        
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志记录器
    file_logger = logging.getLogger(name)
    file_logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    for handler in file_logger.handlers[:]:
        file_logger.removeHandler(handler)
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 设置格式化器
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    # 将处理器添加到日志记录器
    file_logger.addHandler(file_handler)
    
    # 设置propagate=False，确保日志不会传递到父记录器
    file_logger.propagate = False
    
    return file_logger

class FlopsProfilerWrapper:
    """
    DeepSpeed Flops Profiler的封装类，用于计算模型的FLOPs和参数量
    """
    
    def __init__(self, hardware_monitor=None):
        """
        初始化FLOPs分析器
        
        Args:
            hardware_monitor: 硬件监控器实例，用于记录分析结果
        """
        self.hardware_monitor = hardware_monitor
        
        if not DEEPSPEED_AVAILABLE:
            logger.warning("未安装DeepSpeed库，无法使用Flops Profiler功能")
            self.available = False
        else:
            logger.info("成功加载DeepSpeed Flops Profiler")
            self.available = True
            
        self.flops_profiler = None
    
    def start_profiling(self, model):
        """
        开始动态FLOPs分析（在模型运行过程中分析）
        
        Args:
            model: 要分析的模型
        """
        if not self.available:
            logger.warning("DeepSpeed Flops Profiler不可用")
            return
        
        try:
            # 初始化分析器
            self.flops_profiler = FlopsProfiler(model)
            # 开始分析
            self.flops_profiler.start_profile()
            logger.info("动态FLOPs分析已开始")
        except Exception as e:
            logger.error(f"启动FLOPs分析出错: {str(e)}")
            self.flops_profiler = None
    
    def stop_profiling(self, print_results=True):
        """
        停止动态FLOPs分析并获取结果
        
        Args:
            print_results: 是否直接打印结果到控制台
            
        Returns:
            dict: 分析结果，包含详细的每层统计信息
        """
        if not self.available or self.flops_profiler is None:
            logger.warning("DeepSpeed Flops Profiler不可用或未启动分析")
            return None
        
        try:
            # 停止分析
            self.flops_profiler.stop_profile()
            
            # 获取统计信息
            try:
                flops = self.flops_profiler.get_total_flops()
                macs = self.flops_profiler.get_total_macs()
                params = self.flops_profiler.get_total_params()
                forward_elapsed_time = self.flops_profiler.get_total_duration()
                flops_per_second = flops / forward_elapsed_time
            except AttributeError as ae:
                # 处理 '__flops__' 属性不存在的情况
                logger.warning(f"获取FLOPs统计数据时出现属性错误: {str(ae)}")
                logger.warning("可能是模型结构与DeepSpeed Flops Profiler不完全兼容")
                # 设置默认值
                flops = 0
                macs = 0
                params = 0
                forward_elapsed_time = 0.001  # 避免除以零
                flops_per_second = 0
            
            # 获取详细的每层统计信息
            detailed_results = {}
            try:
                # 基本信息输出到常规日志（简短信息，显示在控制台）
                logger.info("FLOPs分析完成：")
                logger.info(f"总FLOPs: {self._format_count(flops)}, 总MACs: {self._format_count(macs)}")
                logger.info(f"总参数量: {self._format_count(params)}, 前向传播时间: {forward_elapsed_time:.4f}s")
                logger.info(f"每秒FLOPs: {self._format_count(flops_per_second)}/s")
                
                # 创建一个只输出到文件的特殊日志记录器用于详细信息
                # 使用固定的日志文件名，而不是每次都创建新的
                flops_log_file = os.path.join("logs", "flops_analysis.log")
                
                # 确保logs目录存在
                os.makedirs(os.path.dirname(flops_log_file), exist_ok=True)
                
                # 检查文件是否存在，如果不存在则创建并写入头部信息
                file_exists = os.path.exists(flops_log_file)
                flops_logger = create_file_only_logger("flops_analysis", flops_log_file)
                
                # 如果是新文件，写入头部信息
                if not file_exists:
                    flops_logger.info("=" * 80)
                    flops_logger.info("FLOPs分析日志 - 所有训练步骤的分析结果")
                    flops_logger.info("=" * 80)
                
                # 添加时间戳分隔每次分析结果
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                flops_logger.info("\n\n" + "=" * 80)
                flops_logger.info(f"分析时间: {current_time}")
                flops_logger.info("=" * 80)
                
                # 向特殊日志写入详细的FLOPs分析
                flops_logger.info("FLOPs分析结果摘要")
                flops_logger.info("-" * 40)
                flops_logger.info(f"总FLOPs: {self._format_count(flops)}")
                flops_logger.info(f"总MACs: {self._format_count(macs)}")
                flops_logger.info(f"总参数量: {self._format_count(params)}")
                flops_logger.info(f"前向传播时间: {forward_elapsed_time:.4f} s")
                flops_logger.info(f"每秒FLOPs: {self._format_count(flops_per_second)}/s")
                flops_logger.info("-" * 40)
                flops_logger.info("模型架构FLOPs详细分析")
                flops_logger.info("-" * 40)
                
                # 使用自定义输出流将print_model_profile的输出重定向到特殊日志
                # 保存原始stdout
                old_stdout = sys.stdout
                
                try:
                    # 创建自定义writer，将输出重定向到flops_logger
                    logger_writer = LoggerWriter(flops_logger)
                    sys.stdout = logger_writer
                    
                    # 输出详细的模型分析结果，直接重定向到日志文件
                    self.flops_profiler.print_model_profile(
                        profile_step=1,
                        module_depth=-1,  # 所有层级
                        top_modules=50,   # 增加显示的顶级模块数量
                        detailed=True,    # 显示详细信息
                        output_file=None  # 不输出到文件
                    )
                    
                    # 确保所有内容都被刷新到日志
                    logger_writer.flush()
                    
                finally:
                    # 恢复原始stdout
                    sys.stdout = old_stdout
                
                # 告知用户日志文件位置 - 只在第一次分析时显示
                if not file_exists:
                    logger.info(f"FLOPs详细分析正在保存到: {flops_log_file}")
                
                # 如果用户要求直接打印到控制台
                if print_results and False:  # 设置为False以禁用直接打印到控制台
                    self.flops_profiler.print_model_profile(
                        profile_step=1,
                        module_depth=-1,
                        top_modules=3,
                        detailed=True,
                        output_file=None
                    )
                
                # 同时保留我们的详细统计信息 (用于程序内部使用)
                if hasattr(self.flops_profiler, 'module_flops'):
                    module_flops = self.flops_profiler.module_flops
                    module_macs = self.flops_profiler.module_macs
                    module_params = self.flops_profiler.module_params
                    module_latency = getattr(self.flops_profiler, 'module_latency', {})
                    
                    # 构建每层的详细统计信息
                    for module_name, flops_value in module_flops.items():
                        macs_value = module_macs.get(module_name, 0)
                        params_value = module_params.get(module_name, 0)
                        latency_value = module_latency.get(module_name, 0)
                        
                        detailed_results[module_name] = {
                            "flops": flops_value,
                            "macs": macs_value,
                            "params": params_value,
                            "latency": latency_value,
                            "flops_percent": (flops_value / flops * 100) if flops > 0 else 0,
                            "macs_percent": (macs_value / macs * 100) if macs > 0 else 0,
                            "params_percent": (params_value / params * 100) if params > 0 else 0,
                            "latency_percent": (latency_value / forward_elapsed_time * 100) 
                                              if forward_elapsed_time > 0 else 0
                        }
                
            except Exception as e:
                logger.warning(f"获取详细模型分析结果出错: {str(e)}")
                import traceback
                logger.warning(traceback.format_exc())
            
            # 收集基本结果
            results = {
                "flops": flops,
                "macs": macs,
                "params": params,
                "forward_elapsed_time": forward_elapsed_time,
                "flops_per_second": flops_per_second,
                "total_flops": flops,     # 添加总值，方便计算百分比
                "total_macs": macs,
                "total_params": params,
                "total_latency": forward_elapsed_time
            }
            
            # 生成可读的结果
            readable_results = {
                "flops": self._format_count(flops),
                "macs": self._format_count(macs),
                "params": self._format_count(params),
                "forward_elapsed_time": f"{forward_elapsed_time:.4f} s",
                "flops_per_second": self._format_count(flops_per_second) + "/s"
            }
            
            # 如果有硬件监控器，添加指标
            if self.hardware_monitor is not None:
                self.hardware_monitor.add_model_metric("dynamic_flops", flops)
                self.hardware_monitor.add_model_metric("dynamic_macs", macs)
                self.hardware_monitor.add_model_metric("dynamic_params", params)
                self.hardware_monitor.add_model_metric("forward_elapsed_time", forward_elapsed_time)
                self.hardware_monitor.add_model_metric("flops_per_second", flops_per_second)
            
            try:
                # 重置分析器
                self.flops_profiler.end_profile()
            except Exception as e:
                logger.warning(f"重置FLOPs分析器出错: {str(e)}")
            finally:
                self.flops_profiler = None
            
            # 返回包含详细信息的结果
            return {
                "numeric": results,
                "readable": readable_results,
                "detailed": detailed_results
            }
        
        except Exception as e:
            logger.error(f"停止FLOPs分析出错: {str(e)}")
            # 确保分析器被清理
            self.flops_profiler = None
            return None
    
    def _format_count(self, count):
        """
        将数值格式化为可读的字符串
        
        Args:
            count: 要格式化的数值
            
        Returns:
            str: 格式化后的字符串
        """
        if count >= 1e12:
            return f"{count/1e12:.2f} T"
        elif count >= 1e9:
            return f"{count/1e9:.2f} G"
        elif count >= 1e6:
            return f"{count/1e6:.2f} M"
        elif count >= 1e3:
            return f"{count/1e3:.2f} K"
        else:
            return f"{count:.2f}"
    
    def _parse_flops_string(self, flops_str):
        """
        解析FLOPs字符串为数值
        
        Args:
            flops_str: FLOPs字符串，如"1.23 GFLOPs"
            
        Returns:
            float: 解析后的数值
        """
        if not isinstance(flops_str, str):
            return flops_str
        
        # 移除单位并分割
        parts = flops_str.strip().split()
        if len(parts) != 2:
            return float(flops_str)
        
        value, unit = float(parts[0]), parts[1].upper()
        
        # 转换单位
        if 'T' in unit:
            value *= 1e12
        elif 'G' in unit:
            value *= 1e9
        elif 'M' in unit:
            value *= 1e6
        elif 'K' in unit:
            value *= 1e3
        
        return value 