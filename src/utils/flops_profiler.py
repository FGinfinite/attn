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

# 导入DeepSpeed的Flops Profiler
try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

logger = logging.getLogger("attn_experiment")

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
            print_results: 是否打印结果
            
        Returns:
            dict: 分析结果
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
            
            try:
                # 如果需要，打印结果
                if print_results:
                    self.flops_profiler.print_model_profile(
                        profile_step=1,
                        module_depth=-1,
                        top_modules=3,
                        detailed=True,
                        output_file=None
                    )
            except Exception as e:
                logger.warning(f"打印模型分析结果出错: {str(e)}")
            
            # 收集结果
            results = {
                "flops": flops,
                "macs": macs,
                "params": params,
                "forward_elapsed_time": forward_elapsed_time,
                "flops_per_second": flops_per_second
            }
            
            # 生成可读的结果
            readable_results = {
                "flops": self._format_count(flops),
                "macs": self._format_count(macs),
                "params": self._format_count(params),
                "forward_elapsed_time": f"{forward_elapsed_time:.4f} s",
                "flops_per_second": self._format_count(flops_per_second) + "/s"
            }
            
            logger.info(f"动态FLOPs分析完成: FLOPs={readable_results['flops']}, "
                       f"MACs={readable_results['macs']}, 参数量={readable_results['params']}, "
                       f"每秒FLOPs={readable_results['flops_per_second']}")
            
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
            
            return {
                "numeric": results,
                "readable": readable_results
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