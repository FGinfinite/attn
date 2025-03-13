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
    from deepspeed.profiling.flops_profiler import get_model_profile
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
        self.model_stats = {}
    
    def profile_model_statistics(self, model, input_shape=(1, 512), input_type=torch.LongTensor,
                                detailed=False, warm_up=1, as_string=True, print_results=True):
        """
        静态分析模型参数量和理论FLOPs
        
        Args:
            model: 要分析的模型
            input_shape: 输入形状，默认为(1, 512)
            input_type: 输入类型，默认为torch.LongTensor
            detailed: 是否输出详细信息
            warm_up: 预热次数
            as_string: 是否将结果转换为字符串
            print_results: 是否打印结果
            
        Returns:
            tuple: (flops, macs, params, results)
        """
        if not self.available:
            logger.warning("DeepSpeed Flops Profiler不可用")
            return None, None, None, None
        
        try:
            logger.info(f"开始分析模型架构，输入形状: {input_shape}")
            
            
            # 获取模型配置
            flops, macs, params, results = get_model_profile(
                model=model,
                input_shape=input_shape,
                detailed=detailed,
                warm_up=warm_up,
                as_string=as_string,
                print_profile=print_results
            )
            
            logger.info(f"模型静态分析完成: FLOPs={flops}, MACs={macs}, 参数量={params}")
            
            # 保存结果
            self.model_stats = {
                "flops": flops,
                "macs": macs,
                "params": params,
                "detailed_results": results
            }
            
            # 如果有硬件监控器，添加指标
            if self.hardware_monitor is not None:
                # 将字符串转换为数值（如果需要）
                if as_string and isinstance(flops, str):
                    try:
                        flops_val = self._parse_flops_string(flops)
                        macs_val = self._parse_flops_string(macs)
                        params_val = self._parse_flops_string(params)
                        
                        self.hardware_monitor.add_model_metric("model_flops", flops_val)
                        self.hardware_monitor.add_model_metric("model_macs", macs_val)
                        self.hardware_monitor.add_model_metric("model_params", params_val)
                    except Exception as e:
                        logger.error(f"解析FLOPs字符串出错: {str(e)}")
                else:
                    self.hardware_monitor.add_model_metric("model_flops", flops)
                    self.hardware_monitor.add_model_metric("model_macs", macs)
                    self.hardware_monitor.add_model_metric("model_params", params)
            
            return flops, macs, params, results
        
        except Exception as e:
            logger.error(f"分析模型静态FLOPs出错: {str(e)}")
            return None, None, None, None
    
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
            flops = self.flops_profiler.get_total_flops()
            macs = self.flops_profiler.get_total_macs()
            params = self.flops_profiler.get_total_params()
            forward_elapsed_time = self.flops_profiler.get_total_duration()
            flops_per_second = flops / forward_elapsed_time
            
            # 如果需要，打印结果
            if print_results:
                self.flops_profiler.print_model_profile(
                    profile_step=1,
                    module_depth=-1,
                    top_modules=3,
                    detailed=True,
                    output_file=None
                )
            
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
            
            # 重置分析器
            self.flops_profiler.end_profile()
            
            return {
                "numeric": results,
                "readable": readable_results
            }
        
        except Exception as e:
            logger.error(f"停止FLOPs分析出错: {str(e)}")
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