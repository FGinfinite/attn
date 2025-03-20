"""
自定义注意力机制模块 - 用于验证注意力替换是否成功
"""

import torch
import torch.nn as nn
import logging
import random
from typing import Callable, Optional, Tuple

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

logger = logging.getLogger("attn_experiment")

def get_custom_attention_config(noise_level=0.01):
    """
    获取自定义注意力机制配置
    
    Args:
        noise_level: 随机噪声水平，默认为0.01
        
    Returns:
        config: 配置字典
    """
    return {
        "name": "自定义验证注意力",
        "description": "在标准注意力基础上添加随机扰动，用于验证替换是否成功",
        "type": "custom",
        "noise_level": noise_level
    }

class CustomQwen2Attention(nn.Module):
    """
    自定义注意力机制，基于Qwen2Attention实现
    添加随机扰动和日志输出，以验证替换是否成功
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: int, noise_level=0.01):
        """
        初始化
        
        Args:
            config: 配置
            layer_idx: 层索引
            noise_level: 随机噪声水平，默认为0.01
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.noise_level = noise_level
        
        # 保持与原始Qwen2Attention相同的参数和结构
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        # 注意：这里保持与Qwen2Attention相同的投影层
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播
        与原始Qwen2Attention一致，但添加随机扰动
        """
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 添加随机扰动，用于验证替换是否成功
        if self.training or True:
            noise = torch.randn_like(query_states) * self.noise_level
            query_states = query_states + noise
            
            noise = torch.randn_like(key_states) * self.noise_level
            key_states = key_states + noise
            
            noise = torch.randn_like(value_states) * self.noise_level
            value_states = value_states + noise

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

        # 添加额外的随机扰动
        if self.training:
            noise = torch.randn_like(attn_output) * self.noise_level
            attn_output = attn_output + noise

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights

def replace_with_custom_attention(model, noise_level=0.01):
    """
    将模型的注意力机制替换为自定义注意力机制
    
    Args:
        model: 原始模型
        noise_level: 随机噪声水平，默认为0.01
    
    Returns:
        model: 替换后的模型
    """
    
    # 递归查找并替换所有的Qwen2Attention模块
    replace_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'self_attn') and hasattr(module.self_attn, '__class__') and module.self_attn.__class__.__name__ == 'Qwen2Attention':

            # 获取原有参数
            old_attn = module.self_attn
            layer_idx = old_attn.layer_idx
            
            # 获取原始注意力层的设备和数据类型
            orig_device = old_attn.q_proj.weight.device
            orig_dtype = old_attn.q_proj.weight.dtype
            

            
            # 创建新的注意力层
            new_attn = CustomQwen2Attention(
                config=old_attn.config, 
                layer_idx=layer_idx,
                noise_level=noise_level
            )
            
            # 复制权重并保持相同的数据类型
            new_attn.q_proj.weight.data = old_attn.q_proj.weight.data.clone().to(dtype=orig_dtype)
            new_attn.k_proj.weight.data = old_attn.k_proj.weight.data.clone().to(dtype=orig_dtype)
            new_attn.v_proj.weight.data = old_attn.v_proj.weight.data.clone().to(dtype=orig_dtype)
            new_attn.o_proj.weight.data = old_attn.o_proj.weight.data.clone().to(dtype=orig_dtype)
            
            # 复制偏置（如果存在）并保持相同的数据类型
            if hasattr(old_attn.q_proj, 'bias') and old_attn.q_proj.bias is not None:
                new_attn.q_proj.bias.data = old_attn.q_proj.bias.data.clone().to(dtype=orig_dtype)
            if hasattr(old_attn.k_proj, 'bias') and old_attn.k_proj.bias is not None:
                new_attn.k_proj.bias.data = old_attn.k_proj.bias.data.clone().to(dtype=orig_dtype)
            if hasattr(old_attn.v_proj, 'bias') and old_attn.v_proj.bias is not None:
                new_attn.v_proj.bias.data = old_attn.v_proj.bias.data.clone().to(dtype=orig_dtype)
            
            # 将新的注意力层移到与原始层相同的设备上
            new_attn = new_attn.to(device=orig_device, dtype=orig_dtype)
            
            # 替换注意力层
            module.self_attn = new_attn
            replace_count += 1
            

    
    if replace_count == 0:
        logger.warning("没有找到任何Qwen2Attention层！替换失败！")
    else:
        logger.info(f"成功替换了 {replace_count} 个注意力层，保持了原始的设备和数据类型")
    
    return model 