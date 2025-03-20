"""
Linformer注意力实现模块
通过低秩投影实现线性注意力复杂度
"""

import logging
import math
import traceback
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.processing_utils import Unpack

logger = logging.getLogger("attn_experiment")


def get_linformer_attention_config(k_ratio=0.25):
    """
    获取Linformer注意力机制配置

    Args:
        k_ratio: 投影维度与序列长度的比例，控制压缩程度

    Returns:
        config: 配置字典
    """
    return {
        "name": "Linformer Attention",
        "description": "通过低秩投影矩阵实现线性复杂度的注意力机制",
        "type": "linformer",
        "k_ratio": k_ratio,
    }


class LinformerQwen2Attention(nn.Module):
    """
    Linformer自注意力模块，将序列长度降维以实现线性复杂度
    兼容Qwen2模型接口规范
    """

    def __init__(
        self, config: Qwen2Config, layer_idx: int, k_ratio=0.25, max_seq_length=4096
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.k_ratio = k_ratio

        # 保持与原始Qwen2Attention相同的参数结构
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # 投影层
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

        # Linformer特有的低秩投影矩阵
        # 计算投影维度k
        self.seq_len = max_seq_length
        self.k = max(1, int(self.seq_len * k_ratio))

        # 修改: E和F投影矩阵直接使用attention_heads维度
        # 而不是使用key_value_heads维度，以兼容分组注意力机制
        self.E = nn.Parameter(
            torch.Tensor(1, self.seq_len, self.k)  # 简化为一个共享的投影矩阵
        )
        self.F = nn.Parameter(
            torch.Tensor(1, self.seq_len, self.k)  # 简化为一个共享的投影矩阵
        )

        # 初始化投影矩阵
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)

        logger.info(
            f"初始化Linformer注意力: hidden_size={config.hidden_size}, "
            f"attention_heads={config.num_attention_heads}, "
            f"kv_heads={config.num_key_value_heads}, "
            f"kv_groups={self.num_key_value_groups}, "
            f"k={self.k}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播
        实现Linformer注意力机制，同时兼容Qwen2Attention接口
        """
        try:
            batch_size, seq_len = hidden_states.shape[:2]

            # 确保序列长度不超过预设的max_seq_length
            if seq_len > self.seq_len:
                raise ValueError(
                    f"输入序列长度 {seq_len} 超过了Linformer的最大序列长度 {self.seq_len}"
                )

            # 提取要输出注意力权重的标志
            output_attentions = kwargs.get("output_attentions", False)

            # 获取输入形状
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            # 生成查询、键、值投影
            query_states = (
                self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )  # [batch, heads, seq, head_dim]
            key_states = (
                self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )  # [batch, kv_heads, seq, head_dim]
            value_states = (
                self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )  # [batch, kv_heads, seq, head_dim]

            # 应用旋转位置编码 (RoPE)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            # 处理KV缓存 (如果有)
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            # 如果使用了kv缓存，则需要调整当前序列长度
            current_seq_len = key_states.shape[2]

            # 计算滑动窗口 (如果配置中定义了)
            sliding_window = None
            if (
                self.config.use_sliding_window
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers
            ):
                sliding_window = self.config.sliding_window

            # 复制键值状态以匹配查询头的数量 (在低秩投影之前)
            if self.num_key_value_groups > 1:
                key_states = key_states.repeat_interleave(
                    self.num_key_value_groups, dim=1
                )
                value_states = value_states.repeat_interleave(
                    self.num_key_value_groups, dim=1
                )

            # ==== Linformer核心实现部分开始 ====

            # 使用低秩投影矩阵降低序列长度维度
            if current_seq_len <= self.seq_len:
                # 调整E和F矩阵的形状，使其适用于当前批次大小和序列长度
                # 获取适当的投影矩阵子集
                E_proj = self.E[:, :current_seq_len, :]  # [1, seq, k]
                F_proj = self.F[:, :current_seq_len, :]  # [1, seq, k]

                # 转置键和值以便于矩阵乘法
                # 从 [batch, heads, seq, dim] 转置为 [batch, heads, dim, seq]
                key_states_t = key_states.transpose(-1, -2)  # [batch, heads, dim, seq]
                value_states_t = value_states.transpose(
                    -1, -2
                )  # [batch, heads, dim, seq]

                # 使用投影矩阵E降低键的序列长度
                # [batch, heads, dim, seq] x [1, seq, k] -> [batch, heads, dim, k]
                projected_keys = torch.matmul(
                    key_states_t, E_proj
                )  # [batch, heads, dim, k]

                # 使用投影矩阵F降低值的序列长度
                # [batch, heads, dim, seq] x [1, seq, k] -> [batch, heads, dim, k]
                projected_values = torch.matmul(
                    value_states_t, F_proj
                )  # [batch, heads, dim, k]

                # 再次转置回原始维度顺序以计算注意力分数
                # 从 [batch, heads, dim, k] 转置为 [batch, heads, k, dim]
                projected_keys = projected_keys.transpose(
                    -1, -2
                )  # [batch, heads, k, dim]
                projected_values = projected_values.transpose(
                    -1, -2
                )  # [batch, heads, k, dim]

                # 计算注意力分数 Q @ K.T / sqrt(head_dim)
                attention_scores = torch.matmul(
                    query_states, projected_keys.transpose(-1, -2)
                )  # [batch, heads, seq, k]
                attention_scores = attention_scores * self.scaling

                # 应用注意力掩码 (如果提供)
                if attention_mask is not None:
                    # 注意：由于我们的键已经被投影到较小的维度，掩码的应用方式需要调整
                    # 我们不能直接使用原始掩码
                    # 这是Linformer与标准注意力的一个区别点
                    attention_scores = attention_scores + attention_mask.unsqueeze(
                        1
                    ).unsqueeze(-1).to(attention_scores.dtype)

                # 应用softmax得到注意力权重
                attention_probs = F.softmax(attention_scores, dim=-1)

                # 应用dropout (如果在训练中)
                if self.training and self.attention_dropout > 0:
                    attention_probs = F.dropout(
                        attention_probs, p=self.attention_dropout
                    )

                # 应用注意力权重到投影的值上
                # [batch, heads, seq, k] x [batch, heads, k, dim] -> [batch, heads, seq, dim]
                attn_output = torch.matmul(attention_probs, projected_values)

            else:
                # 如果序列太长，则退回到标准注意力计算方式
                logger.warning(
                    "序列长度超过了LinformerAttention的预设值，回退到标准注意力计算"
                )

                attention_interface: Callable = eager_attention_forward
                if self.config._attn_implementation != "eager":
                    if self.config._attn_implementation == "sdpa" and output_attentions:
                        logger.warning_once(
                            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                            "Falling back to eager attention."
                        )
                    else:
                        attention_interface = ALL_ATTENTION_FUNCTIONS[
                            self.config._attn_implementation
                        ]

                attn_output, attention_probs = attention_interface(
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

            # ==== Linformer核心实现部分结束 ====

            # 转置回原始维度并合并注意力头
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1)

            # 投影到输出维度
            attn_output = self.o_proj(attn_output)

            if output_attentions:
                # Linformer不直接返回完整的注意力矩阵，因为它使用了投影
                # 我们返回投影后的注意力矩阵
                return attn_output, attention_probs

            return attn_output, None

        except Exception as e:
            logger.error(f"Linformer注意力机制计算失败: {str(e)}")
            traceback.print_exc()
            raise e


def replace_with_linformer_attention(model, k_ratio=0.25, max_seq_length=4096):
    """
    将模型的注意力机制替换为Linformer注意力机制

    Args:
        model: 原始模型
        k_ratio: k的比例，用于确定投影维度
        max_seq_length: 最大序列长度

    Returns:
        model: 替换后的模型
    """
    logger.info(
        f"替换为Linformer注意力机制: k_ratio={k_ratio}, max_seq_length={max_seq_length}"
    )

    # 递归查找并替换所有的Qwen2Attention模块
    replace_count = 0
    for name, module in model.named_modules():
        if (
            hasattr(module, "self_attn")
            and hasattr(module.self_attn, "__class__")
            and module.self_attn.__class__.__name__ == "Qwen2Attention"
        ):
            # 获取原有参数
            old_attn = module.self_attn
            layer_idx = old_attn.layer_idx

            # 获取原始注意力层的设备和数据类型
            orig_device = old_attn.q_proj.weight.device
            orig_dtype = old_attn.q_proj.weight.dtype

            # 创建新的注意力层
            new_attn = LinformerQwen2Attention(
                config=old_attn.config,
                layer_idx=layer_idx,
                k_ratio=k_ratio,
                max_seq_length=max_seq_length,
            )

            # 复制权重并保持相同的数据类型
            new_attn.q_proj.weight.data = old_attn.q_proj.weight.data.clone().to(
                dtype=orig_dtype
            )
            new_attn.k_proj.weight.data = old_attn.k_proj.weight.data.clone().to(
                dtype=orig_dtype
            )
            new_attn.v_proj.weight.data = old_attn.v_proj.weight.data.clone().to(
                dtype=orig_dtype
            )
            new_attn.o_proj.weight.data = old_attn.o_proj.weight.data.clone().to(
                dtype=orig_dtype
            )

            # 复制偏置（如果存在）并保持相同的数据类型
            if hasattr(old_attn.q_proj, "bias") and old_attn.q_proj.bias is not None:
                new_attn.q_proj.bias.data = old_attn.q_proj.bias.data.clone().to(
                    dtype=orig_dtype
                )
            if hasattr(old_attn.k_proj, "bias") and old_attn.k_proj.bias is not None:
                new_attn.k_proj.bias.data = old_attn.k_proj.bias.data.clone().to(
                    dtype=orig_dtype
                )
            if hasattr(old_attn.v_proj, "bias") and old_attn.v_proj.bias is not None:
                new_attn.v_proj.bias.data = old_attn.v_proj.bias.data.clone().to(
                    dtype=orig_dtype
                )

            # 初始化Linformer特有的投影矩阵E和F (注意数据类型一致性)
            nn.init.xavier_uniform_(new_attn.E)
            nn.init.xavier_uniform_(new_attn.F)
            new_attn.E.data = new_attn.E.data.to(dtype=orig_dtype)
            new_attn.F.data = new_attn.F.data.to(dtype=orig_dtype)

            # 将新的注意力层移到与原始层相同的设备上
            new_attn = new_attn.to(device=orig_device, dtype=orig_dtype)

            # 替换注意力层
            module.self_attn = new_attn
            replace_count += 1

    if replace_count == 0:
        logger.warning("没有找到任何Qwen2Attention层！替换失败！")
    else:
        logger.info(
            f"成功替换了 {replace_count} 个注意力层为Linformer注意力机制，保持了原始的设备和数据类型"
        )

    return model
