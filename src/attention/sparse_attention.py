"""
稀疏注意力机制模块
"""

import math
import torch
import logging
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger("attn_experiment")

def get_sparse_attention_config(sparsity=0.8):
    """
    获取稀疏注意力机制配置
    
    Args:
        sparsity: 稀疏度，表示保留的注意力比例
    
    Returns:
        config: 配置字典
    """
    return {
        "name": "Sparse Attention",
        "description": "修改RotaryEmbedding模块并集成动态稀疏掩码",
        "type": "sparse",
        "sparsity": sparsity
    }

def replace_with_sparse_attention(model, sparsity=0.8, last_layer_only=False):
    """
    将模型的最后一层注意力机制替换为稀疏注意力机制
    
    Args:
        model: 原始模型
        sparsity: 稀疏度，表示保留的注意力比例
        last_layer_only: 是否只替换最后一层注意力，默认为False（当前实现已固定为只替换最后一层）
    
    Returns:
        model: 替换后的模型
    """
    logger.info(f"使用稀疏注意力机制，稀疏度: {sparsity}")
    logger.info(f"注意：只替换最后一层注意力机制")
    replaced_count = 0
    
    # Qwen2.5特定的注意力层名称模式
    qwen_attn_names = [
        "Qwen2SdpaAttention", 
        "Qwen2Attention", 
        "SdpaAttention", 
        "self_attn"
    ]
    
    # 收集所有注意力模块
    attention_modules = []
    for name, module in model.named_modules():
        # 检查是否是注意力模块
        is_attention_module = False
        for attn_name in qwen_attn_names:
            if attn_name in name or (hasattr(module, "__class__") and attn_name in module.__class__.__name__):
                is_attention_module = True
                break
        
        # 检查是否有compute_attention方法
        has_compute_attention = hasattr(module, "compute_attention") and callable(getattr(module, "compute_attention"))
        
        # 检查子模块，寻找q_proj, k_proj, v_proj, o_proj
        has_qkv_projections = False
        if hasattr(module, "named_children"):
            child_names = [n for n, _ in module.named_children()]
            if all(p in child_names for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                has_qkv_projections = True
        
        if is_attention_module or has_compute_attention or has_qkv_projections:
            attention_modules.append((name, module))
            
            # 记录模块的关键属性和方法
            methods = [m for m in dir(module) if not m.startswith('_') and callable(getattr(module, m))]
            important_methods = [m for m in methods if 'atten' in m.lower() or 'forw' in m.lower()]
            logger.debug(f"  方法: {important_methods}")
            
            # 检查模块的关键子模块
            submodules = {n: type(m).__name__ for n, m in module.named_children() 
                         if any(p in n for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'query', 'key', 'value'])}
            if submodules:
                logger.debug(f"  子模块: {submodules}")
    
    if not attention_modules:
        logger.warning("未找到任何符合Qwen2.5注意力层名称模式的模块！")
        logger.info("尝试打印完整的模型结构以协助调试...")
        module_types = set()
        for name, module in model.named_modules():
            module_type = type(module).__name__
            module_types.add(module_type)
            if 'atten' in name.lower() or 'atten' in module_type.lower():
                logger.info(f"可能的注意力相关模块: {name}, 类型: {module_type}")
        
        logger.info(f"模型中的所有模块类型: {sorted(list(module_types))}")
        return model
    
    # 只处理最后一个注意力模块
    if attention_modules:
        name, module = attention_modules[-1]
        logger.info(f"准备替换最后一层注意力模块: {name}")
        
        try:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent_module = model.get_submodule(parent_name) if parent_name else model
            
            # 创建新的稀疏注意力模块
            config = module.config if hasattr(module, "config") else getattr(model, "config", None)
            sparse_module = SparseSelfAttention(
                config=config,
                sparsity=sparsity
            )
            
            # 复制必要的属性
            for attr_name in ['hidden_size', 'num_heads', 'head_dim', 'rotary_emb', 
                             'num_key_value_heads', 'num_attention_heads']:
                if hasattr(module, attr_name):
                    setattr(sparse_module, attr_name, getattr(module, attr_name))
            
            # 复制投影层
            if all(hasattr(module, proj) for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                sparse_module.q_proj = module.q_proj
                sparse_module.k_proj = module.k_proj
                sparse_module.v_proj = module.v_proj
                sparse_module.o_proj = module.o_proj
                logger.info(f"已复制原始模块的投影层参数")
            
            # 添加特殊标记，方便后续检查是否真的替换了
            sparse_module._attn_type = "sparse"
            sparse_module._sparsity = sparsity
            
            # 替换模块
            module_name = name.split(".")[-1]
            if hasattr(parent_module, module_name):
                setattr(parent_module, module_name, sparse_module)
                replaced_count += 1
                logger.info(f"成功替换最后一层注意力模块: {name}")
            else:
                # 如果不能直接替换整个模块，则使用猴子补丁替换方法
                logger.info(f"尝试为最后一层 {name} 应用猴子补丁...")
                
                # 尝试替换计算注意力的函数
                if hasattr(module, "compute_attention"):
                    # 保存原始的计算注意力函数
                    original_compute_attention = module.compute_attention
                    
                    # 定义新的计算注意力函数并添加标记
                    def sparse_compute_attention(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask=None,
                        output_attentions=False,
                        **kwargs
                    ):
                        # 添加标记属性
                        if not hasattr(self, "_attn_type"):
                            self._attn_type = "sparse"
                            self._sparsity = sparsity
                        
                        # 打印输入形状以帮助调试
                        logger.debug(f"Sparse attention input shapes - query: {query_states.shape}, key: {key_states.shape}")
                        
                        # 以下是稀疏注意力的实现
                        batch_size, num_heads, query_length, head_dim = query_states.shape
                        key_length = key_states.size(-2)
                        
                        # 计算注意力分数
                        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
                        attention_scores = attention_scores / math.sqrt(head_dim)
                        
                        # 应用注意力掩码（如果有）
                        if attention_mask is not None:
                            attention_scores = attention_scores + attention_mask
                        
                        # 保存原始注意力分数用于调试
                        orig_shape = attention_scores.shape
                        
                        # 应用稀疏掩码 - 只保留每行最重要的(1-sparsity)比例的连接
                        k = max(1, int((1 - sparsity) * key_length))
                        
                        # 获取前k个最大值及其索引
                        topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1)
                        
                        # 创建稀疏掩码
                        sparse_mask = torch.ones_like(attention_scores) * float('-inf')
                        
                        # 将topk位置设置为原始分数
                        for b in range(batch_size):
                            for h in range(num_heads):
                                for q in range(query_length):
                                    sparse_mask[b, h, q].scatter_(-1, topk_indices[b, h, q], attention_scores[b, h, q, topk_indices[b, h, q]])
                        
                        # 应用掩码后的注意力分数
                        attention_scores = sparse_mask
                        
                        # 应用softmax得到注意力权重
                        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(value_states.dtype)
                        
                        # 计算注意力输出
                        context_states = torch.matmul(attention_probs, value_states)
                        
                        # 添加调试信息
                        if logger.level <= logging.DEBUG:
                            nonzero_count = (attention_probs > 1e-5).float().sum().item()
                            total_count = attention_probs.numel()
                            actual_sparsity = 1.0 - (nonzero_count / total_count)
                            logger.debug(f"Actual attention sparsity: {actual_sparsity:.4f} (target: {sparsity:.4f})")
                        
                        # 根据原始函数的返回值格式返回结果
                        if output_attentions:
                            return context_states, attention_probs
                        else:
                            return context_states
                    
                    # 替换计算注意力函数
                    try:
                        module.compute_attention = sparse_compute_attention.__get__(module, type(module))
                        replaced_count += 1
                        logger.info(f"成功替换最后一层注意力函数: {name}.compute_attention")
                    except Exception as e:
                        logger.warning(f"替换最后一层attention函数失败: {str(e)}")
                
                # 尝试替换forward方法
                elif hasattr(module, "forward"):
                    # 保存原始的forward函数
                    original_forward = module.forward
                    
                    # 定义新的forward函数
                    def sparse_forward(self, *args, **kwargs):
                        # 添加标记属性
                        if not hasattr(self, "_attn_type"):
                            self._attn_type = "sparse"
                            self._sparsity = sparsity
                        
                        # 检查输入参数
                        hidden_states = args[0] if args else kwargs.get('hidden_states', None)
                        if hidden_states is None:
                            logger.warning(f"无法识别的forward输入参数，回退到原始forward方法")
                            return original_forward(*args, **kwargs)
                        
                        # 获取其他参数
                        attention_mask = kwargs.get('attention_mask', None)
                        output_attentions = kwargs.get('output_attentions', False)
                        
                        # 其他所有参数传递给原始forward函数处理
                        outputs = original_forward(*args, **kwargs)
                        
                        # 如果输出是元组，检查第二个元素是否是注意力权重
                        if isinstance(outputs, tuple) and len(outputs) > 1 and output_attentions:
                            attn_output, attn_weights = outputs[0], outputs[1]
                            
                            # 对注意力权重应用稀疏掩码
                            if attn_weights is not None:
                                try:
                                    # 获取注意力权重的形状
                                    batch_size, num_heads, seq_len, _ = attn_weights.shape
                                    
                                    # 创建稀疏掩码
                                    k = max(1, int((1 - sparsity) * seq_len))
                                    
                                    # 获取前k个最大值及其索引
                                    topk_values, topk_indices = torch.topk(attn_weights, k, dim=-1)
                                    
                                    # 创建稀疏掩码
                                    sparse_mask = torch.zeros_like(attn_weights)
                                    
                                    # 将mask的topk位置设置为1
                                    for b in range(batch_size):
                                        for h in range(num_heads):
                                            for q in range(seq_len):
                                                sparse_mask[b, h, q].scatter_(-1, topk_indices[b, h, q], 
                                                                   attn_weights[b, h, q, topk_indices[b, h, q]])
                                    
                                    # 使用稀疏mask重新生成注意力权重
                                    sparse_attn_weights = F.normalize(sparse_mask, p=1, dim=-1)
                                    
                                    # 创建新的输出元组
                                    outputs = (attn_output,) + (sparse_attn_weights,) + outputs[2:]
                                    logger.debug(f"已应用稀疏注意力掩码到注意力权重")
                                except Exception as e:
                                    logger.warning(f"应用稀疏掩码失败: {str(e)}")
                        
                        return outputs
                    
                    # 替换forward函数
                    try:
                        module.forward = sparse_forward.__get__(module, type(module))
                        replaced_count += 1
                        logger.info(f"成功替换最后一层forward函数: {name}.forward")
                    except Exception as e:
                        logger.warning(f"替换最后一层forward函数失败: {str(e)}")
        
        except Exception as e:
            logger.warning(f"替换最后一层注意力模块 {name} 失败: {str(e)}")
    
    if replaced_count > 0:
        logger.info(f"已成功替换最后一层注意力模块/函数")
    else:
        logger.warning("未能替换最后一层注意力模块或函数！这可能意味着模型架构与预期不符。")
        logger.warning("请检查Qwen2.5模型的具体实现，并更新注意力层的匹配逻辑。")
    
    return model

class SparseSelfAttention(torch.nn.Module):
    """
    稀疏自注意力机制模块，兼容Qwen2.5模型的注意力层结构
    """
    
    def __init__(self, config, sparsity=0.8):
        super().__init__()
        self.config = config
        self.sparsity = sparsity
        # 标记注意力类型
        self._attn_type = "sparse"
        self._sparsity = sparsity
        
        # 设置默认值，防止属性错误
        self.hidden_size = 4096 # 默认值，会在复制时被覆盖
        self.num_heads = 32 # 默认值，会在复制时被覆盖
        self.num_key_value_heads = 32 # 默认值，可能被覆盖
        self.num_attention_heads = 32 # Qwen2.5可能使用的字段
        
        # 计算head_dim
        if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
            self.head_dim = config.hidden_size // config.num_attention_heads
        else:
            self.head_dim = 128 # 默认值
        
        # 初始化必要的投影层
        if config is not None and hasattr(config, "hidden_size"):
            self.hidden_size = config.hidden_size
            
            # 创建查询、键、值投影层
            self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
            
            # 初始化旋转位置编码(如有需要)
            self.rotary_emb = None
    
    def _split_heads(self, tensor, num_heads, head_dim):
        """将隐藏状态拆分为多个注意力头"""
        new_shape = tensor.shape[:-1] + (num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
    
    def _merge_heads(self, tensor, num_heads, head_dim):
        """将多个注意力头合并回隐藏状态"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.shape[:-2] + (num_heads * head_dim,)
        return tensor.view(new_shape)
    
    def compute_attention(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        output_attentions=False,
    ):
        """
        计算稀疏注意力
        
        参数:
            query_states: 查询状态，形状为(batch_size, num_heads, seq_length, head_dim)
            key_states: 键状态，形状为(batch_size, num_heads, seq_length, head_dim)
            value_states: 值状态，形状为(batch_size, num_heads, seq_length, head_dim)
            attention_mask: 注意力掩码
            output_attentions: 是否输出注意力权重
        """
        # 打印输入形状以帮助调试
        logger.debug(f"Sparse attention input shapes - query: {query_states.shape}, key: {key_states.shape}")
        
        # 计算注意力分数
        batch_size, num_heads, query_length, head_dim = query_states.shape
        key_length = key_states.size(-2)
        
        # 计算注意力分数 (batch_size, num_heads, query_length, key_length)
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(head_dim)
        
        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            # 确保掩码形状正确
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
                
            attention_scores = attention_scores + attention_mask
        
        # 应用稀疏掩码 - 只保留每行最重要的(1-sparsity)比例的连接
        k = max(1, int((1 - self.sparsity) * key_length))
        
        # 获取每行的topk，保留最重要的注意力连接
        topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1)
        
        # 创建稀疏掩码
        sparse_mask = torch.ones_like(attention_scores) * float('-inf')
        
        # 使用scatter操作将topk位置设置为原始分数
        for b in range(batch_size):
            for h in range(num_heads):
                for q in range(query_length):
                    sparse_mask[b, h, q].scatter_(-1, topk_indices[b, h, q], 
                                    attention_scores[b, h, q, topk_indices[b, h, q]])
        
        # 应用稀疏掩码得到最终注意力分数
        attention_scores = sparse_mask
        
        # 应用softmax得到注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(value_states.dtype)
        
        # 计算注意力输出 (batch_size, num_heads, seq_length, head_dim)
        context_states = torch.matmul(attention_probs, value_states)
        
        # 添加调试信息
        if logger.level <= logging.DEBUG:
            nonzero_count = (attention_probs > 1e-5).float().sum().item()
            total_count = attention_probs.numel()
            actual_sparsity = 1.0 - (nonzero_count / total_count)
            logger.debug(f"Actual attention sparsity: {actual_sparsity:.4f} (target: {self.sparsity:.4f})")
        
        if output_attentions:
            return context_states, attention_probs
        else:
            return context_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        """
        前向传播
        
        Args:
            hidden_states: 输入隐藏状态
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_value: 过去的键值对
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用缓存
        
        Returns:
            输出隐藏状态，注意力权重，键值对
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # 处理注意力掩码
        if attention_mask is not None:
            # 确保掩码形状正确
            if attention_mask.dim() == 2:
                # 扩展掩码形状以适应注意力计算
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # 掩码中的0表示被遮掩的token，1表示需要关注的token
            # 将掩码转换为attention bias: 0 -> 0, 1 -> -10000
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 投影查询、键、值
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 处理GQA (分组查询注意力) - 如果有必要
        is_gqa = hasattr(self, "num_key_value_heads") and self.num_key_value_heads != self.num_heads
        
        # 调整形状以适应注意力计算
        head_dim = self.hidden_size // self.num_heads
        
        # 拆分头
        query_states = self._split_heads(query_states, self.num_heads, head_dim)
        
        # 对于GQA，键值头可能少于查询头
        kv_heads = self.num_key_value_heads if is_gqa else self.num_heads
        key_states = self._split_heads(key_states, kv_heads, head_dim)
        value_states = self._split_heads(value_states, kv_heads, head_dim)
        
        # 应用旋转位置编码(如果存在)
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None and position_ids is not None:
            if callable(getattr(self.rotary_emb, "__call__", None)):
                query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
            else:
                logger.warning("rotary_emb属性存在但不是可调用的函数")
        
        # 处理过去的键值对(如果用于缓存)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=-2)
            value_states = torch.cat([past_value, value_states], dim=-2)
        
        # 如果使用缓存，保存当前的键值对
        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None
        
        # 计算稀疏注意力
        context_states = self.compute_attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        # 处理返回值
        if output_attentions:
            context_states, attention_weights = context_states
        else:
            attention_weights = None
        
        # 合并头
        context_states = self._merge_heads(context_states, self.num_heads, head_dim)
        
        # 输出投影
        output = self.o_proj(context_states)
        
        # 构建返回元组
        if output_attentions:
            return (output, attention_weights, present_key_value)
        else:
            return (output, None, present_key_value) 