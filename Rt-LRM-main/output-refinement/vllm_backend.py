"""
VLLM Backend for Local Model Deployment
支持多卡并行推理 (Tensor Parallelism)
"""

import os

# 设置vLLM环境变量（必须在导入vllm之前）
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60")

# 禁用V1引擎以避免 EngineCoreRequestType 错误
# 必须在导入vllm之前设置此环境变量
os.environ.setdefault("VLLM_USE_V1", "0")

# import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import threading

# 线程锁，保护VLLM调用（VLLM不是线程安全的）
_VLLM_LOCK = threading.Lock()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储vLLM实例
_VLLM_MODELS = {}


@dataclass
class VLLMConfig:
    """VLLM配置类"""
    model_path: str  # 本地模型路径或HuggingFace模型名
    tensor_parallel_size: int = 1  # GPU并行数
    gpu_memory_utilization: float = 0.85  # GPU内存利用率
    max_model_len: int = 4096  # 最大模型长度
    dtype: str = "auto"  # 数据类型: auto, float16, bfloat16, float32
    quantization: Optional[str] = None  # 量化: awq, gptq, None
    trust_remote_code: bool = True
    
    # 推理参数
    temperature: float = 1.0
    max_tokens: int = 256
    top_p: float = 0.9
    top_k: int = -1
    repetition_penalty: float = 1.0


def get_vllm_model(config: VLLMConfig):
    """
    获取或创建vLLM模型实例（单例模式）
    
    Args:
        config: VLLM配置
        
    Returns:
        LLM实例
    """
    global _VLLM_MODELS
    
    # 使用配置作为key
    cache_key = f"{config.model_path}_{config.tensor_parallel_size}"
    
    if cache_key not in _VLLM_MODELS:
        try:
            from vllm import LLM
            
            logger.info(f"Initializing vLLM model: {config.model_path}")
            logger.info(f"Tensor parallel size: {config.tensor_parallel_size}")
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
            
            # 设置环境变量以确保多卡正确初始化
            if config.tensor_parallel_size > 1:
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            
            model_kwargs = {
                "model": config.model_path,
                "tensor_parallel_size": config.tensor_parallel_size,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "max_model_len": config.max_model_len,
                "trust_remote_code": config.trust_remote_code,
            }
            
            # 添加dtype（如果不是auto）
            if config.dtype != "auto":
                model_kwargs["dtype"] = config.dtype
            
            # 添加量化配置
            if config.quantization:
                model_kwargs["quantization"] = config.quantization
            
            _VLLM_MODELS[cache_key] = LLM(**model_kwargs)
            logger.info("vLLM model initialized successfully")
            
        except ImportError:
            raise ImportError(
                "vllm is not installed. Please install it with: "
                "pip install vllm"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise
    
    return _VLLM_MODELS[cache_key]


def query_vllm(
    model_path: str,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 256,
    temperature: float = 1.0,
    stop_sequences: List[str] = None,
    tensor_parallel_size: int = 1,
    **kwargs
) -> str:
    """
    使用vLLM进行推理，保持与原有query函数的兼容性
    
    注意：此函数使用线程锁保护，确保VLLM在多线程环境下安全运行
    
    Args:
        model_path: 模型路径
        user_prompt: 用户提示
        system_prompt: 系统提示
        max_tokens: 最大生成token数
        temperature: 温度
        stop_sequences: 停止序列
        tensor_parallel_size: GPU并行数
        **kwargs: 其他参数
        
    Returns:
        生成的文本
    """
    try:
        from vllm import SamplingParams
        
        # 构建配置
        config = VLLMConfig(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # 获取模型
        llm = get_vllm_model(config)
        
        # 构建完整提示（支持不同模型的对话格式）
        full_prompt = format_prompt(model_path, system_prompt, user_prompt)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            stop=stop_sequences if stop_sequences else None,
        )
        
        # 使用线程锁保护VLLM调用（VLLM不是线程安全的）
        global _VLLM_LOCK
        with _VLLM_LOCK:
            outputs = llm.generate(full_prompt, sampling_params)
        
        # 提取生成的文本
        generated_text = outputs[0].outputs[0].text
        
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f"Error in query_vllm: {e}")
        raise


def format_prompt(model_path: str, system_prompt: str, user_prompt: str) -> str:
    """
    根据不同模型格式化提示
    
    Args:
        model_path: 模型路径或名称
        system_prompt: 系统提示
        user_prompt: 用户提示
        
    Returns:
        格式化后的提示
    """
    model_lower = model_path.lower()
    
    # Qwen模型格式
    if "qwen" in model_lower:
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Llama-2/3 Chat格式
    elif "llama" in model_lower:
        if "llama-3" in model_lower:
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:  # Llama-2
            return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
    
    # ChatGLM格式
    elif "chatglm" in model_lower:
        return f"[gMASK]sop<|system|>\n{system_prompt}<|user|>\n{user_prompt}<|assistant|>\n"
    
    # Baichuan格式
    elif "baichuan" in model_lower:
        return f"<reserved_106>{system_prompt}<reserved_107>{user_prompt}"
    
    # 默认使用通用对话格式
    else:
        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"


class VLLMChatModel:
    """
    兼容LangChain接口的VLLM聊天模型
    可以替换原有的ChatOpenAI
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # 初始化模型
        config = VLLMConfig(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        self.llm = get_vllm_model(config)
        self.config = config
    
    def __call__(self, messages):
        """
        兼容LangChain的调用方式
        
        Args:
            messages: List[BaseMessage] 或类似格式
            
        Returns:
            AIMessage
        """
        from langchain.schema import AIMessage
        
        # 解析消息
        system_prompt = "You are a helpful assistant."
        user_prompt = ""
        
        for msg in messages:
            if msg.type == "system":
                system_prompt = msg.content
            elif msg.type == "human":
                user_prompt = msg.content
        
        # 调用vLLM
        response = query_vllm(
            model_path=self.model_path,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tensor_parallel_size=self.tensor_parallel_size,
            **self.kwargs
        )
        
        return AIMessage(content=response)
    
    def generate(self, prompts):
        """批量生成接口（线程安全）"""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        
        # 使用线程锁保护VLLM调用
        global _VLLM_LOCK
        with _VLLM_LOCK:
            outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


def batch_query_vllm(
    model_path: str,
    prompts: List[Dict[str, str]],
    tensor_parallel_size: int = 1,
    max_tokens: int = 256,
    temperature: float = 1.0,
    batch_size: int = 8,
    **kwargs
) -> List[str]:
    """
    批量查询vLLM，提高多卡利用率
    
    Args:
        model_path: 模型路径
        prompts: 提示列表，每个元素是{"system": "...", "user": "..."}
        tensor_parallel_size: GPU并行数
        max_tokens: 最大token数
        temperature: 温度
        batch_size: 批处理大小
        **kwargs: 其他参数
        
    Returns:
        生成文本列表
    """
    try:
        from vllm import SamplingParams
        
        config = VLLMConfig(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        llm = get_vllm_model(config)
        
        # 格式化所有提示
        formatted_prompts = [
            format_prompt(model_path, p.get("system", ""), p.get("user", ""))
            for p in prompts
        ]
        
        # 采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # 批量生成（使用线程锁保护）
        all_outputs = []
        global _VLLM_LOCK
        for i in range(0, len(formatted_prompts), batch_size):
            batch = formatted_prompts[i:i+batch_size]
            with _VLLM_LOCK:
                outputs = llm.generate(batch, sampling_params)
            all_outputs.extend([o.outputs[0].text for o in outputs])
        
        return all_outputs
        
    except Exception as e:
        logger.error(f"Error in batch_query_vllm: {e}")
        raise


def clear_vllm_cache():
    """清除vLLM模型缓存，释放GPU内存"""
    global _VLLM_MODELS
    _VLLM_MODELS.clear()
    torch.cuda.empty_cache()
    logger.info("VLLM cache cleared")


# 便捷函数：检测是否使用vLLM
def is_vllm_model(model_name: str) -> bool:
    """
    检测模型名称是否指向本地vLLM模型
    
    Args:
        model_name: 模型名称或路径
        
    Returns:
        是否是vLLM模型
    """
    # 如果模型名是路径或包含特定关键词，认为是本地模型
    indicators = [
        "/",  # 路径分隔符
        "./",
        "../",
        "vllm://",  # vLLM协议前缀
    ]
    
    # 如果模型名以vllm://开头，明确使用vLLM
    if model_name.startswith("vllm://"):
        return True
    
    # 检查是否是文件路径
    if any(ind in model_name for ind in indicators):
        return os.path.exists(model_name) or os.path.isdir(model_name)
    
    return False


def get_model_path(model_name: str) -> str:
    """
    从模型名称中提取实际路径（去除vllm://前缀）
    
    Args:
        model_name: 模型名称
        
    Returns:
        实际模型路径
    """
    if model_name.startswith("vllm://"):
        return model_name[7:]  # 去掉 "vllm://"
    return model_name



def diagnose_vllm_error(error: Exception) -> str:
    """
    诊断vLLM错误并提供解决方案
    
    Args:
        error: 异常对象
        
    Returns:
        诊断信息和解决方案
    """
    error_msg = str(error)
    
    # 常见的vLLM V1错误
    if "EngineCoreRequestType" in error_msg or "is not a valid EngineCoreRequestType" in error_msg:
        return """
[错误诊断] vLLM V1版本多进程错误

原因：vLLM V1版本的多进程通信存在问题

解决方案1（推荐）：
    export VLLM_USE_V1=0
    python your_script.py

解决方案2：
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    python your_script.py

解决方案3：降级vLLM
    pip install vllm==0.5.4
        """
    
    elif "CUDA out of memory" in error_msg:
        return """
[错误诊断] CUDA显存不足

解决方案1：减少GPU内存利用率
    python your_script.py --vllm_gpu_util 0.70

解决方案2：使用更多GPU
    python your_script.py --vllm_tp_size 8

解决方案3：使用量化模型
    # 在VLLMConfig中设置
    quantization="awq"  # 或 "gptq"
        """
    
    elif "No module named 'vllm'" in error_msg:
        return """
[错误诊断] vLLM未安装

解决方案：
    pip install vllm

或指定版本：
    pip install vllm==0.5.4
        """
    
    else:
        return f"""
[错误诊断] 未知错误

错误信息：{error_msg}

通用解决方案：
1. 设置环境变量：
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_V1=0

2. 清理GPU缓存：
    python -c "import torch; torch.cuda.empty_cache()"

3. 重启Python进程
        """


def check_vllm_installation():
    """检查vLLM安装状态并打印诊断信息"""
    print("=" * 60)
    print("vLLM安装检查")
    print("=" * 60)
    
    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch未安装")
    
    # 检查vLLM
    try:
        import vllm
        print(f"vLLM版本: {vllm.__version__}")
    except ImportError:
        print("❌ vLLM未安装")
    except AttributeError:
        print("✅ vLLM已安装（版本号未知）")
    
    # 检查环境变量
    print("\n环境变量：")
    env_vars = ["VLLM_WORKER_MULTIPROC_METHOD", "VLLM_USE_V1", "VLLM_TP_SIZE", "CUDA_VISIBLE_DEVICES"]
    for var in env_vars:
        value = os.getenv(var, "未设置")
        print(f"  {var}: {value}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 运行安装检查
    check_vllm_installation()
