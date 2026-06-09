import logging
import os
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models import ChatAnthropic as LangchainChatAnthropic
from langchain.chat_models import ChatOpenAI as LangchainChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.openai import convert_dict_to_message
from langchain.llms import VLLM, OpenAI
from .transformers_llm import TransformersLLM
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .my_typing import *

model_path = "D:/myPyProgram/llm-feedback-main"

MODEL_BASE_NAMES = [
    "gpt-4",
    "gpt-4o",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "claude-1",
    "claude-2",
    "lmsys/vicuna-7b-v1.5-16k",
    "lmsys/vicuna-13b-v1.5-16k",
    "qwen2.5-72b-instruct",
    "llama-3.1-405b-instruct",
    f"transformers/{model_path}/RealSafe_R1",
]

MODEL_SHORT_NAMES = [
    "gpt-4",
    "gpt-4o",
    "gpt-3.5",
    "gpt-3.5-16k",
    "claude-1",
    "claude-2",
    "vicuna-7b",
    "vicuna-13b",
    "qwen",
    "llama3",
    "RealSafe_R1",
]


MODEL_NAMES_MAPPING = dict(zip(MODEL_SHORT_NAMES, MODEL_BASE_NAMES))
FIXED_VERSIONS = {
    "gpt-4": ["turbo","0613", "0314"],
    "gpt-3.5": ["turbo","0613", "0301","1106","1106"],
    "gpt-3.5-turbo-16k": ["0613"],
}  # UPDATE this when fixed version changed
MODEL_NAMES = set(MODEL_BASE_NAMES)
for base_name in MODEL_BASE_NAMES:
    for version in FIXED_VERSIONS.get(base_name, []):
        MODEL_NAMES.add(f"{base_name}-{version}")
# can be model_names or model_short_names
AVAILABLE_MODEL_NAMES = list(MODEL_NAMES | set(MODEL_SHORT_NAMES))


def llm_register_args(parser, prefix=None, shortprefix=None, defaults={}):
    model_name = defaults.get("model_name", "gpt-4")
    temperature = defaults.get("temperature", 0.0)
    max_tokens = defaults.get("max_tokens", None)
    default_retries = 8 if model_name.startswith("claude") else 12
    max_retries = defaults.get("max_retries", default_retries)
    request_timeout = defaults.get("request_timeout", 300)
    if prefix is None:
        prefix = ""
        shortprefix = ""
    else:
        prefix += "-"
        shortprefix = shortprefix or prefix[0]
    parser.add_argument(
        f"--{prefix}model-name",
        f"-{shortprefix}m",
        type=str,
        default=model_name,
        choices=AVAILABLE_MODEL_NAMES,
        help=f"Model name, one of: {AVAILABLE_MODEL_NAMES}",
    )
    parser.add_argument(
        f"--{prefix}temperature",
        f"-{shortprefix}t",
        type=float,
        default=temperature,
        help="Temperature for sampling",
    )
    parser.add_argument(
        f"--{prefix}max-tokens",
        f"-{shortprefix}mt",
        type=int,
        default=max_tokens,
        help="Max tokens for sampling",
    )
    parser.add_argument(
        f"--{prefix}max-retries",
        f"-{shortprefix}mr",
        type=int,
        default=max_retries,
        help="Max retries for each request",
    )
    parser.add_argument(
        f"--{prefix}request-timeout",
        f"-{shortprefix}rt",
        type=int,
        default=request_timeout,
        help="Timeout for each request",
    )


def get_model_name(name, fixed_version=True, version=None):
    """
    获取模型完整名称。支持预定义的简写映射，同时也支持任意自定义模型名称。
    
    对于预定义的简写（如 qwen、llama3 等），会映射到对应的完整版本。
    对于其他名称（如 qwen3-8B、custom-model 等），保持原样返回。
    """
    model_name = None
    if name in MODEL_NAMES_MAPPING:
        model_name = MODEL_NAMES_MAPPING[name]
    elif name in MODEL_NAMES:
        model_name = name
    else:
        # 支持任意自定义模型名称，不再抛出错误
        model_name = name

    if fixed_version:
        # 只对预定义的基础模型应用版本固定
        if version is None and model_name in FIXED_VERSIONS:
            version = FIXED_VERSIONS[model_name][0]
        if version:
            model_name += f"-{version}"
    
    # 移除对 MODEL_NAMES 的检查，支持任意模型名称
    print(model_name)
    return model_name


def get_fixed_model_name(name):
    return get_model_name(name, fixed_version=True)


class ChatOpenAI(LangchainChatOpenAI):
    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(
                    finish_reason=res.get("finish_reason"),
                ),
            )
            generations.append(gen)
        llm_output = {"token_usage": response["usage"], "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)


logger = logging.getLogger(__name__)


def _anthropic_create_retry_decorator(llm: ChatOpenAI) -> Callable[[Any], Any]:
    import anthropic

    min_seconds = 1
    max_seconds = 60
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(anthropic.APITimeoutError)
            | retry_if_exception_type(anthropic.APIError)
            | retry_if_exception_type(anthropic.APIConnectionError)
            | retry_if_exception_type(anthropic.RateLimitError)
            | retry_if_exception_type(anthropic.APIConnectionError)
            | retry_if_exception_type(anthropic.APIStatusError)
            | retry_if_exception_type(anthropic.InternalServerError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


class ChatAnthropic(LangchainChatAnthropic):
    max_retries: int = 6
    max_tokens_to_sample: int = 4000
    """Maximum number of retries to make when generating."""

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _anthropic_create_retry_decorator(self)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.completions.create(**kwargs)

        return _completion_with_retry(**kwargs)

    # note: we only rewrite the _generate method
    # todo: rewrite the _agenerate method
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params, **kwargs}
        if stop:
            params["stop_sequences"] = stop

        if self.streaming:
            completion = ""
            stream_resp = self.completion_with_retry(**params, stream=True)
            for data in stream_resp:
                delta = data.completion
                completion += delta
                if run_manager:
                    run_manager.on_llm_new_token(
                        delta,
                    )
        else:
            response = self.completion_with_retry(**params)
        return self._create_chat_result(response)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        finish_reason = response.stop_reason
        if finish_reason == "max_tokens":
            # for compatibility with OpenAI
            finish_reason = "length"

        # Note: claude only returns one generation
        generations = [
            ChatGeneration(
                message=AIMessage(content=response.completion),
                generation_info=dict(
                    finish_reason=finish_reason,
                ),
            )
        ]
        llm_output = {"model_name": response.model}
        return ChatResult(generations=generations, llm_output=llm_output)


def parse_llm_response(
    res: ChatResult, index: int = 0, one_generation_only: bool = True
) -> str:
    res = res.generations[0]
    if one_generation_only:
        assert len(res) == 1, res
    res = res[index]
    # if the decoding stops due to length, the response text is likely truncated so we discard it
    if res.generation_info.get("finish_reason", None) == "length":
        raise ValueError(f"Discard a response due to length: {res.text}")
    return res.text.strip()


def load_openai_llm(
    model_name: str = "text-davinci-003", **kwargs
) -> BaseLanguageModel:
    """Load the OpenAI language model and Claude and VLLM model"""
    # TODO: rename to load_llm
    if model_name.startswith("claude"):  # claude-1
        # TODO: set max_retry
        kwargs_map = {
            "request_timeout": "default_request_timeout",
            "max_tokens": "max_tokens_to_sample",
            "logit_bias": None,
            "n": None,
        }
        new_kwargs = {}
        for k, v in kwargs.items():
            if k in kwargs_map:
                if kwargs_map[k] is not None and v is not None:
                    new_kwargs[kwargs_map[k]] = v
                else:
                    print(f"Discard kwargs `{k}={v}` for {model_name}")
            else:
                new_kwargs[k] = v
        return ChatAnthropic(model=model_name, **new_kwargs)
    elif (
        model_name == "text-davinci-003" or model_name == "gpt-3.5-turbo-instruct-0914"
    ):
        return OpenAI(model_name=model_name, **kwargs)
    elif "vicuna" in model_name:
        return VLLM(
            model=model_name,
            trust_remote_code=True,
            max_num_batched_tokens=16000,
            **kwargs,
        )
    elif model_name.startswith("transformers/"):
        import torch
        path = model_name[len("transformers/"):]  # 提取模型路径
        print("path",path)
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(path)
        client = pipeline("text-generation", model=model, tokenizer=tokenizer)

        return TransformersLLM(
            client=client,
            model=path.split("/")[-1],
            max_new_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.1),
        )
    elif model_name in MODEL_NAMES:
        return ChatOpenAI(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def load_openai_llm_with_args(args, prefix=None, fixed_version=True, **kwargs):
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"
    model_name = getattr(args, f"{prefix}model_name")
    temperature = getattr(args, f"{prefix}temperature")
    max_tokens = getattr(args, f"{prefix}max_tokens")
    max_retries = getattr(args, f"{prefix}max_retries")
    request_timeout = getattr(args, f"{prefix}request_timeout")

    if model_name.startswith("transformers/"):
        # 不使用 OpenAI Key，本地模型不需要 token
        return load_openai_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            request_timeout=request_timeout,
            **kwargs,
        )
    ## change openai API key and api_base from .env ##
    if model_name.startswith("gpt-3.5"):
        os.environ["OPENAI_API_BASE"] = os.getenv("GPT35_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        os.environ["OPENAI_API_KEY"] = os.getenv("GPT35_API_KEY", os.getenv("OPENAI_API_KEY"))
    elif model_name.startswith("gpt-4"):
        os.environ["OPENAI_API_BASE"] = os.getenv("GPT4_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        os.environ["OPENAI_API_KEY"] = os.getenv("GPT4_API_KEY", os.getenv("OPENAI_API_KEY"))
    elif model_name.startswith("llama"):
        os.environ["OPENAI_API_BASE"] = os.getenv("LLAMA_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        os.environ["OPENAI_API_KEY"] = os.getenv("LLAMA_API_KEY", os.getenv("OPENAI_API_KEY"))
    elif model_name.startswith("qwen"):
        os.environ["OPENAI_API_BASE"] = os.getenv("QWEN_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        os.environ["OPENAI_API_KEY"] = os.getenv("QWEN_API_KEY", os.getenv("OPENAI_API_KEY"))


    return load_openai_llm(
        model_name=get_model_name(model_name, fixed_version=fixed_version),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        request_timeout=request_timeout,
        **kwargs,
    )


def get_model_category(llm: BaseLanguageModel):
    if isinstance(llm, ChatOpenAI):
        return "openai"
    elif isinstance(llm, OpenAI):
        return "openai-text"
    elif isinstance(llm, ChatAnthropic):
        return "claude"
    elif isinstance(llm, VLLM):
        return "vicuna"
    elif isinstance(llm, TransformersLLM):
        return "transformers"
    else:
        raise ValueError(f"Unknown model type: {type(llm)}")
