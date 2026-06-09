from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.llms.openai import BaseOpenAI
from langchain.pydantic_v1 import root_validator
from langchain.schema.output import Generation, LLMResult

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    set_seed,
)


class TransformersLLM(BaseLLM):
    """HuggingFace Transformers-based LLM wrapper for LangChain."""

    model: str = ""
    trust_remote_code: Optional[bool] = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    do_sample: bool = True
    seed: Optional[int] = None

    client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Load model and tokenizer using HuggingFace Transformers if not already provided."""

        if values.get("client") is None:
            try:
                model_path = values["model"]  # 模型路径应该由外部传入
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=values["trust_remote_code"]
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=values["trust_remote_code"]
                ).to(values["device"])

                if values.get("seed") is not None:
                    set_seed(values["seed"])

                # 创建pipeline
                pipeline = TextGenerationPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if values["device"] == "cuda" else -1
                )
                values["client"] = pipeline
            except Exception as e:
                raise ValueError(f"Failed to load model: {e}")
        return values

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """Generate outputs from the model."""

        generations = []
        print(prompts)
        for prompt in prompts:
            print("prompt:", prompt)
            outputs = self.client(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.do_sample,
                eos_token_id=None if self.stop else self.client.tokenizer.eos_token_id,
                return_full_text=False,
                **kwargs,
            )
            text = outputs[0]["generated_text"]
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "transformers"
