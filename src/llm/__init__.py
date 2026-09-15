"""LLM factory utilities."""

from typing import Any

from .base import LLM
from .openai import OpenAILLM
from .llama import LlamaLLM
from .qwen import QwenLLM


def get_llm(model_name: str, **kwargs: Any) -> LLM:
    """Return an LLM instance for the given model name.

    Supported models:
    - GPT models (contains "gpt"): OpenAILLM
    - Llama models (contains "llama"): LlamaLLM
    - Qwen models (contains "qwen"): QwenLLM
    """
    model_name_lower = model_name.lower()

    if "gpt" in model_name_lower:
        return OpenAILLM(model_name=model_name, **kwargs)
    elif "llama" in model_name_lower:
        return LlamaLLM(model_name=model_name, **kwargs)
    elif "qwen" in model_name_lower:
        return QwenLLM(model_name=model_name, **kwargs)

    raise ValueError(
        f"Unsupported model_name '{model_name}'. "
        "Supported models: GPT models (gpt-*), Llama models (llama-*), or Qwen models (qwen*)."
    )
