""" Base class for LLM """
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class LLM(ABC):
    """ Base class for LLM """

    def __init__(self, model_name: str, timeout_s: float = 60.0, max_retries: int = 2):
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @abstractmethod
    def run(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Generate a response from the LLM. Implement in provider subclasses."""
        raise NotImplementedError