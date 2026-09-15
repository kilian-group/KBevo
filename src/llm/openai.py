"""OpenAI LLM adapter using the Responses API."""
import os
import time
from typing import Any, Dict, List, Optional

from .base import LLM, LLMResponse

# Importing here so static analysis works; runtime requires openai>=1.0.0
from openai import OpenAI, APIError, RateLimitError, APITimeoutError


class OpenAILLM(LLM):
    """OpenAI adapter backed by the Responses API."""

    def __init__(self, model_name: str = "gpt-4", timeout_s: float = 60.0, max_retries: int = 2):
        super().__init__(model_name=model_name, timeout_s=timeout_s, max_retries=max_retries)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment")

        base_url = os.environ.get("OPENAI_BASE_URL") or None
        self._client = OpenAI(api_key=api_key, base_url=base_url)

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
        params: Dict[str, Any] = {
            "model": self.model_name,
            "input": prompt,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            # Responses API uses max_output_tokens
            params["max_output_tokens"] = max_tokens
        if stop is not None:
            params["stop"] = stop
        if extra:
            # Allow caller to override/add provider-specific params
            params.update(extra)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.responses.create(**params, timeout=self.timeout_s)

                # Prefer convenience property if available
                try:
                    text = resp.output_text  # type: ignore[attr-defined]
                except Exception:
                    text = ""
                    # Fallback: concatenate output_text chunks if present
                    output = getattr(resp, "output", None)
                    if output:
                        fragments: List[str] = []
                        for item in output:
                            if getattr(item, "type", None) == "output_text":
                                fragments.append(getattr(item, "content", ""))
                        text = "".join(fragments)

                usage = getattr(resp, "usage", None)
                prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
                completion_tokens = getattr(usage, "output_tokens", None) if usage else None
                total_tokens = getattr(usage, "total_tokens", None) if usage else None

                # Finish reason best-effort (may not always be present in Responses API)
                finish_reason = getattr(resp, "stop_reason", None)

                raw = resp.model_dump() if hasattr(resp, "model_dump") else None

                return LLMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    finish_reason=finish_reason,
                    raw=raw,
                )
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_err = e
                if attempt >= self.max_retries:
                    raise
                # Exponential backoff with jitter
                time.sleep((2 ** attempt) * 0.5)

        # Should not reach here; if we do, raise the last captured error
        if last_err is not None:
            raise last_err
        raise RuntimeError("OpenAILLM.generate failed without an exception")


