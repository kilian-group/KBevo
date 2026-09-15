"""Llama LLM adapter using transformers library."""

import time
from typing import Any, Dict, List, Optional

from .base import LLM, LLMResponse

# Importing here so static analysis works; runtime requires transformers and torch
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError(
        "transformers and torch are required for LlamaLLM. "
        "Install with: pip install transformers torch"
    )


# Model name mapping to HuggingFace model IDs
MODEL_MAP = {
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
}


class LlamaLLM(LLM):
    """Llama adapter using transformers library for local inference."""

    def __init__(
        self,
        model_name: str = "llama-3-8b-instruct",
        timeout_s: float = 60.0,
        max_retries: int = 2,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, timeout_s=timeout_s, max_retries=max_retries)

        # Map model name to HuggingFace model ID
        hf_model_id = MODEL_MAP.get(model_name.lower())
        if not hf_model_id:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. "
                f"Supported models: {list(MODEL_MAP.keys())}"
            )
        self.hf_model_id = hf_model_id

        # Device configuration
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Set default torch dtype
        if dtype is None:
            if device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
        )

        # Load model
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": self.dtype,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
            self.use_device_map = True
        else:
            model_kwargs["device_map"] = device
            self.use_device_map = False

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            **model_kwargs,
        )
        self.model.eval()  # Set to evaluation mode

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
        """Generate a response from the Llama model."""
        # Format prompt using chat template for instruct models
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: just use the prompt as-is
            formatted_prompt = prompt

        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
        )
        # Only move to device if not using device_map
        if not self.use_device_map:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Prepare generation parameters
        generation_kwargs: Dict[str, Any] = {
            "temperature": (
                temperature if temperature > 0 else 1.0
            ),  # Transformers uses 1.0 for greedy
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if max_tokens is not None:
            generation_kwargs["max_new_tokens"] = max_tokens

        if stop is not None:
            # For stop sequences, we'll use the stop_strings approach
            # Note: transformers doesn't natively support multi-token stop sequences
            # We'll handle this by checking during generation or using a custom stopping criteria
            # For now, we'll add stop_token_ids as a fallback
            stop_token_ids = set()
            for stop_seq in stop:
                # Encode stop sequence and get token IDs
                stop_tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                if stop_tokens:
                    stop_token_ids.update(stop_tokens)
            if stop_token_ids:
                # Add to existing eos_token_id if present
                existing_eos = generation_kwargs.get("eos_token_id", [])
                if isinstance(existing_eos, int):
                    existing_eos = [existing_eos]
                elif existing_eos is None:
                    existing_eos = []
                generation_kwargs["eos_token_id"] = list(
                    set(list(existing_eos) + list(stop_token_ids))
                )

        # Allow extra parameters to override defaults
        if extra:
            generation_kwargs.update(extra)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs,
                    )

                # Decode only the generated tokens (exclude input)
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                # Calculate token usage
                prompt_tokens = inputs["input_ids"].shape[1]
                completion_tokens = len(generated_tokens)
                total_tokens = prompt_tokens + completion_tokens

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.timeout_s:
                    raise TimeoutError(
                        f"Generation took {elapsed:.2f}s, exceeding timeout of {self.timeout_s}s"
                    )

                # Determine finish reason
                finish_reason = "stop"
                if max_tokens is not None and completion_tokens >= max_tokens:
                    finish_reason = "length"

                return LLMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    finish_reason=finish_reason,
                    raw={"model": self.model_name, "generation_kwargs": generation_kwargs},
                )
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    raise
                # Exponential backoff with jitter
                time.sleep((2**attempt) * 0.5)

        # Should not reach here; if we do, raise the last captured error
        if last_err is not None:
            raise last_err
        raise RuntimeError("LlamaLLM.run failed without an exception")
