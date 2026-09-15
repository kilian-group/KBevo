from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

from llm.base import LLM, LLMResponse


ActionType = Literal["generate", "toolcall", "finish"]

@dataclass
class AgentStep:
    prompt: str
    answer: Optional[str]
    action: ActionType
    error: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    golden_triplets: Optional[str] = None


class Agent:
    def __init__(self, llm: LLM, max_steps: int = 8,  **kwargs,) -> None:
        self.llm = llm
        self.max_steps = max_steps
        self.trace: List[AgentStep] = []

    # ----- overridable hooks -----
    def build_prompt(self, query: str) -> str:
        history = "\n".join(
            f"Step {i + 1} [{s.action}]: {s.answer or s.error or ''}".strip()
            for i, s in enumerate(self.trace)
        )
        return f"{history}\n\nQuestion: {query}".strip()
    
    def build_query(self, question : str) -> str:
        """Instruction to ensure the Agent emits a FINAL_ANSWER the parser recognizes."""
        instruction = (
            "Provide only the final answer prefixed by 'FINAL_ANSWER:' with no extra text."
        )
        return f"{instruction}\n{question}"

    def parse_action(
        self, resp: LLMResponse
    ) -> Tuple[ActionType, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """
        Returns (action, tool_name, tool_args, final_answer_if_finish).
        Base behavior:
        - If 'FINAL_ANSWER:' present, treat as finish and extract trailing text.
        - Otherwise 'generate'. Subclasses can emit 'toolcall'.
        """
        text = resp.text or ""
        if "FINAL_ANSWER:" in text:
            final = text.split("FINAL_ANSWER:", 1)[1].strip()
            return "finish", None, None, final
        return "generate", None, None, None

    def on_toolcall(self, step: AgentStep) -> None:
        # Base agent does not execute tools. ReAct-style subclasses should override.
        pass

    # ----- core API -----
    def step(self, query: str, **llm_kwargs: Any) -> AgentStep:
        prompt = self.build_prompt(query)
        try:
            resp = self.llm.run(prompt, **llm_kwargs)
            action, tool_name, tool_args, final_answer = self.parse_action(resp)
            if action == "finish":
                step = AgentStep(prompt=prompt, answer=final_answer, action="finish")
            elif action == "toolcall":
                step = AgentStep(
                    prompt=prompt,
                    answer=resp.text,
                    action="toolcall",
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
                self.on_toolcall(step)
            else:
                step = AgentStep(prompt=prompt, answer=resp.text, action="generate")
        except Exception as e:
            step = AgentStep(prompt=prompt, answer=None, action="finish", error=str(e))
        return step

    def run(self, question: list[str] | str, **llm_kwargs: Any) -> Tuple[Optional[str], List[AgentStep]]:
        if isinstance(question, list):
            if len(question) > 1:
                raise NotImplementedError("Agent class does not support batch inference. Please set the batch size to 1.")
            question = question[0]

        query = self.build_query(question)
        trace = []
        final_answer: Optional[str] = None
        for _ in range(self.max_steps):
            step = self.step(query, **llm_kwargs)
            # Update self.trace so build_prompt() can access step history
            self.trace.append(step)
            if step.action == "finish":
                final_answer = step.answer
                break
        # Return a copy of the trace for the caller
        return [final_answer], [list(self.trace)]

