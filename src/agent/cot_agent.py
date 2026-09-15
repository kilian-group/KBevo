from typing import List

from agent.agent_class import Agent, LLM


def _join_evidence(docs: List[str]) -> str:
    return "\n================\n\n".join(docs)


class CotAgent(Agent):
    """
    Chain-of-Thought Agent.
    - Includes all available context paragraphs (title + article) in the prompt
    - Prompts the model to reason step-by-step before giving the final answer
    """

    def __init__(
        self,
        llm: LLM,
        contexts: List[str] = [],
        max_steps: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(llm=llm, max_steps=max_steps)
        self.contexts = contexts or []

    def reset(self, contexts: List[str]) -> None:
        """Reset agent state for a new question with new contexts."""
        self.contexts = contexts or []
        self.trace = []

    def build_query(self, question: str) -> str:
        instruction = (
            "Think step by step to answer the question. "
            "At the end, prefix your final answer with 'FINAL_ANSWER:'."
        )
        return f"{instruction}\n{question}"

    def build_prompt(self, query: str | list[str]) -> str:
        if isinstance(query, list):
            if len(query) > 1:
                raise NotImplementedError("Agent class does not support batch inference. Please set the batch size to 1.")
            query = query[0]

        history = "\n".join(
            f"Step {i + 1} [{s.action}]: {s.answer or s.error or ''}".strip() for i, s in enumerate(self.trace)
        )
        evidence_block = _join_evidence(self.contexts) if self.contexts else ""
        if history:
            return f"Evidence:\n{evidence_block}\n\n{history}\n\nQuestion: {query}".strip()
        else:
            return f"Evidence:\n{evidence_block}\n\nQuestion: {query}".strip()
