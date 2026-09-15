from typing import Any, Dict, List, Optional, Tuple

from agent.agent_class import Agent, AgentStep, LLM, LLMResponse
from tools.retrieval import BaseRetriever, FlashRAGBM25Retriever, FlashRAGBM25CorpusRetriever


def _doc_to_text(doc: Any) -> str:
    """Render retrieved evidence (dict or string) into a prompt-friendly string."""
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        contents = str(doc.get("contents", "")).strip()
        if title and contents:
            return f"{title}: {contents}".strip()
        return contents or title
    return str(doc)


def _join_evidence(docs: List[Any]) -> str:
    return "\n================\n\n".join(_doc_to_text(doc) for doc in docs)


class RAGAgent(Agent):
    """
    Retrieval-Augmented Agent.
    - Retrieves top-k context paragraphs (title + article) for a given question
    - Prepends an 'Evidence' block to the prompt before the question
    """

    def __init__(
        self,
        llm: LLM,
        retriever_type: str = "bm25",
        contexts: List[Any] = [],
        corpus: Optional[List[Any]] = None,
        rag_k: int = 4,
        max_steps: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(llm=llm, max_steps=max_steps)
        # Initialize retriever by type
        match (retriever_type or "").lower():
            case "bm25":
                if corpus:
                    self.retriever = FlashRAGBM25CorpusRetriever(corpus)
                else:
                    self.retriever = FlashRAGBM25Retriever()
            case _:
                raise NotImplementedError(f"Retriever type '{retriever_type}' is not implemented.")
        self.contexts = contexts or []
        self._corpus = corpus or []
        self.rag_k = rag_k
        self._evidence_docs: List[Any] = []

    def gather_evidence(self, query: str) -> None:
        """Retrieve evidence once per question to avoid repeated indexing."""
        if self._evidence_docs:
            return
        documents = self._corpus if self._corpus else self.contexts
        self._evidence_docs = self.retriever.retrieve(
            query=query,
            documents=documents,
            top_k=self.rag_k,
        )

    def build_prompt(self, query: str) -> str:
        # Build step history (same style as base Agent)
        history = "\n".join(
            f"Step {i + 1} [{s.action}]: {s.answer or s.error or ''}".strip() for i, s in enumerate(self.trace)
        )
        evidence_block = _join_evidence(self._evidence_docs) if self._evidence_docs else ""
        if history:
            return f"Evidence:\n{evidence_block}\n\n{history}\n\nQuestion: {query}".strip()
        else:
            return f"Evidence:\n{evidence_block}\n\nQuestion: {query}".strip()

    def reset(self, contexts: List[Any]) -> None:
        """Reset agent state for a new question with new contexts."""
        if not self._corpus:
            self.contexts = contexts or []
        self._evidence_docs = []
        self.trace = []

    # NOTE: We compute evidence once per run to avoid repeated indexing within multi-step loops
    def run(self, question: str | list[str], **llm_kwargs: Any) -> Tuple[Optional[str], List[AgentStep]]:
        if isinstance(question, list):
            if len(question) > 1:
                print("DEBUG: ", question)
                raise NotImplementedError("RAGAgent does not support batch inference. Please set the batch size to 1.")

        query = self.build_query(question)
        self._evidence_docs = []
        self.gather_evidence(query)
        return super().run(query, **llm_kwargs)
