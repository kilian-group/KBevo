"""Agent factory utilities."""

from typing import Any, Dict

from .agent_class import Agent
from .rag_agent import RAGAgent
from .icl_agent import ICLAgent
from .cot_agent import CotAgent
from .lmlm_agent import LMLMAgent
from .two_phase_agent import TwoPhaseAgent


def get_agent(method: str, agent_kwargs: Dict[str, Any]) -> Agent:
    """Return an Agent instance for the given method.

    Args:
        method: Agent method type ("rag", "icl", "cot", "lmlm", "two_phase")
        agent_kwargs: Dictionary containing agent-specific parameters

    Returns:
        Agent instance of the appropriate type

    Raises:
        NotImplementedError: For unsupported method types or configurations
        Exception: For missing required parameters
    """
    agent: Agent

    match method:
        case "icl":
            # Extract parameters
            dataset = agent_kwargs["dataset"]
            setting = agent_kwargs["setting"]

            # Guard against unsupported setting
            if dataset == "hotpotqa" and setting == "fullwiki":
                raise NotImplementedError("ICL is not supported for --setting fullwiki.")

            # Create ICL agent with empty contexts (will be reset per question)
            agent = ICLAgent(
                **agent_kwargs
            )

        case "rag":
            # Extract parameters
            retrieval = agent_kwargs.get("retrieval", "bm25")


            # Guard against unsupported combinations (we only support bm25 + distractor for now)
            if retrieval != "bm25":
                raise NotImplementedError("Only --retrieval bm25 is supported currently.")

            # Create RAG agent with empty contexts (will be reset per question)
            agent = RAGAgent(
                **agent_kwargs,
            )
        case "lmlm":
            # Extract parameters
            model_path = agent_kwargs.get("model_path")
            database_path = agent_kwargs.get("database_path")
            if model_path is None:
                raise Exception("You must set a local model path for lmlm setting")
            if database_path is None:
                raise Exception("You must set a local database path for lmlm setting")

            agent = LMLMAgent(**agent_kwargs)

        case "cot":
            llm = agent_kwargs.get("llm")
            if llm is None:
                raise Exception("You must set an LLM for CoT inference.")
            agent = CotAgent(llm=llm, max_steps=agent_kwargs.get("max_steps", 8))

        case "direct":
            llm = agent_kwargs.get("llm")
            if llm is None:
                raise Exception("You must set an LLM for direct inference.")
            agent = Agent(llm=llm, max_steps=agent_kwargs.get("max_steps", 8))

        case "two_phase":
            model_path = agent_kwargs.get("model_path")
            if model_path is None:
                raise Exception("You must set --model-path for two_phase method.")
            agent = TwoPhaseAgent(**agent_kwargs)

        case _:
            raise NotImplementedError(f"Method '{method}' is not implemented.")

    return agent


__all__ = ["Agent", "RAGAgent", "ICLAgent", "CotAgent", "LMLMAgent", "TwoPhaseAgent", "get_agent"]
