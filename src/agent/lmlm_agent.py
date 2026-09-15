import json
from agent.agent_class import Agent, AgentStep
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from multi_lmlm.database.database_manager import DatabaseManager
from transformers import LogitsProcessor
from multi_lmlm.constants import DB_END_TOKEN, ANSWER_START_TOKEN, DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, ANSWER_END_TOKEN
import os
from vllm import LLM, SamplingParams


def _decode_with_special_tokens(outputs, tokenizer, input_len, input_text):
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if input_text in output_text:
        output_text = output_text.split(input_text)[-1]
    else:
        output_text = tokenizer.decode(outputs[0][input_len:], clean_up_tokenization_spaces=True) 
    return output_text  

class LogitBiasProcessor(LogitsProcessor):
    def __init__(self, bias_dict: dict):
        """
        bias_dict: {token_id: bias_value (positive = more likely)}
        """
        super().__init__()
        self.bias_dict = bias_dict

    def __call__(self, input_ids, scores):
        for token_id, bias in self.bias_dict.items():
            scores[:, token_id] += bias
        return scores

class LMLMAgent(Agent):
    def __init__(self, model_path: str, database_path: str | None = None, similarity_threshold = 0.6, adaptive : bool = False, top_k : int = 4, return_triplets : bool = False, use_inverses : bool = False,
                 top_p : float = 1.0, vllm_top_k : int = 0, repetition_penalty : float = 1.0, max_model_len : int | None = None, **kwargs ):
        self.model_path = model_path
        # vLLM sampling params. Defaults reproduce the historical greedy setup
        # (top_p=1.0, top_k=0 → disabled); set them to match training sampling.
        self.top_p = top_p
        self.vllm_top_k = vllm_top_k
        self.repetition_penalty = repetition_penalty
        self.database_path = database_path
        self.top_k = top_k
        metadata_file = os.path.join(os.path.dirname(database_path), "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

        if use_inverses:
            print("USING INVERSES WOOHOO")

        self.return_triplets = return_triplets

        self.db = DatabaseManager()
        self.db.load_database(database_path, top_k=top_k, default_threshold=similarity_threshold, adaptive=adaptive, use_inverses=use_inverses)
        self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path)

        self.similarity_threshold = similarity_threshold

        # Initialize vLLM for batch generation
        self.stop_token_ids = [self.tok.eos_token_id, self.tok.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]]
        self.db_retrieve_token_id = self.tok.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]
        self.answer_end_token_id = self.tok.encode(ANSWER_END_TOKEN, add_special_tokens=False)[0]


        # Add validation in __init__
        print(f"EOS token ID: {self.tok.eos_token_id}")
        print(f"DB_RETRIEVE_TOKEN ID: {self.tok.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]}")
        print(f"Vocab size: {len(self.tok)}")

        # Ensure they're within vocab bounds
        def check_token_in_vocab(tokenizer):
            special_tokens = [DB_START_TOKEN, DB_END_TOKEN, DB_RETRIEVE_TOKEN, 
                  ANSWER_START_TOKEN, ANSWER_END_TOKEN, DB_SEP_TOKEN]
            for token in special_tokens:
                encoded = tokenizer.encode(token, add_special_tokens=False)
                print(f"{token}: {encoded}")
                assert len(encoded) > 0, f"Token {token} not in vocabulary"

        check_token_in_vocab(self.tok)

        _llm_kwargs = {}
        if max_model_len is not None:
            _llm_kwargs["max_model_len"] = max_model_len
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            seed=42,
            tokenizer=model_path,
            **_llm_kwargs,
        )
        check_token_in_vocab(self.llm.get_tokenizer())

        self.max_turns = 16

    def create_prompt_from_query(self, query):
        return f"Question:\n{query}\nAnswer:\n"
    
    def create_prompt_from_query_batch(self, queries : list[str]):
        return [self.create_prompt_from_query(query) for query in queries]


    def run(self, queries: list[str], indices: list[int] | None = None, max_tokens=256, temperature=0.0):
        """
        Run batch inference using vLLM with database lookups.

        Args:
            queries: List of questions to answer
            indices: List of indices corresponding to each query (for metadata lookup)
            max_tokens: Maximum tokens to generate per turn before hitting a stop token
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            List of (answer, trace) tuples, one per query
        """
        # Create initial prompts for all queries
        prompts = self.create_prompt_from_query_batch(queries)

        # Track which queries are still generating
        active = [True] * len(queries)
        results = [(None, None) for _ in range(len(queries))]
        self._lookup_logs = [[] for _ in range(len(queries))]

        # Generation loop - continue until all queries complete
        # Max iterations to prevent infinite loops (in case of malformed outputs)
        
        # BUG: potential bug here. need to check the prompt length each turn to make sure it doesn't exceed the max model length
        for turn in range(self.max_turns):
            # Only generate for active queries
            active_prompts = [p for i, p in enumerate(prompts) if active[i]]
            if not active_prompts:
                break
    
            # DEBUG: Check prompt lengths
            for idx, prompt in enumerate(active_prompts):
                prompt_len = len(self.tok.encode(prompt))
                max_len = self.llm.llm_engine.model_config.max_model_len
                if prompt_len + max_tokens > max_len:
                    print(f"Warning: Prompt {idx} length {prompt_len} + max_tokens {max_tokens} exceeds max {max_len}")
                    # Either truncate or mark as inactive
                    active[idx] = False
                    continue

            # Setup sampling parameters with stop tokens
            # vLLM will automatically stop at DB_RETRIEVE_TOKEN or EOS
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=self.top_p,
                top_k=self.vllm_top_k,
                repetition_penalty=self.repetition_penalty,
                max_tokens=max_tokens,  # Generate up to max_tokens or until stop token
                stop_token_ids=self.stop_token_ids,
                # logprobs=0, # help solve the bug of illegal cuda memory access
            )

            # Generate for all active prompts
            outputs = self.llm.generate(active_prompts, sampling_params=sampling_params, use_tqdm=False)

            # Process outputs and update prompts
            active_idx = 0
            for i in range(len(queries)):
                if not active[i]:
                    continue

                output = outputs[active_idx]
                active_idx += 1

                # Extract generated text
                if len(output.outputs) > 0 and len(output.outputs[0].token_ids) > 0:
                    generated_tokens = output.outputs[0].token_ids
                    generated_text = self.tok.decode(generated_tokens, skip_special_tokens=False)
                    prompts[i] += generated_text
                else:
                    # No tokens generated, likely hit stop token immediately
                    generated_text = ""

                # Check if this query has completed
                if ANSWER_END_TOKEN in prompts[i]:
                    active[i] = False
                    try:
                        answer = prompts[i].split(ANSWER_START_TOKEN)[1].split(ANSWER_END_TOKEN)[0]
                        if self.metadata and indices and indices[i] < len(self.metadata):
                            golden_triplets = ", ".join(
                                f"({entity}, {rel}, {val})"
                                for (entity, rel, val) in self.metadata[indices[i]]["triplets"]
                            )
                        else:
                            golden_triplets = 'No metadata provided'
                        trace = [AgentStep(prompts[i], answer, "generate", golden_triplets=golden_triplets)]
                        results[i] = (answer, trace)
                    except Exception as e:
                        results[i] = ("", [AgentStep(prompts[i], "", "generate")])
                    continue

                # Check if we need to perform database lookup
                if DB_RETRIEVE_TOKEN in generated_text:
                    # Extract database query
                    return_value = "unknown"
                    lookup_log = {
                        "query": None,
                        "success": False,
                        "returned_count": 0,
                        "error": None,
                    }
                    try:
                        split = prompts[i].rsplit(DB_START_TOKEN)
                        db_query = split[-1]
                        lookup_log["query"] = db_query
                        return_values = self.db.retrieve_from_database(
                            DB_START_TOKEN + db_query,
                            self.similarity_threshold,
                            return_triplets=self.return_triplets,
                            top_k=self.top_k
                        )
                        lookup_log["returned_count"] = len(return_values)
                        lookup_log["success"] = len(return_values) > 0
                        return_value = ", ".join(return_values)
                    except Exception as e:
                        lookup_log["error"] = str(e)
                    self._lookup_logs[i].append(lookup_log)

                    # Append retrieved value and db_end token
                    prompts[i] += return_value + DB_END_TOKEN

        # Finalize any queries that didn't complete
        for i in range(len(queries)):
            if results[i][0] is None:
                results[i] = ("", [AgentStep(prompts[i], "", "generate")])

        answers = [r[0] for r in results]
        traces = [r[1] for r in results]
        return answers, traces

if __name__ == '__main__':
    # Manual smoke: set KBEVO_AGENT_MODEL (and optionally KBEVO_AGENT_DB) to
    # a local checkpoint directory or a Hugging Face repo id, e.g.
    #   KBEVO_AGENT_MODEL=kilian-group/KBevo-Qwen3-1.7B-GRPO \
    #   python -m src.agent.lmlm_agent
    import os
    model = os.environ.get("KBEVO_AGENT_MODEL")
    if not model:
        raise SystemExit(
            "Set KBEVO_AGENT_MODEL to a local checkpoint dir or HF repo id "
            "(e.g. kilian-group/KBevo-Qwen3-1.7B-GRPO)."
        )
    agent = LMLMAgent(model_path=model,
                      database_path=os.environ.get("KBEVO_AGENT_DB"),
                      use_inverses=True, return_triplets=True)
    for _ in range(5):
        results = agent.run(["Walter Piston studied composition with"], 0)
        print("results :\n\n", results)
    



