import os

# Get package root directory (2 levels up from this file)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define paths relative to root. These are used as read locations by the
# annotator and legacy SFT-dataset loader; do NOT auto-create them at import
# time — that leaves empty ghost directories behind on every Python invocation.
# Real prompt JSONs live under `data/prompts/` and `src/multi_lmlm/prompts/`.
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


# --- Special token format ---
DB_START_TOKEN = "<|db_entity|>"          # Begins a lookup call
DB_SEP_TOKEN = "<|db_relationship|>"                # Separates entity and relation in the query
DB_RETRIEVE_TOKEN = "<|db_return|>"   # Signals insertion point for returned value
DB_END_TOKEN = "<|db_end|>"            # Marks end of lookup block

# New format experiment, the model is allowed to look up all relatinoships for an entity.
DB_ALL_RELATIONSHIPS_TOKEN="<|db_all_relationships|>" # Signals retrieval of all relationships for an entity


# --- Legacy format (used for annotation) ---
LEGACY_DB_START_TOKEN = "[dblookup"
LEGACY_DB_SEP_TOKEN = "', '"
LEGACY_DB_RETRIEVE_TOKEN = "') -> "
LEGACY_DB_END_TOKEN = "]"

TINY_LLAMA2_TOKENIZER_PATH = "./tokenizer/tiny-llama2"

#multi-hop tokens
ANSWER_START_TOKEN="<answer>"
ANSWER_END_TOKEN="</answer>"
THINKING_END_TOKEN="</thinking>"
THINKING_START_TOKEN="<thinking>"
