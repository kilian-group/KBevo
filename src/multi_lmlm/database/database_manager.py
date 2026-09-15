from __future__ import annotations

import os
import re
import json
import time
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from collections import Counter
from datasets import DatasetDict
from multi_lmlm.database.topk_retriever import TopkRetriever
from sentence_transformers import SentenceTransformer

# Setup logger
logger = logging.getLogger(__name__)


def _deduplicate_triplets(triplets: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Return triplets in first-seen order with duplicates removed."""
    return list(dict.fromkeys(tuple(triplet) for triplet in triplets))


def _add_inverse_triplets(triplets: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Add ``(value, relationship, entity)`` triplets without duplicates."""
    unique_triplets = _deduplicate_triplets(triplets)
    inverse_triplets = [
        (return_value, relationship, entity)
        for entity, relationship, return_value in unique_triplets
    ]
    return _deduplicate_triplets(unique_triplets + inverse_triplets)


def _triplet_search_text(triplet: tuple[str, str, str]) -> str:
    """Build the normalized entity-relation text embedded by TopkRetriever."""
    entity, relationship, _ = triplet
    return (
        f"{TopkRetriever._normalize_text(entity)} "
        f"{TopkRetriever._normalize_text(relationship)}"
    )

def extract_lookups(text, pattern=None):
    """Extract (entity, attribute, answer) triplets from annotated text."""
    if pattern is None:
        pattern_lst = [
            r'\[dblookup\(([^,]+),\s*([^,]+)\)\s*->\s*(.*?)\]',
            r"\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->\s*(.+?)\]",
            r"\s?<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>(.+?)<\|db_end\|>",
        ]

    matches = []
    for p in pattern_lst:
        matches.extend(re.findall(p, text, re.DOTALL))
    matches = list({tuple(match) for match in matches})

    return [[item.strip("'").strip('"').strip() for item in match] for match in matches]

def update_atomic_knowledge(example, current_version=1):
    triplets = extract_lookups(example["annotated_text"])
    example.update({"atomic_knowledge": triplets})
    return example

def extract_database(dataset):
    if "atomic_knowledge" not in dataset.column_names:
        dataset = dataset.map(
            update_atomic_knowledge,
            batched=False,
            fn_kwargs={"current_version": 1},
            num_proc=1,  # optional parallelism
            desc="Extracting atomic knowledge",
        )
    return dataset

class DatabaseLookupError(Exception):
    """Custom exception for database lookup failures, tracking failure statistics."""
    failure_statistics = Counter()

    def __init__(self, message, failure_type):
        super().__init__(message)
        self.__class__.failure_statistics[failure_type] += 1

    @classmethod
    def get_failure_statistics(cls):
        return dict(cls.failure_statistics)
    

    
def build_databases_from_triplets_batch(triplets_batch: list[list[tuple[str, str, str]]], top_k: int = 4, default_threshold: float = 0.6, adaptive: bool = False, use_inverses: bool = False, batch_size: int = 2048) -> list["DatabaseManager"]:
    """Build multiple DatabaseManagers from a batch of triplet lists, computing embeddings once.

    Args:
        triplets_batch: List of triplet lists, one per example
        top_k: Number of top results to retrieve
        default_threshold: Similarity threshold for retrieval
        adaptive: Whether to use adaptive threshold
        use_inverses: Whether to include inverse relationships
        batch_size: Batch size for embedding encoding

    Returns:
        List of DatabaseManager instances, one per input triplet list
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device, local_files_only=True)

    # Inverse entries need their own (value, relationship) embeddings. Augment
    # each example before flattening so all embeddings are still computed in a
    # single model call and split at the correct per-example boundaries.
    database_triplets_batch = [
        _add_inverse_triplets(triplet_list) if use_inverses else [tuple(t) for t in triplet_list]
        for triplet_list in triplets_batch
    ]

    # Flatten all triplets and compute embeddings at once
    combined_triplets = [t for triplet_list in database_triplets_batch for t in triplet_list]

    if not combined_triplets:
        logger.warning("No triplets provided to build_databases_from_triplets_batch")
        return [DatabaseManager() for _ in triplets_batch]

    logger.info(f"Encoding {len(combined_triplets)} triplets from {len(triplets_batch)} examples in batch...")
    texts = [_triplet_search_text(triplet) for triplet in combined_triplets]
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Split embeddings back to per-example chunks
    database_managers = []
    start_idx = 0

    for triplet_list in database_triplets_batch:
        end_idx = start_idx + len(triplet_list)
        example_embeddings = embeddings[start_idx:end_idx]

        # Create DatabaseManager for this example
        db_manager = DatabaseManager()
        db_manager.build_database_from_triplets_with_embeddings(
            triplets=triplet_list,
            embeddings=example_embeddings,
            top_k=top_k,
            default_threshold=default_threshold,
            adaptive=adaptive,
            # Triplets and embeddings were augmented together above.
            use_inverses=False,
            embedding_model = embedding_model
        )
        database_managers.append(db_manager)

        start_idx = end_idx

    logger.info(f"Built {len(database_managers)} databases from batch")
    return database_managers


class DatabaseManager:
    def __init__(self):
        self.database_name = None
        self.database_org_file = []
        self.database = {
            "entities": set(),
            "relationships": set(),
            "return_values": set(),
            "triplets": set(),
        }
        self.topk_retriever = None
        self._queried_pairs: set = set()  # To track the utilization ratio, tracks unique (entity, relationship) pairs successfully retrieved

    def __len__(self):
        return len(self.database["triplets"])

    def __str__(self):
        return (
            f"DatabaseManager: {len(self)} triplets, "
            f"{len(self.database['entities'])} entities, "
            f"{len(self.database['relationships'])} relationships, "
            f"{len(self.database['return_values'])} return values."
        )

    def init_topk_retriever(self, model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=None, default_threshold=0.6, adaptive : bool = False, use_inverses : bool = False):

        if self.topk_retriever is None:
            self.topk_retriever = TopkRetriever(
                self.database["triplets"], model_name, top_k, adaptive, default_threshold, database_name=self.database_name, use_hf_cache = False, use_inverses = use_inverses
            )
            logger.info(f"Top-k retriever initialized with {len(self)} triplets and threshold {self.topk_retriever.default_threshold}.")

    def retrieve_all_relationships_for_entity(self, entity: str, threshold: float, max_relationships: int) -> List[str]:
        """Return all (relationship, value) pairs for the entity closest to *entity*.

        Delegates to :meth:`TopkRetriever.retrieve_all_relationships_for_entity`.
        """
        if self.topk_retriever is None:
            raise RuntimeError("TopkRetriever is not initialized. Call init_topk_retriever() first.")
        return self.topk_retriever.retrieve_all_relationships_for_entity(entity, threshold, max_relationships)

    def retrieve_from_database(self, prompt: str, threshold: Optional[float] = None, top_k : int  = 4, return_triplets : bool = False):
        """Retrieve a single top-1 database result from a prompt containing dblookup. If lookup fails, raise an error."""
        pattern_lst = [
            r"\[dblookup\('((?:[^'\\]|\\.)+)',\s*'((?:[^'\\]|\\.)+)'\)\s*->",
            r"\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->",
            r"<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>"
        ]

        matches = {tuple(match) for pattern in pattern_lst for match in re.findall(pattern, prompt)}
        if not matches:
            raise DatabaseLookupError(
                f"[dblookup_fail_5] No valid dblookup pattern found in prompt: {prompt}", 
                "no_match_found"
            )

        if len(matches) > 1:
            raise DatabaseLookupError(
                f"[dblookup_fail_1] Multiple dblookup matches found: {matches} in prompt: {prompt}", 
                "multiple_matches"
            )
       
        entity, relationship = matches.pop()
        results = self.topk_retriever.retrieve_top_k(entity, relationship, threshold=threshold, return_triplets = return_triplets)

        if not results:
            raise DatabaseLookupError(
                f"[dblookup_fail_3] No retrieval results for entity='{entity}', relationship='{relationship}'",
                "no_retrieval_data_found"
            )
        self._queried_pairs.add((entity.strip(), relationship.strip()))
        return results

    def build_database(self, dataset: Union[DatasetDict, List[Dict]], database_name: Optional[str] = None, database_org_file: Optional[str] = None):
        """Build database from atomic knowledge dataset."""
        self.database_name = database_name
        if database_org_file:
            self.database_org_file.append(database_org_file)

        if isinstance(dataset, DatasetDict):
            dataset = extract_database(dataset)

        if "atomic_knowledge" not in dataset.column_names:
            dataset = extract_database(dataset)
        for example in dataset:
                # raise ValueError("Example missing 'atomic_knowledge'.")
            for entity, relationship, return_value in example["atomic_knowledge"]:
                self.database["entities"].add(entity)
                self.database["relationships"].add(relationship)
                self.database["return_values"].add(return_value)
                self.database["triplets"].add((entity, relationship, return_value))

        logger.info(f"Built database with {len(self)} triplets.")

    def build_database_from_triplets(self, triplets : list[tuple[str,str,str]], top_k : int = 4, default_threshold : float = 0.6, adaptive : bool = False, use_inverses : bool = False ) -> None:
        triplets = _add_inverse_triplets(triplets) if use_inverses else [tuple(t) for t in triplets]
        self.database["entities"] = [t[0] for t in triplets]
        self.database["relationships"] = [t[1] for t in triplets]
        self.database["return_values"] = [t[2] for t in triplets]
        self.database["triplets"] = triplets
        logger.info(f"Loaded database from triplets.")
        self.init_topk_retriever(top_k=top_k, default_threshold=default_threshold, adaptive = adaptive, use_inverses = False)

    def build_database_from_triplets_with_embeddings(self, triplets: list[tuple[str, str, str]], embeddings, top_k: int = 4, default_threshold: float = 0.6, adaptive: bool = False, use_inverses: bool = False, embedding_model : Optional[SentenceTransformer] = None) -> None:
        """Build database from triplets with pre-computed embeddings.

        Args:
            triplets: List of (entity, relationship, value) tuples
            embeddings: Pre-computed embeddings for the triplets
            top_k: Number of top results to retrieve
            default_threshold: Similarity threshold for retrieval
            adaptive: Whether to use adaptive threshold
            use_inverses: Whether to include inverse relationships
        """
        if len(triplets) != len(embeddings):
            raise ValueError(
                f"Expected one embedding per triplet, got {len(embeddings)} embeddings "
                f"for {len(triplets)} triplets."
            )

        # Deduplicate triplets (keep first occurrence), selecting matching embeddings
        seen = set()
        unique_indices = []
        for i, t in enumerate(triplets):
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                unique_indices.append(i)
        triplets = [triplets[i] for i in unique_indices]
        embeddings = embeddings[unique_indices]

        if use_inverses:
            inverse_triplets = []
            for entity, relationship, return_value in triplets:
                inverse_triplet = (return_value, relationship, entity)
                if inverse_triplet not in seen:
                    seen.add(inverse_triplet)
                    inverse_triplets.append(inverse_triplet)

            if inverse_triplets:
                if embedding_model is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    embedding_model = SentenceTransformer(
                        "sentence-transformers/all-MiniLM-L6-v2",
                        device=device,
                        local_files_only=True,
                    )
                inverse_texts = [_triplet_search_text(triplet) for triplet in inverse_triplets]
                inverse_embeddings = embedding_model.encode(
                    inverse_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                triplets.extend(inverse_triplets)
                embeddings = np.concatenate((np.asarray(embeddings), inverse_embeddings), axis=0)

        self.database["entities"] = [t[0] for t in triplets]
        self.database["relationships"] = [t[1] for t in triplets]
        self.database["return_values"] = [t[2] for t in triplets]
        self.database["triplets"] = triplets

        # Initialize TopkRetriever with precomputed embeddings
        self.topk_retriever = TopkRetriever(
            self.database["triplets"],
            top_k=top_k,
            threshold=default_threshold,
            adaptive=adaptive,
            # DatabaseManager owns inverse augmentation so the triplets and
            # their precomputed embeddings always remain aligned.
            use_inverses=False,
            database_name=self.database_name,
            use_hf_cache=False,
            precomputed_embeddings=embeddings,
            model = embedding_model,
        )


    def load_database(self, load_path: str, top_k : int = 4, default_threshold : float = 0.6, adaptive : bool = False, use_inverses : bool = False):
        """Load database from JSON file."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Database file not found: {load_path}")

        with open(load_path, "r") as f:
            data = json.load(f)

        if not all(k in data for k in ["entities", "relationships", "return_values", "triplets"]):
            raise ValueError(f"Invalid database format in {load_path}.")
        
        self.database_name = os.path.basename(load_path).split(".json")[0]
        self.database_org_file.append(load_path)

        self.database["entities"].update(data["entities"])
        self.database["relationships"].update(data["relationships"])
        self.database["return_values"].update(data["return_values"])
        self.database["triplets"].update(tuple(triplet) for triplet in data["triplets"])

        if use_inverses:
            self.database["triplets"] = set(_add_inverse_triplets(list(self.database["triplets"])))
            self.database["entities"] = {triplet[0] for triplet in self.database["triplets"]}
            self.database["return_values"] = {triplet[2] for triplet in self.database["triplets"]}

        logger.info(f"Loaded database from {load_path}.")

        self.init_topk_retriever(top_k=top_k, default_threshold=default_threshold, adaptive = adaptive, use_inverses = False)

    def save_database(self, save_path: str):
        """Save current database to a JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        serializable = {k: list(v) for k, v in self.database.items()}
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=4)
        logger.info(f"Database saved at {save_path}.")

class PerExampleRetriever:
    """Lightweight per-example database for two-phase generation.

    Stores triplets from Phase 1 (model-generated) and retrieves values
    using a multi-tier matching strategy:

        1. **Exact match** – normalized (entity, relationship) lookup (fastest).
        2. **Substring containment** – one string contains the other
           (catches plural/prefix differences like "voice actor" vs "voice actors").
        3. **Token Jaccard similarity** – overlap of word tokens
           (catches rewordings like "publication type" vs "type of publication").

    The entity and relationship scores are computed independently and then
    multiplied so that *both* must be reasonable.  A configurable
    ``fuzzy_threshold`` (default 0.3) governs the minimum combined score.
    """

    def __init__(self, triplets: list[tuple[str, str, str]], fuzzy_threshold: float = 0.3):
        """
        Args:
            triplets: List of (entity, relationship, value) tuples produced by Phase 1.
            fuzzy_threshold: Minimum combined score (entity × relationship) for
                a fuzzy match to be accepted.  Set to ``1.0`` to disable fuzzy
                matching and require exact matches only.
        """
        self._raw_triplets = list(triplets)
        self._fuzzy_threshold = fuzzy_threshold

        # --- Tier 1: exact lookup dict ---
        self._lookup: dict[tuple[str, str], str] = {}
        for entity, relationship, value in triplets:
            key = (self._normalize(entity), self._normalize(relationship))
            if key not in self._lookup:
                self._lookup[key] = value

        # --- Pre-compute for Tier 2/3: normalized strings + token sets ---
        self._indexed: list[tuple[str, str, set[str], set[str], str]] = []
        for entity, relationship, value in triplets:
            ent_norm = self._normalize(entity)
            rel_norm = self._normalize(relationship)
            self._indexed.append((
                ent_norm,
                rel_norm,
                self._tokenize(ent_norm),
                self._tokenize(rel_norm),
                value,
            ))

        # Running counters (lightweight stats, no lock needed – single-threaded)
        self._exact_hits = 0
        self._fuzzy_hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # String helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(s: str) -> str:
        """Lowercase, strip, collapse whitespace, replace underscores."""
        return " ".join(s.lower().strip().replace("_", " ").split())

    @staticmethod
    def _tokenize(s: str) -> set[str]:
        """Split a *normalized* string into a set of word tokens."""
        return set(s.split()) if s else set()

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _containment_score(query: str, stored: str) -> float:
        """Return 1.0 if one string fully contains the other, else 0.0."""
        if query in stored or stored in query:
            return 1.0
        return 0.0

    def _field_score(self, query_norm: str, query_tokens: set[str],
                     stored_norm: str, stored_tokens: set[str]) -> float:
        """Score a single field (entity or relationship) using the best of
        substring containment and Jaccard similarity."""
        return max(
            self._containment_score(query_norm, stored_norm),
            self._jaccard(query_tokens, stored_tokens),
        )

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(self, entity: str, relationship: str) -> str | None:
        """Return the value for (entity, relationship), or ``None`` if no
        match above ``fuzzy_threshold`` is found.

        Matching strategy (tried in order):
            1. Exact normalized match.
            2. Best fuzzy match above ``self._fuzzy_threshold``.
        """
        q_ent = self._normalize(entity)
        q_rel = self._normalize(relationship)

        # --- Tier 1: exact ---
        exact = self._lookup.get((q_ent, q_rel))
        if exact is not None:
            self._exact_hits += 1
            return exact

        # --- Tier 2/3: fuzzy ---
        if not self._indexed:
            self._misses += 1
            return None

        q_ent_tokens = self._tokenize(q_ent)
        q_rel_tokens = self._tokenize(q_rel)

        best_score = 0.0
        best_value: str | None = None
        for stored_ent, stored_rel, stored_ent_tok, stored_rel_tok, value in self._indexed:
            ent_score = self._field_score(q_ent, q_ent_tokens, stored_ent, stored_ent_tok)
            rel_score = self._field_score(q_rel, q_rel_tokens, stored_rel, stored_rel_tok)
            # Both entity AND relationship must be reasonable
            combined = ent_score * rel_score
            if combined > best_score:
                best_score = combined
                best_value = value

        if best_score >= self._fuzzy_threshold:
            self._fuzzy_hits += 1
            return best_value

        self._misses += 1
        return None

    # ------------------------------------------------------------------
    # Drop-in API compatible with DatabaseManager
    # ------------------------------------------------------------------

    def retrieve_from_database(self, prompt: str, return_triplets: bool = False, threshold: float | None = None) -> list[str]:
        """Drop-in replacement for ``DatabaseManager.retrieve_from_database``.

        Parses the db-lookup tokens from *prompt*, looks up (entity, relationship)
        in the per-example store, and returns the value wrapped in a list.
        Raises ``DatabaseLookupError`` on failure (same contract as DatabaseManager).
        """
        import re
        pattern_lst = [
            r"\[dblookup\('((?:[^'\\]|\\.)+)',\s*'((?:[^'\\]|\\.)+)'\)\s*->",
            r"\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->",
            r"<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>",
        ]
        matches = {tuple(match) for pattern in pattern_lst for match in re.findall(pattern, prompt)}

        if not matches:
            raise DatabaseLookupError(
                f"[per_example_fail] No valid dblookup pattern found in prompt: {prompt}",
                "no_match_found",
            )
        if len(matches) > 1:
            raise DatabaseLookupError(
                f"[per_example_fail] Multiple dblookup matches found: {matches}",
                "multiple_matches",
            )

        entity, relationship = matches.pop()
        value = self.retrieve(entity, relationship)
        if value is None:
            raise DatabaseLookupError(
                f"[per_example_fail] No result for entity='{entity}', relationship='{relationship}'"
                f" (db has {len(self._raw_triplets)} triplets, {self._exact_hits} exact/{self._fuzzy_hits} fuzzy hits so far)",
                "no_retrieval_data_found",
            )
        if return_triplets:
            return [f"({entity}, {relationship}, {value})"]
        return [value]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return lookup statistics."""
        return {
            "exact_hits": self._exact_hits,
            "fuzzy_hits": self._fuzzy_hits,
            "misses": self._misses,
        }

    def __len__(self) -> int:
        return len(self._raw_triplets)

    def __repr__(self) -> str:
        return (
            f"PerExampleRetriever({len(self._raw_triplets)} triplets, "
            f"threshold={self._fuzzy_threshold}, "
            f"exact={self._exact_hits}/fuzzy={self._fuzzy_hits}/miss={self._misses})"
        )


def load_current_database(database_dir: str) -> DatabaseManager:
    """Load all database files from directory."""
    db_manager = DatabaseManager()
    for file in os.listdir(database_dir):
        if file.endswith(".json"):
            db_manager.load_database(os.path.join(database_dir, file))
    return db_manager


if __name__ == "__main__":
    import time
    import psutil
    import os

    def run_retrieval_tests(db_manager):
        """Run a series of retrieval tests using the provided knowledge triplets"""
        
        # Test entity-relationship pairs from provided knowledge triplets
        test_entity_lst = [
            "Thomas Macdonald-Paterson",
            "Trivium",
            "Runar Berg",
            "Prattville",
            "Izzy Asper",
            "Josef Paldus"
        ]
        
        test_relationship_lst = [
            "Political Office",
            "Components",
            "Sport",
            "State",
            "Role in Media",
            "Affiliation"
        ]
        
        # Additional variant test cases (variants of the same entities and relationships)
        variant_entity_lst = [
            "Macdonald-Paterson, Thomas",  # Name format variation
            "The Trivium",                 # With article
            "Berg, Runar",                 # Last name first format
            "Prattville, Alabama",         # Combined with state
            "Israel Asper",                # Full name instead of nickname
            "Joseph Paldus"                # Spelling variation
        ]
        
        # Variations in relationship formatting/naming
        variant_relationship_lst = [
            "political_office",            # Underscore format
            "elements",                    # Synonym for components
            "played_sport",                # Verb form
            "located_in",                  # Alternative relationship
            "profession",                  # Alternative relationship
            "university"                   # More specific relationship
        ]
        
        # Incorrect/non-existent entity-relationship pairs for negative testing
        incorrect_entity_lst = [
            "Thomas Macdonald",            # Incomplete name
            "Quadrivium",                  # Related but different concept
            "Runar Smith",                 # Wrong last name
            "Prattville",                  # Correct entity
            "Izzy Aspers",                 # Misspelled
            "Josef Einstein"               # Wrong person
        ]
        
        incorrect_relationship_lst = [
            "Political Party",             # Wrong but related relationship
            "Inventors",                   # Unrelated relationship
            "Team",                        # More specific than just sport
            "Country",                     # Higher level geographical unit
            "Birth Date",                  # Unrelated relationship
            "Publications"                 # Related but different relationship
        ]
        
        print("\n=== RUNNING EXACT MATCH TESTS ===")
        start_time = time.time()
        db_manager.init_topk_retriever()
        end_time = time.time()

        print(f"Top-k Retriever Initialization Time: {end_time - start_time:.2f}s")
        print_system_usage()

        for entity, relationship in zip(test_entity_lst, test_relationship_lst):
            try:
                start_time = time.time()
                results = db_manager.topk_retriever.retrieve_top_k(entity, relationship, threshold=0.8)
                end_time = time.time()
                print(f"Query Results for '{entity}' and '{relationship}':", results)
                print(f"Query Time: {end_time - start_time:.4f}s")
                print_system_usage()
            except Exception as e:
                print(f"Query Error for '{entity}' and '{relationship}':", e)
        
        print("\n=== RUNNING VARIANT TESTS ===")
        for entity, relationship in zip(variant_entity_lst, variant_relationship_lst):
            try:
                start_time = time.time()
                results = db_manager.topk_retriever.retrieve_top_k(entity, relationship, threshold=0.8)
                end_time = time.time()
                print(f"Query Results for '{entity}' and '{relationship}':", results)
                print(f"Query Time: {end_time - start_time:.4f}s")
                print_system_usage()
            except Exception as e:
                print(f"Query Error for '{entity}' and '{relationship}':", e)
        
        print("\n=== RUNNING NEGATIVE TESTS ===")
        for entity, relationship in zip(incorrect_entity_lst, incorrect_relationship_lst):
            try:
                start_time = time.time()
                results = db_manager.topk_retriever.retrieve_top_k(entity, relationship, threshold=0.8)
                end_time = time.time()
                print(f"Query Results for '{entity}' and '{relationship}':", results)
                print(f"Query Time: {end_time - start_time:.4f}s")
                print_system_usage()
            except Exception as e:
                print(f"Query Error for '{entity}' and '{relationship}':", e)
        
        # Test expected output for known triplets
        print("\n=== RUNNING VALIDATION TESTS ===")
        expected_results = {
            ("Thomas Macdonald-Paterson", "Political Office"): "Parliament of Queensland",
            ("Trivium", "Components"): "grammar, logic, rhetoric",
            ("Runar Berg", "Sport"): "football",
            ("Prattville", "State"): "Alabama",
            ("Izzy Asper", "Role in Media"): "media baron",
            ("Josef Paldus", "Affiliation"): "University of Waterloo"
        }
        
        for (entity, relationship), expected in expected_results.items():
            try:
                results = db_manager.topk_retriever.retrieve_top_k(entity, relationship)
                if results and expected in str(results):
                    print(f"✓ PASS: '{entity}' + '{relationship}' correctly returned '{expected}'")
                else:
                    print(f"✗ FAIL: '{entity}' + '{relationship}' returned {results}, expected to contain '{expected}'")
            except Exception as e:
                print(f"✗ ERROR: '{entity}' + '{relationship}' test failed with error:", e)
        

    def print_system_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"Memory Usage: {mem_info.rss / (1024 ** 3):.2f} GB")
        print(f"CPU Usage: {cpu_percent}%")

    database_path = "./database_v4/trex11k-annotator_database.json"
    
    db_manager = DatabaseManager()
    
    start_time = time.time()

    db_manager.load_database(database_path)
    print(db_manager)
    db_manager.database_name = database_path.split("/")[-1].split(".json")[0]

    end_time = time.time()
    print(f"Loading Time: {end_time - start_time:.2f}s")
    print_system_usage()

    run_retrieval_tests(db_manager)
