import os
import pickle
import torch
import faiss
import gc
import logging
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class TopkRetriever:
    def __init__(self,
                 database: List[Tuple[str, str, str]],
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 top_k: int = 5,
                 adaptive : bool = False,
                 threshold: float = 0.6,
                 batch_size: int = 2048,
                 cache_dir: Optional[str] = "./data/database_cache",
                 database_name: str = "default_db",
                 use_hf_cache: bool = True,
                 use_inverses : bool = False,
                 hf_repo_id: str = "kilian-group/LMLM-database-cache",
                 precomputed_embeddings: Optional[np.ndarray] = None,
                 model: Optional[SentenceTransformer] = None):
        """
        Args:
            database: List of (entity, relation, value) triplets
            model_name: SentenceTransformer model path or name
            top_k: Number of nearest neighbors to retrieve
            threshold: Similarity threshold
            batch_size: Batch size for encoding
            cache_dir: Directory to cache FAISS index and mappings
            database_name: Name used for cache files
            use_hf_cache: Whether to use Hugging Face cache
            hf_repo_id: Hugging Face repository ID
            precomputed_embeddings: Optional pre-computed embeddings (skips encoding)
            model: Used to initialize model before building index.
        """
        self.database = database
        self.top_k = top_k if top_k else 5
        self.default_threshold = threshold
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.database_name = database_name
        self.is_adaptive = adaptive
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is None:
            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            self.model = model
        if precomputed_embeddings is None:
            self.model = self.model.half().eval()
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = precomputed_embeddings.shape[1]

        self.index = None
        self.id_to_triplet = {}
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        optional_inv_text = ""
        if use_inverses:
            optional_inv_text = "_use_inv"

        self._build_index(precomputed_embeddings=precomputed_embeddings)

        gc.collect()
        torch.cuda.empty_cache()

    def _get_cached_paths(self, cache_path, use_hf_cache=False, hf_repo_id=None):
        """
        Returns path to .index and .mapping files, downloading from HF if needed.
        If local cache exists, return the path.
        """
        index_path = f"{cache_path}.index"
        mapping_path = f"{cache_path}.mapping"

        if os.path.exists(index_path) and os.path.exists(mapping_path):
            return cache_path

        if use_hf_cache and hf_repo_id:
            try:
                from huggingface_hub import hf_hub_download

                index_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=os.path.basename(index_path),
                    repo_type="dataset"
                )
                mapping_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=os.path.basename(mapping_path),
                    repo_type="dataset"
                )
                logger.info(f"Downloaded cache from Hugging Face Hub: {hf_repo_id}")
                return os.path.splitext(index_path)[0] # strip .index or .mapping → return the shared prefix
            except Exception as e:
                logger.warning(f"[WARNING] Failed to download from Hugging Face Hub: {e}")
                return None

        return None

    def _save_to_cache(self, cache_path):
        faiss.write_index(self.index, f"{cache_path}.index")
        with open(f"{cache_path}.mapping", 'wb') as f:
            pickle.dump(self.id_to_triplet, f)

    def _load_from_cache(self, cache_path):
        self.index = faiss.read_index(f"{cache_path}.index")
        with open(f"{cache_path}.mapping", 'rb') as f:
            self.id_to_triplet = pickle.load(f)

    def _build_index(self, precomputed_embeddings=None):
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))

        texts = [f"{self._normalize_text(ent)} {self._normalize_text(rel)}" for ent, rel, _ in self.database]
        if (len(texts) == 0):
            logger.warning("No triplets created! leaving index empty...")
            return

        if precomputed_embeddings is not None:
            embeddings = precomputed_embeddings
        else:
            logger.info(f"Encoding {len(texts)} (entity, relationship) pairs, which may take a long time...")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

        ids = np.arange(len(self.database))
        self.index.add_with_ids(embeddings, ids)
        self.id_to_triplet = {i: triplet for i, triplet in enumerate(self.database)}

        # Build a separate entity-only index for all-relationships lookup
        self._build_entity_index()

    def _build_entity_index(self):
        """Build a FAISS index over unique entity names for all-relationships lookup."""
        # Collect unique entities and their (relationship, value) pairs
        entity_to_rels: dict = {}
        for ent, rel, val in self.database:
            ent_norm = self._normalize_text(ent)
            entity_to_rels.setdefault(ent_norm, []).append((rel, val))

        self._entity_to_rels = entity_to_rels
        unique_entities = list(entity_to_rels.keys())
        self._entity_index_keys = unique_entities  # id → entity string

        entity_embeddings = self.model.encode(
            unique_entities,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._entity_index = faiss.IndexIDMap(faiss.IndexFlatIP(entity_embeddings.shape[1]))
        self._entity_index.add_with_ids(entity_embeddings, np.arange(len(unique_entities)))

    def retrieve_all_relationships_for_entity(self, entity: str, threshold: float, max_relationships: int) -> List[str]:
        """Return up to *max_relationships* ``(relationship, value)`` strings for the
        entity whose embedding is closest to *entity* and whose similarity is at
        least *threshold*.  Returns an empty list when nothing passes the threshold.
        """
        from multi_lmlm.constants import DB_ALL_RELATIONSHIPS_TOKEN

        entity = entity.replace(DB_ALL_RELATIONSHIPS_TOKEN, "").strip()

        if self._entity_index is None:
            return []

        query_emb = self.model.encode(
            [self._normalize_text(entity)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        distances, indices = self._entity_index.search(query_emb, 1)
        dist, idx = float(distances[0][0]), int(indices[0][0])

        if idx == -1 or dist < threshold:
            return []

        matched_entity = self._entity_index_keys[idx]
        rels = self._entity_to_rels.get(matched_entity, [])
        return [rel for rel, val in rels][:max_relationships]

    def retrieve_top_k(self, entity: str, relation: str, threshold: Optional[float] = None, return_triplets: bool = False) -> List[str]:
        query_text = f"{self._normalize_text(entity)} {self._normalize_text(relation)}"
        query_embedding = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

        distances, indices = self.index.search(query_embedding, self.top_k + 1)

        # Use per-query threshold if passed, otherwise fallback to default
        th = threshold if threshold is not None else self.default_threshold

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in self.id_to_triplet and dist >= th:
                assert 1.001 >= dist >= -1.001, f"FAISS dot product with normalized vectors should lie in [-1, 1]"

                triplet = self.id_to_triplet[idx]
                results.append((triplet[0], triplet[1], triplet[2], float(dist)))
        
        results.sort(key=lambda x: x[-1], reverse=True)

        if self.is_adaptive and len(results) > 1:
            differences = [results[i][-1] - results[i + 1][-1] for i in range(len(results) - 1)]
            max_diff = 0.0
            for i in range(len(differences)):
                if differences[i] >= max_diff:
                    max_idx = i
                    max_diff = differences[i]
            results = results[:max_idx + 1]

        if return_triplets:
            return_values = [f"({r[0]}, {r[1]}, {r[2]})" for r in results]
            return return_values[:self.top_k]

        return_values = [r[2] for r in results]
        return return_values[:self.top_k]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.lower().replace("_", " ").strip()
