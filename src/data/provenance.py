"""Pinned sources and run-level provenance for evaluation datasets.

All network-backed datasets are identified by an immutable Git revision.  Files
that do not live in a Hugging Face dataset repository are additionally checked
against a SHA-256 digest before use.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

DATASET_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "confiqa": {
        "upstream": "https://github.com/byronBBL/Context-DPO",
        "revision": "557dadeeb9f47407890a5626128ca99907770b22",
        "file": "ConFiQA/ConFiQA-MR.json",
        "url": (
            "https://raw.githubusercontent.com/byronBBL/Context-DPO/"
            "557dadeeb9f47407890a5626128ca99907770b22/ConFiQA/ConFiQA-MR.json"
        ),
        "sha256": "dbb76f361831b754b87219344d80a386eaf83078ffe5888272dbe9d8e6c0eede",
        "bytes": 29846860,
        "license": "not specified by upstream",
    },
    "hotpotqa": {
        "hf_repo": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "upstream": "https://hotpotqa.github.io/",
        "license": "CC BY-SA 4.0",
    },
    "musique": {
        "hf_repo": "dgslibisey/MuSiQue",
        "revision": "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321",
        "upstream": "https://github.com/StonyBrookNLP/musique",
        "license": "not specified by the Hugging Face mirror",
    },
    "mquake": {
        "hf_repo": "henryzhongsc/MQuAKE-Remastered",
        "revision": "b54712d4b464d7e2d4edccd4022f95ddbcb719e7",
        "file": "data/CF6334-00000-of-00001.parquet",
        "upstream": "https://github.com/princeton-nlp/MQuAKE",
        "license": "CC BY 4.0 (remastered Hugging Face dataset)",
    },
    "2wiki": {
        "hf_repo": "kamelliao/2wikimultihopqa",
        "revision": "f4f0d7e4ae275d9281e90c46c93e66d6cdda3674",
        "file": "data/dev.json",
        "upstream": "https://github.com/Alab-NII/2wikimultihop",
        "license": "not specified by the Hugging Face mirror",
    },
    "synthworlds": {
        "hf_repo": "kenqgu/SynthWorlds",
        "revision": "d0f02ed540fe8b50b74fcf30eb7f342fd7dcae49",
        "license": "CC BY 4.0",
    },
    "trivia_qa": {
        "hf_repo": "mandarjoshi/trivia_qa",
        "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        "upstream": "https://nlp.cs.washington.edu/triviaqa/",
        "license": "unknown in the Hugging Face dataset card",
    },
    "popqa": {
        "hf_repo": "akariasai/PopQA",
        "revision": "098765c79ea10a2cb19c828324e33281b8336ec0",
        "file": "test.tsv",
        "upstream": "https://github.com/AlexTMallen/adaptive-retrieval",
        "license": "not specified by upstream",
    },
    "popqa_contexts": {
        "hf_repo": "ryannoonan/popqa-wikipedia-contexts",
        "revision": "1b91217d540cd4ab34e3efee1d7abe07c46f0209",
        "file": "popqa_contexts.jsonl.gz",
        "sha256": "afcc52bd4ab5ebe4f63a249fb305b21de288e8a4eafbf8c73cbd183ec3320482",
        "bytes": 45394537,
        "records": 12244,
        "upstream": "https://www.mediawiki.org/wiki/API:Main_page",
        "license": "CC BY-SA 4.0",
    },
}


_ALIASES = {
    "mquake-remastered": "mquake",
    "synth": "synthworlds",
    "triviaqa": "trivia_qa",
    "two_wiki": "2wiki",
}


def canonical_dataset_name(name: str) -> str:
    normalized = name.lower()
    return _ALIASES.get(normalized, normalized)


def dataset_source(name: str) -> Dict[str, Any]:
    """Return a JSON-serializable copy of a dataset's pinned source record."""
    canonical = canonical_dataset_name(name)
    if canonical not in DATASET_PROVENANCE:
        raise KeyError(f"No provenance record for dataset: {name}")
    return json.loads(json.dumps(DATASET_PROVENANCE[canonical]))


def hf_source(name: str) -> Dict[str, str]:
    source = dataset_source(name)
    return {"path": source["hf_repo"], "revision": source["revision"]}


def sha256_file(path: Union[os.PathLike, str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_data_cache() -> Path:
    configured = os.environ.get("MULTIHOP_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return root / "multi-hop-reasoning"


def download_verified_file(
    *, url: str, sha256: str, relative_path: str, cache_dir: Optional[str] = None
) -> str:
    """Download a file once and reject both stale cache files and bad downloads."""
    destination = (
        Path(cache_dir).expanduser() if cache_dir else default_data_cache()
    ) / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if actual == sha256:
            return str(destination)
        raise ValueError(
            f"Cached dataset file has SHA-256 {actual}, expected {sha256}: {destination}"
        )

    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(url) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    actual = sha256_file(temporary_path)
    if actual != sha256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file has SHA-256 {actual}, expected {sha256}: {url}")
    os.replace(temporary_path, destination)
    return str(destination)


def hf_dataset_file(name: str, filename: Optional[str] = None) -> str:
    """Resolve a cached pinned Hugging Face file and verify its registered checksum."""
    source = dataset_source(name)
    selected_file = filename or source.get("file")
    if not selected_file:
        raise ValueError(f"No file is registered for dataset: {name}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=source["hf_repo"],
        filename=selected_file,
        revision=source["revision"],
        repo_type="dataset",
    )
    expected_sha256 = source.get("sha256")
    if expected_sha256:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Hugging Face artifact has SHA-256 {actual_sha256}, "
                f"expected {expected_sha256}: {path}"
            )
    return path


def selected_rows_provenance(
    name: str,
    ids: Iterable[Any],
    *,
    seed: Optional[int],
    setting: Optional[str],
    counterfactual_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Describe the exact ordered rows used by one evaluation run."""
    ordered_ids = [str(value) for value in ids]
    id_digest = hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()
    result: Dict[str, Any] = {
        "source": dataset_source(name),
        "selection": {
            "seed": seed,
            "setting": setting,
            "count": len(ordered_ids),
            "ordered_ids_sha256": id_digest,
            "ordered_ids": ordered_ids,
        },
    }
    if counterfactual_count is not None:
        result["selection"]["counterfactual_count"] = counterfactual_count
    return result
