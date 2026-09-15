from typing import Optional

from datasets import Dataset as HFDataset  # type: ignore

from .confiqa import load_confiqa
from .hotpotqa import load_hotpotqa
from .mquake import load_mquake
from .musique import load_musique
from .popqa import load_popqa
from .provenance import dataset_source, selected_rows_provenance
from .synthworlds import load_synthworlds
from .trivia_qa import load_trivia_qa
from .two_wiki import load_2wiki


def get_dataset(
    name: str,
    setting: str,
    split: str,  # hf split
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    sub_split: Optional[str] = None,  # sub split of 'split'
) -> HFDataset:
    """Dispatch to dataset-specific loaders and return a unified HF Dataset."""
    name_norm = name.lower()
    if name_norm == "hotpotqa":
        return load_hotpotqa(
            setting=setting, split=split, source=source, limit=limit, seed=seed, sub_split=sub_split
        )
    if name_norm == "musique":
        # MuSiQue has no 'setting'; ignore the argument
        return load_musique(split=split, source=source, limit=limit, seed=seed)
    if name_norm == "mquake-remastered" or name_norm == "mquake":
        return load_mquake(split=split, limit=limit, seed=seed, source=source)
    if name_norm == "2wiki":
        return load_2wiki(setting=setting, split=split, source=source, limit=limit, seed=seed)
    if name_norm in {"synthworlds", "synth"}:
        # For SynthWorlds, 'setting' is used as the subset (qa-sm or qa-rm)
        return load_synthworlds(subset=setting, split=split, source=source, limit=limit, seed=seed)
    if name_norm in {"trivia_qa", "triviaqa"}:
        return load_trivia_qa(split=split, source=source, limit=limit, seed=seed, setting=setting)
    if name_norm == "popqa":
        return load_popqa(split=split, source=source, limit=limit, seed=seed, setting=setting)
    if name_norm == "confiqa":
        return load_confiqa(split=split, source=source, limit=limit, seed=seed, setting=setting)
    raise ValueError(f"Unsupported dataset: {name}")


__all__ = [
    "dataset_source",
    "get_dataset",
    "load_confiqa",
    "load_hotpotqa",
    "load_musique",
    "load_popqa",
    "load_synthworlds",
    "load_trivia_qa",
    "selected_rows_provenance",
]
