"""Build a reproducible Wikipedia context corpus for PopQA.

The corpus contains the plain-text content of each subject's Wikipedia article
split on non-empty lines. This module fetches that content through the official
MediaWiki Action API while retaining enough metadata to audit redirects, missing
pages, and the exact Wikipedia revisions returned.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from datasets import load_dataset  # type: ignore

from .provenance import hf_source, selected_rows_provenance, sha256_file

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_LICENSE = {
    "name": "Creative Commons Attribution-ShareAlike 4.0 International",
    "url": "https://creativecommons.org/licenses/by-sa/4.0/",
}
DEFAULT_USER_AGENT = (
    "MultiHopReasoningPopQABot/1.0 " "(https://github.com/MihirMishra23/Multi-Hop-Reasoning)"
)
CORPUS_FILENAME = "popqa_contexts.jsonl"
MANIFEST_FILENAME = "manifest.json"
RETRIABLE_HTTP_CODES = {429, 500, 502, 503, 504}
RETRIABLE_API_CODES = {"maxlag", "readonly", "ratelimited"}


class WikipediaAPIError(RuntimeError):
    """An error returned in an otherwise successful MediaWiki response."""

    def __init__(self, code: str, message: str):
        super().__init__(f"Wikipedia API error {code}: {message}")
        self.code = code


def extract_paragraphs(extract: str) -> List[str]:
    """Split a MediaWiki plain-text extract into non-empty lines."""
    return [line.strip() for line in extract.splitlines() if line.strip()]


def _redirect_targets(query: Mapping[str, Any]) -> Dict[str, str]:
    targets: Dict[str, str] = {}
    for field in ("normalized", "redirects"):
        for item in query.get(field, []) or []:
            source = str(item.get("from", ""))
            destination = str(item.get("to", ""))
            if source and destination:
                targets[source] = destination
    return targets


def _resolve_title(title: str, redirects: Mapping[str, str]) -> str:
    resolved = title
    visited = set()
    while resolved in redirects and resolved not in visited:
        visited.add(resolved)
        resolved = redirects[resolved]
    return resolved


def records_from_response(
    requested_titles: Sequence[str], payload: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Return one ordered corpus record per requested title."""
    if "error" in payload:
        error = payload["error"]
        raise WikipediaAPIError(str(error.get("code", "unknown")), str(error.get("info", error)))

    query = payload.get("query", {})
    redirects = _redirect_targets(query)
    pages = query.get("pages", []) or []
    pages_by_title = {str(page.get("title", "")): page for page in pages}

    records: List[Dict[str, Any]] = []
    for requested_title in requested_titles:
        resolved_title = _resolve_title(requested_title, redirects)
        page = pages_by_title.get(resolved_title)
        if page is None:
            records.append(
                {
                    "requested_title": requested_title,
                    "title": resolved_title,
                    "status": "error",
                    "error": "Wikipedia response did not contain the requested page",
                    "paragraphs": [],
                }
            )
            continue

        if page.get("missing") is not None:
            records.append(
                {
                    "requested_title": requested_title,
                    "title": resolved_title,
                    "status": "missing",
                    "page_id": None,
                    "revision_id": None,
                    "revision_timestamp": None,
                    "url": page.get("fullurl"),
                    "paragraphs": [],
                }
            )
            continue

        revisions = page.get("revisions") or []
        revision = revisions[0] if revisions else {}
        paragraphs = extract_paragraphs(str(page.get("extract", "")))
        records.append(
            {
                "requested_title": requested_title,
                "title": str(page.get("title", resolved_title)),
                "status": "ok" if paragraphs else "empty",
                "page_id": page.get("pageid"),
                "revision_id": revision.get("revid"),
                "revision_timestamp": revision.get("timestamp"),
                "url": page.get("fullurl"),
                "paragraphs": paragraphs,
            }
        )
    return records


class WikipediaClient:
    """Small, policy-aware MediaWiki client with retries and batching."""

    def __init__(
        self,
        *,
        api_url: str = WIKIPEDIA_API_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 60.0,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        request_delay: float = 0.25,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("A descriptive Wikipedia User-Agent is required")
        self.api_url = api_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self._opener = opener
        self._sleep = sleeper

    def _request(
        self,
        titles: Sequence[str],
        continuation: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
            "redirects": "1",
            "prop": "extracts|info|revisions",
            "explaintext": "1",
            "exsectionformat": "plain",
            # MediaWiki caps whole-article extracts at one per response. It
            # returns an `excontinue` token when more titles remain.
            "exlimit": "1",
            "inprop": "url",
            "rvprop": "ids|timestamp",
            "titles": "|".join(titles),
        }
        if continuation:
            params.update({str(key): str(value) for key, value in continuation.items()})
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
        )
        with self._opener(request, timeout=self.timeout) as response:
            return json.load(response)

    def _request_all(self, titles: Sequence[str]) -> Dict[str, Any]:
        """Follow MediaWiki continuation and merge the per-title extracts."""
        merged_query: Dict[str, Any] = {"normalized": [], "redirects": [], "pages": []}
        pages_by_title: Dict[str, Dict[str, Any]] = {}
        continuation: Optional[Mapping[str, Any]] = None
        seen_continuations = set()

        while True:
            payload = self._request(titles, continuation)
            if "error" in payload:
                error = payload["error"]
                raise WikipediaAPIError(
                    str(error.get("code", "unknown")), str(error.get("info", error))
                )

            query = payload.get("query", {})
            for field in ("normalized", "redirects"):
                merged_query[field].extend(query.get(field, []) or [])
            for page in query.get("pages", []) or []:
                title = str(page.get("title", ""))
                pages_by_title.setdefault(title, {}).update(page)

            continuation = payload.get("continue")
            if not continuation:
                break
            marker = tuple(sorted((str(key), str(value)) for key, value in continuation.items()))
            if marker in seen_continuations:
                raise RuntimeError(f"Repeated Wikipedia continuation token: {marker}")
            seen_continuations.add(marker)
            if self.request_delay:
                self._sleep(self.request_delay)

        merged_query["pages"] = list(pages_by_title.values())
        return {"query": merged_query}

    def fetch(self, titles: Sequence[str]) -> List[Dict[str, Any]]:
        if not titles:
            return []
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                payload = self._request_all(titles)
                return records_from_response(titles, payload)
            except WikipediaAPIError as exc:
                last_error = exc
                retriable = exc.code in RETRIABLE_API_CODES
            except urllib.error.HTTPError as exc:
                last_error = exc
                retriable = exc.code in RETRIABLE_HTTP_CODES
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                retriable = True

            if not retriable or attempt >= self.max_retries:
                break

            delay = self.retry_delay * (2**attempt)
            if isinstance(last_error, urllib.error.HTTPError):
                retry_after = last_error.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            self._sleep(delay)

        assert last_error is not None
        raise last_error


def _ordered_unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def load_popqa_selection(*, seed: Optional[int], limit: Optional[int]) -> Dict[str, Any]:
    """Load the pinned PopQA rows and return their ordered unique subject titles."""
    source = hf_source("popqa")
    dataset = load_dataset(source["path"], split="test", revision=source["revision"])
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    titles = _ordered_unique(row.get("s_wiki_title", "") for row in dataset)
    row_ids = [str(row.get("id", "")) for row in dataset]
    return {
        "titles": titles,
        "rows": len(dataset),
        "provenance": selected_rows_provenance("popqa", row_ids, seed=seed, setting="test"),
    }


def _read_records(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            requested_title = str(record.get("requested_title") or record.get("title") or "")
            if requested_title:
                records[requested_title] = record
    return records


def _write_canonical_records(
    path: Path, titles: Sequence[str], records: Mapping[str, Mapping[str, Any]]
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for title in titles:
            record = records.get(title)
            if record is not None:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_deterministic_gzip(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    with source.open("rb") as input_stream, temporary.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as output_stream:
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(chunk)
    os.replace(temporary, destination)


def build_corpus(
    *,
    output_dir: Path,
    selection: Mapping[str, Any],
    client: WikipediaClient,
    batch_size: int = 10,
    request_delay: float = 0.25,
    refresh: bool = False,
    create_gzip: bool = False,
) -> Dict[str, Any]:
    """Build or resume a corpus and return its manifest."""
    if batch_size < 1 or batch_size > 20:
        raise ValueError("batch_size must be between 1 and 20")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / CORPUS_FILENAME
    titles = list(selection["titles"])
    records = _read_records(corpus_path)
    completed_statuses = {"ok", "missing", "empty"}
    pending = [
        title
        for title in titles
        if refresh or records.get(title, {}).get("status") not in completed_statuses
    ]

    with corpus_path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            try:
                fetched = client.fetch(batch)
            except Exception as exc:  # Preserve progress and make failures auditable.
                fetched = [
                    {
                        "requested_title": title,
                        "title": title,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "paragraphs": [],
                    }
                    for title in batch
                ]
            for record in fetched:
                records[str(record["requested_title"])] = record
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            if request_delay and start + batch_size < len(pending):
                time.sleep(request_delay)

    _write_canonical_records(corpus_path, titles, records)
    selected_records = [records[title] for title in titles if title in records]
    status_counts: Dict[str, int] = {}
    for record in selected_records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    artifact_path = corpus_path
    if create_gzip:
        artifact_path = output_dir / f"{CORPUS_FILENAME}.gz"
        _write_deterministic_gzip(corpus_path, artifact_path)

    manifest = {
        "format_version": 1,
        "dataset": selection["provenance"],
        "wikipedia": {
            "api_url": client.api_url,
            "license": WIKIPEDIA_LICENSE,
            "user_agent": client.user_agent,
        },
        "selection": {
            "rows": selection["rows"],
            "unique_requested_titles": len(titles),
        },
        "corpus": {
            "path": artifact_path.name,
            "sha256": sha256_file(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "records": len(selected_records),
            "status_counts": status_counts,
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a title-keyed Wikipedia context corpus for pinned PopQA"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Select only this many PopQA rows")
    parser.add_argument("--seed", type=int, help="Shuffle PopQA before applying --limit")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--gzip", action="store_true", dest="create_gzip")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    selection = load_popqa_selection(seed=args.seed, limit=args.limit)
    client = WikipediaClient(
        user_agent=args.user_agent,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_delay=args.request_delay,
    )
    manifest = build_corpus(
        output_dir=args.output_dir,
        selection=selection,
        client=client,
        batch_size=args.batch_size,
        request_delay=args.request_delay,
        refresh=args.refresh,
        create_gzip=args.create_gzip,
    )
    print(json.dumps(manifest["corpus"], sort_keys=True))
    failures = sum(
        count
        for status, count in manifest["corpus"]["status_counts"].items()
        if status not in {"ok"}
    )
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
