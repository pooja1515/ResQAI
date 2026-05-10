from __future__ import annotations

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

from resqai.rag.vector_store import VectorStoreConfig, build_vector_store
from resqai.rag.utils import resolve_from_repo, dumps_json

logger = logging.getLogger("resqai.rag")


@dataclass(frozen=True)
class RetrieverConfig:
    k: int = 5
    score_threshold: float | None = None


def retrieve(query: str, *, cfg: RetrieverConfig, vs_cfg: VectorStoreConfig) -> list[dict]:
    """Retrieve relevant docs for a query.

    Returns a list of dicts: {text, metadata, score?}
    """
    logger.info("retriever_load persist_dir=%s collection=%s", str(vs_cfg.persist_dir), vs_cfg.collection_name)
    vs = build_vector_store(vs_cfg)

    # Prefer similarity_search_with_score when available.
    results: list[dict] = []
    try:
        pairs = vs.similarity_search_with_score(query, k=cfg.k)
        for doc, score in pairs:
            if cfg.score_threshold is not None and score > cfg.score_threshold:
                continue
            meta = doc.metadata or {}
            source_file = Path(str(meta.get("source", "") or "")).name if meta.get("source") else ""
            results.append(
                {"text": doc.page_content, "metadata": meta, "score": float(score), "source_file": source_file}
            )
        return results
    except Exception:
        docs = vs.similarity_search(query, k=cfg.k)
        for doc in docs:
            meta = doc.metadata or {}
            source_file = Path(str(meta.get("source", "") or "")).name if meta.get("source") else ""
            results.append({"text": doc.page_content, "metadata": meta, "source_file": source_file})
        return results


def format_context_snippets(results: list[dict], *, max_chars: int = 1200) -> list[str]:
    """Format retrieved chunks into compact snippets for LLM context."""
    snippets: list[str] = []
    for r in results:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        source_file = r.get("source_file") or _format_source(meta)
        score = r.get("score")

        header_parts = []
        if source_file:
            header_parts.append(f"source={source_file}")
        if isinstance(score, (int, float)):
            header_parts.append(f"score={float(score):.4f}")
        header = " | ".join(header_parts)
        body = text if len(text) <= max_chars else text[: max_chars - 3] + "..."
        snippets.append(f"{header}\n{body}" if header else body)
    return snippets


def _chroma_has_data(persist_dir: Path) -> bool:
    """Heuristic check for an existing persisted Chroma DB."""
    if not persist_dir.exists() or not persist_dir.is_dir():
        return False
    # Typical files/dirs: chroma.sqlite3, index/, or collection parquet files
    for name in ("chroma.sqlite3", "index"):
        if (persist_dir / name).exists():
            return True
    return any(persist_dir.iterdir())


def _format_source(meta: dict) -> str:
    src = meta.get("source")
    if isinstance(src, str) and src:
        return Path(src).name
    return ""


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Retrieve top-k disaster knowledge chunks from ChromaDB.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--k", type=int, default=RetrieverConfig.k)
    p.add_argument("--persist-dir", type=Path, default=VectorStoreConfig.persist_dir)
    p.add_argument("--collection", type=str, default=VectorStoreConfig.collection_name)
    p.add_argument("--embedding-model", type=str, default=VectorStoreConfig.embedding_model)
    args = p.parse_args()

    persist_dir = resolve_from_repo(args.persist_dir)
    logger.info("persist_dir=%s collection=%s", str(persist_dir), args.collection)

    if not _chroma_has_data(persist_dir):
        logger.warning(
            "Vector DB not found or empty at %s. Run ingestion first: python -m resqai.rag.ingest --pdf <file.pdf>",
            str(persist_dir),
        )
        print(dumps_json({"results": [], "message": "vector_db_missing_or_empty"}))
        return

    vs_cfg = VectorStoreConfig(
        persist_dir=persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    cfg = RetrieverConfig(k=args.k)

    try:
        results = retrieve(args.query, cfg=cfg, vs_cfg=vs_cfg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed.")
        raise RuntimeError("Retrieval failed. Verify dependencies and that the vector DB is readable.") from exc

    logger.info("retrieved_chunks=%s", len(results))
    if not results:
        print(dumps_json({"results": [], "message": "no_results"}))
        return

    for i, r in enumerate(results, start=1):
        text = (r.get("text") or "").strip()
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        score = r.get("score")
        source_file = _format_source(meta)

        print(f"\n=== Result {i} ===")
        if isinstance(score, (int, float)):
            print(f"score: {float(score):.6f}")
        if source_file:
            print(f"source_file: {source_file}")
        if meta:
            print("metadata:")
            print(dumps_json(meta))
        print("\nchunk:")
        print(text)


if __name__ == "__main__":
    main()
