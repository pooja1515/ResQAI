from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

from resqai.rag.utils import ensure_dir, resolve_from_repo

logger = logging.getLogger("resqai.rag")

_VS_LOCK = threading.Lock()


def _import_chroma():
    # LangChain moved integrations; support common import paths.
    try:
        from langchain_community.vectorstores import Chroma  # type: ignore
        return Chroma
    except Exception:
        try:
            from langchain.vectorstores import Chroma  # type: ignore
            return Chroma
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Chroma vectorstore is unavailable. Install langchain-community and chromadb.") from exc


def _import_embeddings():
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        return HuggingFaceEmbeddings
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
            return HuggingFaceEmbeddings
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("HuggingFaceEmbeddings unavailable. Install langchain-community and sentence-transformers.") from exc


@dataclass(frozen=True)
class VectorStoreConfig:
    persist_dir: Path = Path("artifacts/rag/chroma")
    collection_name: str = "resqai_disaster_kb"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def cache_key(self) -> tuple[str, str, str]:
        # Normalize to strings for stable hashing.
        return (str(resolve_from_repo(self.persist_dir)), self.collection_name, self.embedding_model)

@lru_cache(maxsize=8)
def _get_embeddings(model_name: str):
    Emb = _import_embeddings()
    t0 = time.perf_counter()
    emb = Emb(model_name=model_name)
    logger.info("embeddings_loaded model=%s duration_s=%.3f", model_name, time.perf_counter() - t0)
    return emb

@lru_cache(maxsize=8)
def _build_vector_store_cached(persist_dir: str, collection_name: str, embedding_model: str):
    Chroma = _import_chroma()
    embeddings = _get_embeddings(embedding_model)
    ensure_dir(Path(persist_dir))

    t0 = time.perf_counter()
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    logger.info(
        "chroma_loaded persist_dir=%s collection=%s duration_s=%.3f",
        persist_dir,
        collection_name,
        time.perf_counter() - t0,
    )
    return vs


def build_vector_store(cfg: VectorStoreConfig):
    persist_dir, collection, emb = cfg.cache_key()
    # Chroma init touches disk; guard with a lock to avoid races on first load.
    with _VS_LOCK:
        return _build_vector_store_cached(persist_dir, collection, emb)
