from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from resqai.rag.utils import resolve_from_repo
from resqai.rag.vector_store import VectorStoreConfig, build_vector_store

logger = logging.getLogger("resqai.rag")


def _import_doc_types():
    try:
        from langchain_core.documents import Document  # type: ignore
        return Document
    except Exception:
        try:
            from langchain.schema import Document  # type: ignore
            return Document
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("LangChain Document type unavailable. Install langchain.") from exc


def _import_text_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
        return RecursiveCharacterTextSplitter
    except Exception:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
            return RecursiveCharacterTextSplitter
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("LangChain text splitter unavailable. Install langchain.") from exc


def _import_pdf_loader():
    try:
        from langchain_community.document_loaders import PyPDFLoader  # type: ignore
        return PyPDFLoader
    except Exception:
        try:
            from langchain.document_loaders import PyPDFLoader  # type: ignore
            return PyPDFLoader
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyPDFLoader unavailable. Install langchain-community and pypdf.") from exc


@dataclass(frozen=True)
class IngestConfig:
    chunk_size: int = 900
    chunk_overlap: int = 120


def ingest_pdfs(paths: list[Path], *, source_tag: str, cfg: IngestConfig, vs_cfg: VectorStoreConfig) -> int:
    PyPDFLoader = _import_pdf_loader()
    Splitter = _import_text_splitter()
    Document = _import_doc_types()

    vs = build_vector_store(vs_cfg)
    splitter = Splitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    docs: list[Document] = []
    for p in paths:
        p = resolve_from_repo(p)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        loader = PyPDFLoader(str(p))
        loaded = loader.load()
        for d in loaded:
            d.metadata = dict(d.metadata or {})
            d.metadata.update({"source": str(p), "source_tag": source_tag, "type": "pdf"})
        docs.extend(loaded)

    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    vs.add_documents(chunks)
    try:
        vs.persist()
    except Exception:
        # Some LangChain versions auto-persist; keep best-effort.
        pass

    return len(chunks)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Ingest PDF disaster knowledge into ChromaDB.")
    p.add_argument("--pdf", type=Path, action="append", required=True, help="PDF path (repeatable).")
    p.add_argument("--source-tag", type=str, default="disaster_kb")
    p.add_argument("--persist-dir", type=Path, default=VectorStoreConfig.persist_dir)
    p.add_argument("--collection", type=str, default=VectorStoreConfig.collection_name)
    p.add_argument("--embedding-model", type=str, default=VectorStoreConfig.embedding_model)
    p.add_argument("--chunk-size", type=int, default=IngestConfig.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=IngestConfig.chunk_overlap)
    args = p.parse_args()

    count = ingest_pdfs(
        paths=args.pdf,
        source_tag=args.source_tag,
        cfg=IngestConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        vs_cfg=VectorStoreConfig(
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
        ),
    )
    logger.info("ingested_chunks=%s persist_dir=%s collection=%s", count, str(args.persist_dir), args.collection)


if __name__ == "__main__":
    main()

