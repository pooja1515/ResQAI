from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from resqai.pipelines.voice_intelligence.gemma_client import GemmaOllamaClient, OllamaConfig
from resqai.rag.prompts import build_rag_prompt, build_repair_prompt, build_stricter_rag_prompt
from resqai.rag.retriever import RetrieverConfig, retrieve, format_context_snippets
from resqai.rag.utils import (
    as_list_of_str,
    cleanup_json_text,
    count_sentences,
    dumps_json,
    find_json_object,
)
from resqai.rag.vector_store import VectorStoreConfig

import json

logger = logging.getLogger("resqai.rag")


@dataclass(frozen=True)
class RAGReasonerConfig:
    model: str = "gemma4"
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 256
    k: int = 3


def _parse_rag_json(raw: str) -> dict[str, object]:
    s = cleanup_json_text(raw)
    if not s:
        raise ValueError("Empty model output.")
    try:
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            raise ValueError("Model output must be a JSON object.")
        data = parsed
    except json.JSONDecodeError:
        data = json.loads(find_json_object(s))

    allowed = {"risk_level", "recommended_actions", "safety_guidelines", "reasoning_summary"}
    extra = set(data.keys()) - allowed
    if extra:
        raise ValueError(f"Unexpected keys: {sorted(extra)}")

    risk_level = str(data.get("risk_level", "")).strip().lower()
    if risk_level not in {"low", "medium", "high", "critical"}:
        raise ValueError("risk_level must be one of: low|medium|high|critical")
    data["risk_level"] = risk_level

    for key in ("recommended_actions", "safety_guidelines"):
        data[key] = as_list_of_str(data.get(key, []))
        # Keep outputs short and stable
        data[key] = data[key][:3]

    rs = str(data.get("reasoning_summary", "")).strip()
    if not rs:
        raise ValueError("reasoning_summary must be non-empty.")
    if count_sentences(rs) > 2:
        raise ValueError("reasoning_summary must be under 2 sentences.")
    data["reasoning_summary"] = rs
    return data


def answer_query(query: str, *, cfg: RAGReasonerConfig, vs_cfg: VectorStoreConfig, ollama: OllamaConfig | None = None) -> dict[str, object]:
    top_k = min(int(cfg.k), 3)
    retrieved = retrieve(query, cfg=RetrieverConfig(k=top_k), vs_cfg=vs_cfg)
    context_snippets = format_context_snippets(retrieved, max_chars=500)[:top_k]

    client = GemmaOllamaClient(cfg=ollama)
    prompt = build_rag_prompt(query, context_snippets)

    last_raw: str | None = None
    last_exc: Exception | None = None
    for attempt in range(3):
        attempt_prompt = prompt
        if attempt == 1:
            attempt_prompt = build_stricter_rag_prompt(query, context_snippets)
        elif attempt >= 2:
            attempt_prompt = build_repair_prompt((last_raw or "")[:2000], query, context_snippets)

        raw = client.generate(
            model=cfg.model,
            prompt=attempt_prompt,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            json_only=True,
            # Add a conservative stop token to discourage trailing text.
            stop=["\n\n", "\n```", "```"],
        )
        last_raw = raw
        try:
            data = _parse_rag_json(raw)
            return data
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("RAG JSON parse/validation failed (attempt %s/3): %s", attempt + 1, exc)

    raise RuntimeError("Gemma did not return valid RAG JSON after retries.") from last_exc


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="ResQAI RAG reasoner (ChromaDB + Gemma via Ollama).")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--model", type=str, default=RAGReasonerConfig.model, help="Ollama model name (gemma4, gemma3:4b).")
    p.add_argument("--k", type=int, default=RAGReasonerConfig.k)
    p.add_argument("--persist-dir", type=str, default=str(VectorStoreConfig.persist_dir))
    p.add_argument("--collection", type=str, default=VectorStoreConfig.collection_name)
    p.add_argument("--embedding-model", type=str, default=VectorStoreConfig.embedding_model)
    args = p.parse_args()

    out = answer_query(
        args.query,
        cfg=RAGReasonerConfig(model=args.model, k=args.k),
        vs_cfg=VectorStoreConfig(
            persist_dir=Path(args.persist_dir),
            collection_name=args.collection,
            embedding_model=args.embedding_model,
        ),
    )
    print(dumps_json(out))


if __name__ == "__main__":
    main()
