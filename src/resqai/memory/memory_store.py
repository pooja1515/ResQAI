from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from resqai.memory.memory_utils import ensure_dir, resolve_from_repo

logger = logging.getLogger("resqai.memory")


@dataclass(frozen=True)
class MemoryStoreConfig:
    db_path: Path = Path("artifacts/memory/resqai_memory.sqlite3")


class MemoryStore:
    def __init__(self, cfg: MemoryStoreConfig) -> None:
        db_path = resolve_from_repo(cfg.db_path)
        self.cfg = MemoryStoreConfig(db_path=db_path)
        ensure_dir(db_path.parent)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              event_type TEXT NOT NULL,
              severity_label TEXT,
              urgency_label TEXT,
              source TEXT,
              payload_json TEXT NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
        self._conn.commit()

    def insert_event(
        self,
        *,
        ts: str,
        event_type: str,
        payload: dict[str, Any],
        severity_label: str | None = None,
        urgency_label: str | None = None,
        source: str | None = None,
    ) -> int:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO events (ts, event_type, severity_label, urgency_label, source, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                event_type,
                severity_label,
                urgency_label,
                source,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def fetch_recent(self, *, limit: int = 30) -> list[dict[str, Any]]:
        return self.fetch(limit=limit, order="desc")

    def fetch(self, *, limit: int = 30, order: str = "desc") -> list[dict[str, Any]]:
        order_sql = "DESC" if str(order).lower() != "asc" else "ASC"
        cur = self._conn.cursor()
        query = (
            "SELECT id, ts, event_type, severity_label, urgency_label, source, payload_json "
            "FROM events "
            f"ORDER BY ts {order_sql} "
            "LIMIT ?"
        )
        cur.execute(query, (int(limit),))
        rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            payload = {}
            try:
                payload = json.loads(r["payload_json"])
            except Exception:
                payload = {"_raw_payload_json": r["payload_json"]}
            out.append(
                {
                    "id": int(r["id"]),
                    "ts": r["ts"],
                    "event_type": r["event_type"],
                    "severity_label": r["severity_label"],
                    "urgency_label": r["urgency_label"],
                    "source": r["source"],
                    "payload": payload,
                }
            )
        return out
