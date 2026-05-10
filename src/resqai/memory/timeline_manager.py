from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from resqai.memory.memory_store import MemoryStore, MemoryStoreConfig
from resqai.memory.memory_utils import safe_time_hhmm, summarize_event, dumps_json

logger = logging.getLogger("resqai.memory")


@dataclass(frozen=True)
class TimelineConfig:
    store: MemoryStoreConfig = MemoryStoreConfig()
    max_events: int = 25
    max_payload_chars: int = 600


class TimelineManager:
    def __init__(self, cfg: TimelineConfig | None = None) -> None:
        self.cfg = cfg or TimelineConfig()
        self.store = MemoryStore(self.cfg.store)

    def close(self) -> None:
        self.store.close()

    def recent_events(self) -> list[dict[str, Any]]:
        # Fetch most recent first by default.
        return self.store.fetch_recent(limit=self.cfg.max_events)

    def recent_events_chronological(self) -> list[dict[str, Any]]:
        # Oldest -> newest for visualization.
        return self.store.fetch(limit=self.cfg.max_events, order="asc")

    def to_context(self, events: list[dict[str, Any]]) -> list[str]:
        """Render events into compact lines for LLM consumption."""
        lines: list[str] = []
        for e in events:
            payload = e.get("payload") or {}
            payload_str = str(payload)
            if len(payload_str) > self.cfg.max_payload_chars:
                payload_str = payload_str[: self.cfg.max_payload_chars - 3] + "..."
            lines.append(
                f"ts={e.get('ts')} type={e.get('event_type')} "
                f"severity={e.get('severity_label')} urgency={e.get('urgency_label')} "
                f"payload={payload_str}"
            )
        return lines


def _format_line(e: dict[str, Any]) -> str:
    ts = str(e.get("ts") or "")
    hhmm = safe_time_hhmm(ts)
    et = str(e.get("event_type") or "unknown")
    sev = str(e.get("severity_label") or "unknown")
    payload = e.get("payload") if isinstance(e.get("payload"), dict) else {}
    summary = summarize_event(et, payload)
    return f"{hhmm} | {et} | {sev} | {summary}"


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="ResQAI memory timeline viewer (SQLite).")
    p.add_argument("--limit", type=int, default=TimelineConfig.max_events)
    p.add_argument("--pretty", action="store_true", help="Pretty-print the timeline lines.")
    p.add_argument("--json", action="store_true", help="Print raw events as JSON instead of formatted lines.")
    args = p.parse_args()

    tm = TimelineManager(TimelineConfig(max_events=int(args.limit)))
    try:
        events = tm.store.fetch(limit=int(args.limit), order="asc")
    finally:
        tm.close()

    if not events:
        logger.info("No events found in memory store.")
        print("(no events)")
        return

    logger.info("timeline_events=%s", len(events))
    if args.json:
        print(dumps_json({"events": events}))
        return

    for e in events:
        line = _format_line(e)
        print(line if args.pretty else line)


if __name__ == "__main__":
    main()
