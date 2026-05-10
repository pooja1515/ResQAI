from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("resqai.agents")


@dataclass(frozen=True)
class AgentContext:
    request_id: str = "request-id-placeholder"
    # Shared cross-agent state (lightweight; avoid large payloads).
    shared: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    agent: str
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 1
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "attempts": self.attempts,
            "duration_s": self.duration_s,
        }


class Agent(Protocol):
    name: str

    def run(self, ctx: AgentContext, inp: dict[str, Any]) -> AgentResult: ...


def dumps_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def run_with_retries(
    *,
    agent_name: str,
    fn,
    max_retries: int = 1,
    backoff_s: float = 0.75,
) -> tuple[dict[str, Any] | None, str | None, int, float]:
    start = time.time()
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            data = fn()
            return data, None, attempt, time.time() - start
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("%s failed (attempt %s/%s): %s", agent_name, attempt, max_retries + 1, exc)
            if attempt <= max_retries:
                time.sleep(backoff_s * (2 ** (attempt - 1)))
                continue
            break
    return None, str(last_exc) if last_exc else "unknown_error", max_retries + 1, time.time() - start


class BaseAgent:
    """Base class for ResQAI agents.

    Agents should implement `_run(ctx, inp)` and inherit retry/logging behavior.
    """

    name: str = "base_agent"
    max_retries: int = 1

    def _run(self, ctx: AgentContext, inp: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def run(self, ctx: AgentContext, inp: dict[str, Any]) -> AgentResult:
        def _do():
            return self._run(ctx, inp)

        data, err, attempts, dur = run_with_retries(
            agent_name=self.name,
            fn=_do,
            max_retries=self.max_retries,
        )
        return AgentResult(
            agent=self.name,
            ok=err is None,
            data=data,
            error=err,
            attempts=attempts,
            duration_s=dur,
        )
