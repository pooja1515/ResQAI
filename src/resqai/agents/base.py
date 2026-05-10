from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AgentInput:
    text: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class AgentOutput:
    response_text: str
    language: str | None = None


class Agent(Protocol):
    name: str

    def run(self, inp: AgentInput) -> AgentOutput: ...

