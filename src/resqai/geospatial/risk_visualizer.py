from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskStyle:
    color: str
    icon: str = "info-sign"
    fill_color: str | None = None
    opacity: float = 0.9
    fill_opacity: float = 0.35
    weight: int = 3
    radius: int = 18
    intensity: float = 0.5  # 0..1 for heatmaps/glows


def risk_to_style(risk: str | None) -> RiskStyle:
    r = (risk or "").strip().lower()
    # Required palette:
    # - green = low
    # - orange = moderate/medium
    # - red = high/critical
    if r in {"critical", "severe"}:
        return RiskStyle(color="red", fill_color="red", icon="warning-sign", fill_opacity=0.55, weight=5, radius=28, intensity=1.0)
    if r in {"high"}:
        return RiskStyle(color="red", fill_color="red", icon="warning-sign", fill_opacity=0.45, weight=4, radius=24, intensity=0.85)
    if r in {"medium", "moderate"}:
        return RiskStyle(color="orange", fill_color="orange", icon="info-sign", fill_opacity=0.40, weight=4, radius=22, intensity=0.6)
    if r in {"low"}:
        return RiskStyle(color="green", fill_color="green", icon="ok-sign", fill_opacity=0.30, weight=3, radius=18, intensity=0.25)
    return RiskStyle(color="gray", fill_color="gray", icon="info-sign", fill_opacity=0.25, weight=2, radius=16, intensity=0.15)


def class_to_style(predicted_class: str | None) -> RiskStyle:
    c = (predicted_class or "").strip().lower()
    if c in {"severe", "flooded"}:
        return RiskStyle(color="red", fill_color="red", icon="tint", fill_opacity=0.50, weight=5, radius=28, intensity=0.95)
    if c in {"mild"}:
        return RiskStyle(color="orange", fill_color="orange", icon="tint", fill_opacity=0.40, weight=4, radius=22, intensity=0.6)
    if c in {"no_flood", "non_flooded"}:
        return RiskStyle(color="green", fill_color="green", icon="ok-sign", fill_opacity=0.30, weight=3, radius=18, intensity=0.2)
    return RiskStyle(color="gray", fill_color="gray", icon="info-sign", fill_opacity=0.25, weight=2, radius=16, intensity=0.15)


def escalation_to_style(flag: bool | None) -> RiskStyle:
    if flag is True:
        return RiskStyle(color="red", fill_color="red", icon="exclamation-sign", fill_opacity=0.55, weight=5, radius=28, intensity=0.95)
    if flag is False:
        return RiskStyle(color="green", fill_color="green", icon="ok-sign", fill_opacity=0.25, weight=3, radius=18, intensity=0.2)
    return RiskStyle(color="gray", fill_color="gray", icon="info-sign", fill_opacity=0.25, weight=2, radius=16, intensity=0.15)


def badge_text(overall_risk: str | None, weather_escalation: bool | None) -> str | None:
    r = (overall_risk or "").strip().lower()
    if weather_escalation is True:
        return "ESCALATING"
    if r in {"critical", "severe"}:
        return "CRITICAL"
    if r in {"high"}:
        return "HIGH PRIORITY"
    if r in {"moderate", "medium"}:
        return "MODERATE"
    if r in {"low"}:
        return "LOW"
    return None
