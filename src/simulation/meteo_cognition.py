"""Neuro-symbolic meteorological cognition pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol

from src.cognition.llm_client import LLMClient


class DownscalerProtocol(Protocol):
    """Protocol describing a Prithvi-like downscaler."""

    def infer(self, coarse_tensor: Any) -> Any:  # pragma: no cover - protocol definition
        ...


class SegmenterProtocol(Protocol):
    """Protocol describing a SAM-Geo like segmenter."""

    def segment(self, hi_res_tensor: Any) -> Dict[str, Any]:  # pragma: no cover - protocol definition
        ...


@dataclass
class MeteoCognitionConfig:
    """Configuration for the meteorological cognition pipeline."""

    downscaler: DownscalerProtocol | Callable[[Any], Any]
    segmenter: SegmenterProtocol | Callable[[Any], Dict[str, Any]]
    llm_client: LLMClient
    llm_prompt: str = (
        "You are a meteorological event generator. "
        "Given the segmentation statistics, produce a structured JSON event description."
    )
    default_event_type: str = "storm"
    h3_fallback_resolution: int = 7
    metrics_fields: List[str] = field(default_factory=lambda: ["max_intensity", "mean_intensity"])


@dataclass
class EventObservation:
    """Structured event output consumed by downstream mappers."""

    type: str
    h3_coverage: List[str]
    intensity_index: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MeteoCognitionPipeline:
    """Coordinates downscaling, segmentation, and LLM event synthesis."""

    def __init__(self, config: MeteoCognitionConfig) -> None:
        self.config = config

    def downscale(self, coarse_tensor: Any) -> Any:
        """Run the configured downscaler."""
        downscaler = self.config.downscaler
        if hasattr(downscaler, "infer"):
            return downscaler.infer(coarse_tensor)
        return downscaler(coarse_tensor)  # type: ignore[operator]

    def segment(self, hi_res_tensor: Any) -> Dict[str, Any]:
        """Execute SAM-Geo segmentation and extract statistics."""
        segmenter = self.config.segmenter
        if hasattr(segmenter, "segment"):
            result = segmenter.segment(hi_res_tensor)
        else:
            result = segmenter(hi_res_tensor)  # type: ignore[operator]
        result.setdefault("stats", {})
        result.setdefault("h3_coverage", [])
        return result

    def describe_event(self, segmentation: Dict[str, Any]) -> EventObservation:
        """Use the LLM client to describe a meteorological event."""
        stats = segmentation.get("stats", {})
        prompt = f"{self.config.llm_prompt}\nStats: {json.dumps(stats)}"
        llm_payload = self.config.llm_client.generate(prompt, payload=segmentation)
        event_json = llm_payload.get("json") or {}
        event_type = event_json.get("type") or self.config.default_event_type
        h3_coverage = event_json.get("h3_coverage") or segmentation.get("h3_coverage") or []
        intensity = float(event_json.get("intensity_index") or stats.get("max_intensity") or 0.0)
        metadata = event_json.get("metadata") or stats
        return EventObservation(
            type=event_type,
            h3_coverage=h3_coverage,
            intensity_index=intensity,
            metadata=metadata,
        )

    def run(self, coarse_tensor: Any) -> EventObservation:
        """End-to-end execution from downscaling to event emission."""
        hi_res = self.downscale(coarse_tensor)
        segmentation = self.segment(hi_res)
        return self.describe_event(segmentation)


__all__ = ["MeteoCognitionPipeline", "MeteoCognitionConfig", "EventObservation"]
