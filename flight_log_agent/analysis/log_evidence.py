from __future__ import annotations

import bisect
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from pyulog import ULog

from flight_log_agent.symbols import parse_signal_reference


class EvidenceSample(BaseModel):
    time_s: float
    value: Any


class EvidenceSeries(BaseModel):
    signal: str
    topic: str
    field: str
    multi_id: int
    samples: list[EvidenceSample] = Field(default_factory=list)


class EvidenceResolution(BaseModel):
    reference: str
    status: Literal["observed", "ambiguous", "unavailable"]
    series: Optional[EvidenceSeries] = None
    candidates: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


class ParameterResolution(BaseModel):
    name: str
    status: Literal["observed", "unavailable"]
    value: Any = None
    reason: Optional[str] = None


class ULogEvidenceIndex:
    def __init__(self, ulog: Any, signal_references: Optional[list[str]] = None) -> None:
        self.parameters = dict(getattr(ulog, "initial_parameters", {}) or {})
        self._series: dict[tuple[str, int, str], EvidenceSeries] = {}
        requested = {
            parsed
            for reference in signal_references or []
            if (parsed := parse_signal_reference(reference)) is not None
        }
        for data in getattr(ulog, "data_list", []) or []:
            topic = str(getattr(data, "name", "") or "")
            multi_id = int(getattr(data, "multi_id", 0) or 0)
            values_by_field = getattr(data, "data", {}) or {}
            timestamps = values_by_field.get("timestamp")
            if not topic or timestamps is None:
                continue
            for field, values in values_by_field.items():
                if field == "timestamp":
                    continue
                if requested and not any(
                    requested_topic == topic
                    and requested_field == field
                    and (requested_instance is None or requested_instance == multi_id)
                    for requested_topic, requested_instance, requested_field in requested
                ):
                    continue
                samples = [
                    EvidenceSample(time_s=timestamp_to_seconds(timestamp), value=json_safe_value(value))
                    for timestamp, value in zip(timestamps, values)
                ]
                self._series[(topic, multi_id, str(field))] = EvidenceSeries(
                    signal=format_signal(topic, str(field), multi_id),
                    topic=topic,
                    field=str(field),
                    multi_id=multi_id,
                    samples=samples,
                )

    @classmethod
    def from_path(cls, log_path: Path, signal_references: Optional[list[str]] = None) -> "ULogEvidenceIndex":
        topics = sorted({
            parsed[0]
            for reference in signal_references or []
            if (parsed := parse_signal_reference(reference)) is not None
        })
        ulog = ULog(str(log_path), topics) if topics else ULog(str(log_path))
        return cls(ulog, signal_references=signal_references)

    def resolve_signal(self, reference: str) -> EvidenceResolution:
        parsed = parse_signal_reference(reference)
        if parsed is None:
            return EvidenceResolution(
                reference=reference,
                status="unavailable",
                reason="invalid signal reference",
            )
        topic, requested_instance, field = parsed
        candidates = [
            series
            for (candidate_topic, multi_id, candidate_field), series in self._series.items()
            if candidate_topic == topic
            and candidate_field == field
            and (requested_instance is None or multi_id == requested_instance)
        ]
        if len(candidates) == 1:
            return EvidenceResolution(
                reference=reference,
                status="observed",
                series=candidates[0],
                candidates=[candidates[0].signal],
            )
        if len(candidates) > 1:
            return EvidenceResolution(
                reference=reference,
                status="ambiguous",
                candidates=sorted(series.signal for series in candidates),
                reason="multiple logged topic instances contain the exact field",
            )
        return EvidenceResolution(
            reference=reference,
            status="unavailable",
            reason="exact topic, instance, and field are not logged",
        )

    def resolve_parameter(self, name: str) -> ParameterResolution:
        if name in self.parameters:
            return ParameterResolution(name=name, status="observed", value=json_safe_value(self.parameters[name]))
        return ParameterResolution(name=name, status="unavailable", reason="parameter is not present in the log")

    def samples(
        self,
        reference: str,
        *,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
    ) -> EvidenceResolution:
        resolution = self.resolve_signal(reference)
        if resolution.status != "observed" or resolution.series is None:
            return resolution
        if start_s is None and end_s is None:
            return resolution
        resolution.series = resolution.series.model_copy(
            update={
                "samples": [
                    sample
                    for sample in resolution.series.samples
                    if (start_s is None or sample.time_s >= start_s)
                    and (end_s is None or sample.time_s <= end_s)
                ]
            }
        )
        return resolution

    def sample_at(
        self,
        reference: str,
        time_s: float,
        *,
        policy: Literal["prior", "nearest"] = "prior",
    ) -> tuple[EvidenceResolution, Optional[EvidenceSample]]:
        resolution = self.resolve_signal(reference)
        if resolution.status != "observed" or resolution.series is None or not resolution.series.samples:
            return resolution, None
        samples = resolution.series.samples
        times = [sample.time_s for sample in samples]
        if policy == "prior":
            index = bisect.bisect_right(times, time_s) - 1
            return resolution, samples[index] if index >= 0 else None
        insertion = bisect.bisect_left(times, time_s)
        choices = [
            index for index in (insertion - 1, insertion)
            if 0 <= index < len(samples)
        ]
        if not choices:
            return resolution, None
        index = min(choices, key=lambda item: abs(times[item] - time_s))
        return resolution, samples[index]


def format_signal(topic: str, field: str, multi_id: int) -> str:
    return f"{topic}[{multi_id}].{field}"


def timestamp_to_seconds(timestamp: Any) -> float:
    return float(json_safe_value(timestamp)) / 1_000_000.0


def json_safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if hasattr(value, "item"):
        return value.item()
    return value
