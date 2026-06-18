from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from pyulog import ULog

from flight_log_agent.px4.msg_schema import load_px4_msg_enum_registry
from flight_log_agent.px4.source_snapshot import SourceInput, source_handle
from flight_log_agent.utils import json_safe_value as _json_safe_value
from flight_log_agent.utils import safe_int as _safe_int


OUTPUT_FUNCTION_RE = re.compile(r"^(?P<bus>PWM_(?:MAIN|AUX|FMU)_FUNC)(?P<channel>\d+)$")
CONTROL_SURFACE_TYPE_RE = re.compile(r"^CA_SV_CS(?P<index>\d+)_TYPE$")


def infer_control_surface(log_path: Path, source_path: SourceInput = None) -> dict:
    source_path = source_handle(source_path)
    result = _empty_result()

    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        result["warning"] = f"Control-surface mapping unavailable: failed to parse ULog: {exc}"
        return result

    parameters = getattr(ulog, "initial_parameters", {}) or {}
    topics = _extract_topic_names(ulog)
    servo_types = _extract_control_surface_types(parameters, source_path)
    output_functions = _extract_output_functions(parameters, source_path)

    result["vehicle_type"] = _observed_vehicle_type(ulog, source_path)
    result["assumed_actuator_mapping"] = _build_actuator_mapping(
        servo_types,
        output_functions,
    )
    result["evidence"] = _build_evidence(parameters, topics, servo_types, output_functions)
    result["confidence"] = _infer_confidence(servo_types, output_functions)

    if result["assumed_actuator_mapping"]:
        result["warning"] = (
            "Control-surface mapping is inferred from logged PX4 parameters only; "
            "physical wiring and linkage direction are not confirmed."
        )

    return result


def _empty_result() -> dict:
    return {
        "vehicle_type": "unknown",
        "assumed_actuator_mapping": {},
        "evidence": [],
        "confidence": "low",
        "warning": "Mapping is not confirmed yet.",
    }


def _extract_topic_names(ulog: Any) -> set[str]:
    return {
        str(data.name)
        for data in getattr(ulog, "data_list", []) or []
        if getattr(data, "name", None)
    }


def _extract_control_surface_types(parameters: dict, source_path: Optional[Path]) -> dict[int, dict]:
    configured_count = _safe_int(parameters.get("CA_SV_CS_COUNT"))
    discovered_indexes = {
        int(match.group("index"))
        for name in parameters
        if (match := CONTROL_SURFACE_TYPE_RE.match(name))
    }

    if configured_count is not None:
        discovered_indexes.update(range(configured_count))

    servo_types = {}
    for index in sorted(discovered_indexes):
        param_name = f"CA_SV_CS{index}_TYPE"
        raw_type = _safe_int(parameters.get(param_name))
        label = control_surface_type_labels(source_path).get(raw_type, "unknown")
        servo_types[index + 1] = {
            "parameter": param_name,
            "raw_type": raw_type,
            "control_surface": label,
        }

    return servo_types


def _extract_output_functions(parameters: dict, source_path: Optional[Path]) -> dict[str, dict]:
    output_functions = {}
    definitions = output_function_definitions(source_path)

    for name, value in sorted(parameters.items()):
        match = OUTPUT_FUNCTION_RE.match(name)
        if not match:
            continue

        function_id = _safe_int(value)
        decoded = _decode_output_function(function_id, definitions)
        output_functions[name] = {
            "channel": int(match.group("channel")),
            "function_id": function_id,
            **decoded,
        }

    return output_functions


def _decode_output_function(function_id: Optional[int], definitions: dict[str, Any]) -> dict:
    if function_id is None:
        return {"function": "unknown", "actuator_index": None}

    exact = definitions.get("exact", {}).get(function_id)
    if exact:
        return {
            "function": exact,
            "actuator_index": None,
        }

    for entry in definitions.get("ranges", []):
        start = entry.get("start")
        count = entry.get("count")
        if start is None or count is None:
            continue
        if start <= function_id < start + count:
            return {
                "function": entry["function"],
                "actuator_index": function_id - start + 1,
            }

    return {"function": "unknown", "actuator_index": None}


def _build_actuator_mapping(
    servo_types: dict[int, dict],
    output_functions: dict[str, dict],
) -> dict:
    mapping = {}

    for servo_index, servo in servo_types.items():
        output_channels = [
            {
                "parameter": name,
                "function_id": output["function_id"],
                "function": f"servo_{servo_index}",
            }
            for name, output in output_functions.items()
            if (
                output["function"] == "servo"
                and output["actuator_index"] == servo_index
            )
        ]

        mapping[f"servo_{servo_index}"] = {
            "control_surface": servo["control_surface"],
            "source_parameter": servo["parameter"],
            "source_value": servo["raw_type"],
            "output_channels": output_channels,
        }

    for name, output in output_functions.items():
        if output["function"] != "servo":
            continue

        servo_key = f"servo_{output['actuator_index']}"
        if servo_key in mapping:
            continue

        mapping.setdefault(
            servo_key,
            {
                "control_surface": "unknown",
                "source_parameter": None,
                "source_value": None,
                "output_channels": [],
            },
        )
        mapping[servo_key]["output_channels"].append(
            {
                "parameter": name,
                "function_id": output["function_id"],
                "function": servo_key,
            }
        )

    return mapping


def _build_evidence(
    parameters: dict,
    topics: set[str],
    servo_types: dict[int, dict],
    output_functions: dict[str, dict],
) -> list[str]:
    evidence = []

    if "actuator_servos" in topics:
        evidence.append("ULog contains actuator_servos topic.")

    if "actuator_motors" in topics:
        evidence.append("ULog contains actuator_motors topic.")

    if "vtol_vehicle_status" in topics:
        evidence.append("ULog contains vtol_vehicle_status topic.")

    control_surface_count = parameters.get("CA_SV_CS_COUNT")
    if control_surface_count is not None:
        evidence.append(f"CA_SV_CS_COUNT={_json_safe_value(control_surface_count)}")

    for servo_index, servo in servo_types.items():
        evidence.append(
            f"{servo['parameter']}={servo['raw_type']} "
            f"({servo['control_surface']}) maps to servo_{servo_index}."
        )

    for name, output in output_functions.items():
        if output["function"] in {"servo", "motor"}:
            evidence.append(
                f"{name}={output['function_id']} maps output channel "
                f"to {output['function']}_{output['actuator_index']}."
            )

    return evidence


def _observed_vehicle_type(ulog: Any, source_path: Optional[Path]) -> str:
    values: list[int] = []
    for data in getattr(ulog, "data_list", []) or []:
        if getattr(data, "name", None) != "vehicle_status":
            continue
        observed = (getattr(data, "data", {}) or {}).get("vehicle_type")
        if observed is None:
            continue
        for value in observed:
            parsed = _safe_int(value)
            if parsed is not None and parsed not in values:
                values.append(parsed)

    if len(values) != 1:
        return "unknown"

    entry = load_px4_msg_enum_registry(source_path).get("vehicle_status.vehicle_type") or {}
    constants = entry.get("constants") or {}
    names = [
        _normalize_vehicle_type_label(name)
        for name, value in constants.items()
        if value == values[0]
    ]
    names = [name for name in names if name]
    return names[0] if len(set(names)) == 1 else "unknown"


def _normalize_vehicle_type_label(name: str) -> str:
    label = str(name or "")
    prefix = "VEHICLE_TYPE_"
    if label.startswith(prefix):
        label = label[len(prefix):]
    return _normalize_label(label)


def _infer_confidence(
    servo_types: dict[int, dict],
    output_functions: dict[str, dict],
) -> str:
    if not servo_types and not output_functions:
        return "low"

    if not servo_types:
        return "low"

    servo_outputs = {
        output["actuator_index"]
        for output in output_functions.values()
        if output["function"] == "servo"
    }

    if servo_outputs and servo_outputs.issuperset(servo_types):
        return "medium"

    return "medium-low"


def control_surface_type_labels(source_path: SourceInput) -> dict[int, str]:
    source = source_handle(source_path)
    if source is None:
        return {}
    try:
        text = source.read_text("src/modules/control_allocator/module.yaml")
    except Exception:
        return {}
    return _parse_parameter_enum_values(text, "CA_SV_CS${i}_TYPE")


def output_function_definitions(source_path: SourceInput) -> dict[str, Any]:
    empty = {"exact": {}, "ranges": []}
    source = source_handle(source_path)
    if source is None:
        return empty
    try:
        lines = source.read_text("src/lib/mixer_module/output_functions.yaml").splitlines()
    except Exception:
        return empty
    return _parse_output_function_definitions(lines)


def _parse_output_function_definitions(lines: list[str]) -> dict[str, Any]:
    exact: dict[int, str] = {}
    ranges: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        match = re.match(r"\s{4}(?P<name>[A-Za-z0-9_]+):\s*(?P<value>\d+)\s*$", lines[index])
        if match:
            exact[int(match.group("value"))] = _normalize_label(match.group("name"))
            index += 1
            continue

        range_match = re.match(r"\s{4}(?P<name>[A-Za-z0-9_]+):\s*$", lines[index])
        if not range_match:
            index += 1
            continue

        name = range_match.group("name")
        start = None
        count = None
        index += 1
        while index < len(lines):
            if re.match(r"\s{4}[A-Za-z0-9_]+:", lines[index]):
                break
            start_match = re.match(r"\s{6}start:\s*(?P<value>\d+)\s*$", lines[index])
            count_match = re.match(r"\s{6}count:\s*(?P<value>\d+)\s*$", lines[index])
            if start_match:
                start = int(start_match.group("value"))
            if count_match:
                count = int(count_match.group("value"))
            index += 1

        if start is not None and count is not None:
            ranges.append({
                "function": _normalize_label(name),
                "start": start,
                "count": count,
            })

    return {"exact": exact, "ranges": ranges}


def _parse_parameter_enum_values(text: str, parameter_name: str) -> dict[int, str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not re.match(rf"\s*{re.escape(parameter_name)}\s*:\s*$", line):
            continue
        values_index = None
        for cursor in range(index + 1, min(index + 80, len(lines))):
            if re.match(r"\s{12}values:\s*$", lines[cursor]):
                values_index = cursor
                break
        if values_index is None:
            continue

        values: dict[int, str] = {}
        for cursor in range(values_index + 1, len(lines)):
            value_line = lines[cursor]
            if value_line.strip() and len(value_line) - len(value_line.lstrip(" ")) <= 12:
                break
            match = re.match(r"\s{16}(?P<key>\d+):\s*(?P<label>.+?)\s*$", value_line)
            if match:
                values[int(match.group("key"))] = _normalize_label(match.group("label"))
        return values
    return {}


def _normalize_label(value: str) -> str:
    label = str(value or "").strip().strip("'\"")
    label = label.strip("()")
    label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return label or "unknown"


