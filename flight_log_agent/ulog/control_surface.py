from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from pyulog import ULog


CONTROL_SURFACE_TYPE_LABELS = {
    0: "not_set",
    1: "left_aileron",
    2: "right_aileron",
    3: "elevator",
    4: "rudder",
    5: "left_elevon",
    6: "right_elevon",
    7: "left_v_tail",
    8: "right_v_tail",
    9: "left_flap",
    10: "right_flap",
    11: "airbrake",
    12: "custom",
    13: "left_a_tail",
    14: "right_a_tail",
    15: "single_channel_aileron",
    16: "steering_wheel",
    17: "left_spoiler",
    18: "right_spoiler",
}

OUTPUT_FUNCTION_DISABLED = {
    0: "disabled",
    1: "constant_min",
    2: "constant_max",
}

OUTPUT_FUNCTION_RE = re.compile(r"^(?P<bus>PWM_(?:MAIN|AUX|FMU)_FUNC)(?P<channel>\d+)$")
CONTROL_SURFACE_TYPE_RE = re.compile(r"^CA_SV_CS(?P<index>\d+)_TYPE$")


def infer_control_surface(log_path: Path, source_path: Optional[Path] = None) -> dict:
    result = _empty_result()

    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        result["warning"] = f"Control-surface mapping unavailable: failed to parse ULog: {exc}"
        return result

    parameters = getattr(ulog, "initial_parameters", {}) or {}
    topics = _extract_topic_names(ulog)
    servo_types = _extract_control_surface_types(parameters)
    output_functions = _extract_output_functions(parameters)

    result["vehicle_type"] = _infer_vehicle_type(parameters, topics, servo_types)
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


def _extract_control_surface_types(parameters: dict) -> dict[int, dict]:
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
        label = CONTROL_SURFACE_TYPE_LABELS.get(raw_type, "unknown")
        servo_types[index + 1] = {
            "parameter": param_name,
            "raw_type": raw_type,
            "control_surface": label,
        }

    return servo_types


def _extract_output_functions(parameters: dict) -> dict[str, dict]:
    output_functions = {}

    for name, value in sorted(parameters.items()):
        match = OUTPUT_FUNCTION_RE.match(name)
        if not match:
            continue

        function_id = _safe_int(value)
        decoded = _decode_output_function(function_id)
        output_functions[name] = {
            "channel": int(match.group("channel")),
            "function_id": function_id,
            **decoded,
        }

    return output_functions


def _decode_output_function(function_id: Optional[int]) -> dict:
    if function_id is None:
        return {"function": "unknown", "actuator_index": None}

    if function_id in OUTPUT_FUNCTION_DISABLED:
        return {
            "function": OUTPUT_FUNCTION_DISABLED[function_id],
            "actuator_index": None,
        }

    if 101 <= function_id <= 112:
        return {
            "function": "motor",
            "actuator_index": function_id - 100,
        }

    if 201 <= function_id <= 208:
        return {
            "function": "servo",
            "actuator_index": function_id - 200,
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


def _infer_vehicle_type(
    parameters: dict,
    topics: set[str],
    servo_types: dict[int, dict],
) -> str:
    parameter_names = set(parameters)

    if "vtol_vehicle_status" in topics or any(name.startswith("VT_") for name in parameter_names):
        return "vtol"

    has_fw_params = any(name.startswith("FW_") for name in parameter_names)
    has_mc_params = any(name.startswith(("MC_", "MPC_")) for name in parameter_names)

    if servo_types and has_fw_params:
        return "fixed_wing"

    if servo_types:
        return "fixed_wing_or_vtol"

    if has_fw_params:
        return "fixed_wing"

    rotor_count = _safe_int(parameters.get("CA_ROTOR_COUNT"))
    if rotor_count and rotor_count > 0:
        return "multicopter_or_vtol"

    if has_mc_params:
        return "multicopter"

    return "unknown"


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


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    if hasattr(value, "item"):
        value = value.item()

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()

    return value
