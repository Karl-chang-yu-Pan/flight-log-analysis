from __future__ import annotations

import json
from typing import Any, Optional

from flight_log_agent.models import AirframeContext


def build_airframe_context(inventory: dict, control_surface: dict) -> AirframeContext:
    params = inventory.get("parameters") or {}

    px4_git_hash = (
        inventory.get("git_hash")
        or inventory.get("px4_git_hash")
        or inventory.get("firmware_git_hash")
    )
    px4_version = inventory.get("firmware_version") or inventory.get("px4_version")
    px4_tag = inventory.get("px4_tag") or inventory.get("git_tag")

    sys_autostart = _maybe_int(params.get("SYS_AUTOSTART"))
    vehicle_type = infer_vehicle_type_string(inventory, control_surface)

    return AirframeContext(
        px4_git_hash=px4_git_hash,
        px4_version=px4_version,
        px4_tag=px4_tag,
        vehicle_type=vehicle_type,
        sys_autostart=sys_autostart,
        airframe_name=str(control_surface.get("airframe") or control_surface.get("airframe_name") or ""),
        control_surface_summary=summarize_control_surface(control_surface),
    )


def infer_vehicle_type_string(inventory: dict, control_surface: dict) -> str:
    vehicle_type = str(control_surface.get("vehicle_type") or inventory.get("vehicle_type") or "unknown")
    return vehicle_type.strip().lower() or "unknown"


def summarize_control_surface(control_surface: dict) -> str:
    if not control_surface:
        return "unknown"
    keys = [
        "vehicle_type",
        "assumed_actuator_mapping",
        "control_surfaces",
        "confidence",
        "warning",
    ]
    compact = {k: control_surface.get(k) for k in keys if k in control_surface}
    return json.dumps(compact, separators=(",", ":"), default=str)[:2000]


def _maybe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
