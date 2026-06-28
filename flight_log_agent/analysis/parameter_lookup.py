"""Centralized PX4 parameter lookup.

Single helper consumed by the verifier, applicability checks, and inventory
code so that "missing parameter" messages, numeric coercion, and PX4-name
validation are consistent everywhere.

Type preservation: PX4 declares parameters as ``PARAM_DEFINE_INT32`` or
``PARAM_DEFINE_FLOAT``, and pyulog stores them as native Python int / float.
``kind="auto"`` (the default) returns that native type without silent
promotion — important for INT32 parameters used as bitmasks (``SYS_AUTOSTART``,
``RTL_TYPE``).
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Literal, Optional, Tuple

import numpy as _np


Kind = Literal["auto", "int", "float", "bool"]


_PX4_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")

# PX4 enforces a 16-character upper bound on parameter names via the
# parameter storage subsystem (see `param_name_t` in src/lib/parameters/).
PX4_PARAM_NAME_LIMIT = 16


# C-stdlib numeric constants that PX4 source uses as comparison RHSes.
# The set is bounded by the C standard / IEEE 754 and version-invariant
# across PX4 releases, so a fixed lookup is appropriate. Values come
# from Python's stdlib + numpy at module load (not transcribed by hand)
# so the table stays in sync with the IEEE 754 definitions.
CXX_STDLIB_CONSTANTS: dict[str, float] = {
    "FLT_EPSILON": float(_np.finfo(_np.float32).eps),
    "DBL_EPSILON": float(_np.finfo(_np.float64).eps),
    "FLT_MAX":     float(_np.finfo(_np.float32).max),
    "FLT_MIN":     float(_np.finfo(_np.float32).tiny),
    "DBL_MAX":     float(_np.finfo(_np.float64).max),
    "DBL_MIN":     float(_np.finfo(_np.float64).tiny),
    "M_PI":        math.pi,
    "M_PI_2":      math.pi / 2.0,
    "M_PI_4":      math.pi / 4.0,
    "M_E":         math.e,
    "INFINITY":    math.inf,
    "NAN":         math.nan,
}


def lookup_cxx_constant(name: str) -> Optional[float]:
    """Return the value of a C-stdlib numeric constant by name.

    ``None`` when the name is not in :data:`CXX_STDLIB_CONSTANTS`. The
    predicate parser and ``resolve_numeric_value`` consult this as the
    last fallback so identifiers like ``FLT_EPSILON`` and ``M_PI`` —
    which appear in PX4 source as comparison RHSes but are not defined
    in the PX4 tree — resolve to their numeric values.
    """
    if not isinstance(name, str):
        return None
    return CXX_STDLIB_CONSTANTS.get(name.strip())


def is_px4_parameter_name(name: str) -> bool:
    """Return True if ``name`` has the PX4 parameter naming shape.

    Uppercase, snake_case, requires at least one underscore, max 16 chars.
    Used to filter LLM-emitted tokens before treating them as parameter
    lookups (so that math identifiers and PX4 enum constants don't get
    misclassified as missing parameters).
    """
    if not isinstance(name, str) or not name:
        return False
    if len(name) > PX4_PARAM_NAME_LIMIT:
        return False
    if "_" not in name:
        return False
    return bool(_PX4_NAME_RE.fullmatch(name))


def get_parameter(
    inventory: Any,
    name: str,
    *,
    kind: Kind = "auto",
) -> Tuple[Optional[Any], Optional[str]]:
    """Look up a PX4 parameter from ``inventory``.

    Returns ``(value, unresolved_reason)``:

    - On success, ``value`` is the parameter's native value (or the coerced
      form when ``kind`` is anything other than ``"auto"``) and
      ``unresolved_reason`` is ``None``.
    - When the parameter is missing or the coercion fails, ``value`` is
      ``None`` and ``unresolved_reason`` is a human-readable string suitable
      for emitting into a check_result message.

    ``inventory`` may be the full inventory dict (with a ``"parameters"``
    sub-key) or the parameters dict itself, since the verifier mixes both
    conventions.
    """
    if not isinstance(name, str) or not name:
        return None, "parameter name is empty"
    if not is_px4_parameter_name(name):
        return None, f"{name} is not a PX4 parameter name"

    parameters = _parameters_from(inventory)
    if name not in parameters:
        return None, (
            f"parameter {name} was not set in this log "
            f"(defaults are not recoverable from ULog)"
        )

    raw = parameters[name]
    if kind == "auto":
        return raw, None
    if kind == "int":
        coerced = _coerce_int(raw)
        if coerced is None:
            return None, f"parameter {name} is not coercible to int"
        return coerced, None
    if kind == "float":
        coerced = _coerce_float(raw)
        if coerced is None:
            return None, f"parameter {name} is not numeric"
        return coerced, None
    if kind == "bool":
        coerced = _coerce_int(raw)
        if coerced is None:
            return None, f"parameter {name} is not coercible to bool"
        return bool(coerced), None
    return None, f"unknown parameter kind: {kind}"


def has_parameter(inventory: Any, name: str) -> bool:
    """Return True if ``name`` is present in the inventory's parameters."""
    return name in _parameters_from(inventory)


def resolve_numeric_value(
    value: Any,
    inventory: Any,
    *,
    kind: Kind = "float",
) -> Tuple[Optional[float], Optional[str]]:
    """Coerce ``value`` to a numeric, looking up PX4 parameter names.

    Accepts either a numeric literal or a string that may be a PX4
    parameter name. Returns ``(numeric_value, missing_reason)`` with
    the same convention as :func:`get_parameter`.

    Used by check handlers that take a comparison value emitted by the
    LLM — the LLM frequently writes parameter names like
    ``FW_AIRSPD_TRIM`` where a float is expected; this helper resolves
    them against the inventory's parameters in one place.
    """
    if value is None:
        return None, None
    if isinstance(value, bool):
        return float(value), None
    if isinstance(value, (int, float)):
        coerced = _coerce_float(value)
        if coerced is None:
            return None, f"value {value!r} is not a finite number"
        return coerced, None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, None
        coerced = _coerce_float(stripped)
        if coerced is not None:
            return coerced, None
        if is_px4_parameter_name(stripped):
            param_value, missing = get_parameter(inventory, stripped, kind=kind)
            if param_value is not None:
                return param_value, None
            stdlib_value = lookup_cxx_constant(stripped)
            if stdlib_value is not None:
                return stdlib_value, None
            return None, missing
        stdlib_value = lookup_cxx_constant(stripped)
        if stdlib_value is not None:
            return stdlib_value, None
        return None, f"value {value!r} is not numeric or a PX4 parameter name"
    return None, f"value {value!r} is not numeric or a PX4 parameter name"


def filter_present_parameters(
    inventory: Any,
    names: Iterable[str],
) -> list[str]:
    """Return ``names`` filtered to those that exist in inventory."""
    parameters = _parameters_from(inventory)
    return [name for name in names if name in parameters]


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _parameters_from(inventory: Any) -> dict[str, Any]:
    if isinstance(inventory, dict):
        nested = inventory.get("parameters")
        if isinstance(nested, dict):
            return nested
        return inventory
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not _isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            return int(value, 0)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if not _isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _isfinite(value: float) -> bool:
    import math
    return math.isfinite(value)
