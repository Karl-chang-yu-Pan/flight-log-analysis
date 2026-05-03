from __future__ import annotations

import math
import os
from bisect import bisect_left
from pathlib import Path
from typing import Any, Optional

from pyulog import ULog


DEFAULT_HISTOGRAM_BINS = 50


def generate_signal_plot(
    log_path: Path,
    output_dir: Path,
    title: str,
    start_s: float,
    end_s: float,
    signals: list[str],
    purpose: str,
    plot_type: str = "timeseries",
    bins: int = DEFAULT_HISTOGRAM_BINS,
    overlays: Optional[list[dict]] = None,
) -> dict:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"{_safe_plot_filename(title)}.png"

    result = {
        "title": title,
        "path": str(plot_path),
        "purpose": purpose,
        "window_s": [start_s, end_s],
        "signals": signals,
        "plot_type": plot_type,
        "overlays": overlays or [],
        "missing_signals": [],
        "warnings": [],
    }

    if end_s < start_s:
        result["missing_signals"] = list(signals)
        result["warnings"].append("end_s must be greater than or equal to start_s.")
        return result

    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        result["missing_signals"] = list(signals)
        result["warnings"].append(f"failed to parse ULog: {exc}")
        return result

    resolved_signals = []
    for signal in signals:
        resolved = resolve_signal(ulog, signal, start_s, end_s)
        if "warning" in resolved:
            result["missing_signals"].append(signal)
            result["warnings"].append(resolved["warning"])
            continue
        resolved_signals.append(resolved)

    try:
        figure_result = _render_plot(
            plot_path,
            title,
            resolved_signals,
            plot_type,
            bins,
            overlays or [],
        )
    except Exception as exc:
        result["warnings"].append(f"failed to render plot: {exc}")
        return result

    result["warnings"].extend(figure_result["warnings"])
    return result


def resolve_signal(ulog: Any, signal: str, start_s: float, end_s: float) -> dict:
    parsed_signal = _parse_signal(signal)
    if parsed_signal is None:
        return {"warning": f"invalid signal '{signal}'; expected 'topic.field'."}

    topic_name, field_name = parsed_signal

    try:
        dataset = ulog.get_dataset(topic_name)
    except Exception:
        dataset = None

    if dataset is None:
        dataset = _find_dataset(ulog, topic_name)

    if dataset is None:
        return {"warning": f"missing topic for signal '{signal}': {topic_name}"}

    data = getattr(dataset, "data", {}) or {}
    timestamps = data.get("timestamp")
    if timestamps is None:
        return {"warning": f"missing timestamp field for topic '{topic_name}'."}

    values = data.get(field_name)
    if values is None:
        return {"warning": f"missing field for signal '{signal}': {field_name}"}

    times = []
    windowed_values = []
    for timestamp, value in zip(timestamps, values):
        time_s = _timestamp_to_seconds(timestamp)
        if start_s <= time_s <= end_s:
            safe_value = _json_safe_value(value)
            if _is_number(safe_value):
                times.append(time_s)
                windowed_values.append(_normalize_signal_value(topic_name, field_name, float(safe_value)))

    if not times:
        return {"warning": f"no numeric samples for signal '{signal}' in window {start_s}-{end_s}s."}

    return {
        "signal": signal,
        "topic": topic_name,
        "field": field_name,
        "unit": _signal_unit(topic_name, field_name),
        "axis_label": _signal_axis_label(topic_name, field_name),
        "time_s": times,
        "values": windowed_values,
    }


def align_signals(
    x_signal: dict,
    y_signal: dict,
    method: str = "linear",
) -> dict:
    if method not in {"linear", "nearest"}:
        return {"warning": f"unsupported alignment method: {method}"}

    x_times = x_signal["time_s"]
    x_values = x_signal["values"]
    y_times = y_signal["time_s"]
    y_values = y_signal["values"]

    aligned_x = []
    aligned_y = []

    for time_s, x_value in zip(x_times, x_values):
        if time_s < y_times[0] or time_s > y_times[-1]:
            continue

        if method == "nearest":
            y_value = _nearest_value(y_times, y_values, time_s)
        else:
            y_value = _interpolated_value(y_times, y_values, time_s)

        if y_value is None:
            continue

        aligned_x.append(x_value)
        aligned_y.append(y_value)

    if not aligned_x:
        return {
            "warning": (
                f"no overlapping samples between '{x_signal['signal']}' "
                f"and '{y_signal['signal']}'."
            )
        }

    return {
        "x_signal": x_signal["signal"],
        "y_signal": y_signal["signal"],
        "x": aligned_x,
        "y": aligned_y,
    }


def _render_plot(
    plot_path: Path,
    title: str,
    resolved_signals: list[dict],
    plot_type: str,
    bins: int,
    overlays: list[dict],
) -> dict:
    warnings = []
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))

    if plot_type == "timeseries":
        _render_timeseries(ax, resolved_signals, warnings)
    elif plot_type == "xy":
        _render_xy(ax, resolved_signals, warnings)
    elif plot_type == "hist2d":
        _render_hist2d(fig, ax, resolved_signals, bins, warnings)
    else:
        warnings.append(f"unsupported plot_type '{plot_type}', falling back to timeseries.")
        _render_timeseries(ax, resolved_signals, warnings)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _render_overlays(ax, overlays, warnings)
    if hasattr(ax, "legend") and _has_legend_entries(ax):
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    return {"warnings": warnings}


def _render_timeseries(ax: Any, resolved_signals: list[dict], warnings: list[str]) -> None:
    if not resolved_signals:
        warnings.append("no plottable signals.")
        ax.text(0.5, 0.5, "No plottable signals", ha="center", va="center", transform=ax.transAxes)
        return

    for resolved in resolved_signals:
        ax.plot(resolved["time_s"], resolved["values"], label=resolved["signal"])

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(_combined_y_axis_label(resolved_signals))


def _render_xy(ax: Any, resolved_signals: list[dict], warnings: list[str]) -> None:
    if len(resolved_signals) < 2:
        warnings.append("xy plot requires two plottable signals.")
        ax.text(0.5, 0.5, "XY plot requires two signals", ha="center", va="center", transform=ax.transAxes)
        return

    aligned = align_signals(resolved_signals[0], resolved_signals[1])
    if "warning" in aligned:
        warnings.append(aligned["warning"])
        ax.text(0.5, 0.5, "No overlapping samples", ha="center", va="center", transform=ax.transAxes)
        return

    ax.scatter(aligned["x"], aligned["y"], s=12, alpha=0.65, label=f"{aligned['y_signal']} vs {aligned['x_signal']}")
    ax.set_xlabel(_signal_label_with_unit(resolved_signals[0]))
    ax.set_ylabel(_signal_label_with_unit(resolved_signals[1]))


def _render_hist2d(
    fig: Any,
    ax: Any,
    resolved_signals: list[dict],
    bins: int,
    warnings: list[str],
) -> None:
    if len(resolved_signals) < 2:
        warnings.append("hist2d plot requires two plottable signals.")
        ax.text(0.5, 0.5, "2D histogram requires two signals", ha="center", va="center", transform=ax.transAxes)
        return

    aligned = align_signals(resolved_signals[0], resolved_signals[1])
    if "warning" in aligned:
        warnings.append(aligned["warning"])
        ax.text(0.5, 0.5, "No overlapping samples", ha="center", va="center", transform=ax.transAxes)
        return

    histogram = ax.hist2d(aligned["x"], aligned["y"], bins=max(1, int(bins)))
    if hasattr(fig, "colorbar"):
        fig.colorbar(histogram[3], ax=ax, label="Samples")
    ax.set_xlabel(_signal_label_with_unit(resolved_signals[0]))
    ax.set_ylabel(_signal_label_with_unit(resolved_signals[1]))


def _render_overlays(ax: Any, overlays: list[dict], warnings: list[str]) -> None:
    for overlay in overlays:
        try:
            start_s = float(overlay["start_s"])
            end_s = float(overlay.get("end_s", start_s))
        except (KeyError, TypeError, ValueError):
            warnings.append(f"skipped invalid overlay: {overlay}")
            continue

        label = overlay.get("label")
        color = overlay.get("color", "#d55e00")
        alpha = float(overlay.get("alpha", 0.12))

        if end_s == start_s:
            if hasattr(ax, "axvline"):
                ax.axvline(start_s, color=color, alpha=max(alpha, 0.45), linestyle="--", label=label)
        elif hasattr(ax, "axvspan"):
            ax.axvspan(start_s, end_s, color=color, alpha=alpha, label=label)


def _parse_signal(signal: str) -> Optional[tuple[str, str]]:
    if "." not in signal:
        return None

    topic, field = signal.split(".", 1)
    if not topic or not field:
        return None

    return topic, field


def _find_dataset(ulog: Any, topic_name: str) -> Any:
    for dataset in getattr(ulog, "data_list", []) or []:
        if getattr(dataset, "name", None) == topic_name:
            return dataset

    return None


def _nearest_value(times: list[float], values: list[float], target_time: float) -> Optional[float]:
    index = bisect_left(times, target_time)
    if index == 0:
        return values[0]

    if index >= len(times):
        return values[-1]

    before = index - 1
    if abs(times[before] - target_time) <= abs(times[index] - target_time):
        return values[before]

    return values[index]


def _interpolated_value(times: list[float], values: list[float], target_time: float) -> Optional[float]:
    index = bisect_left(times, target_time)
    if index == 0:
        return values[0]

    if index >= len(times):
        return values[-1]

    t0 = times[index - 1]
    t1 = times[index]
    v0 = values[index - 1]
    v1 = values[index]
    if t1 == t0:
        return v0

    ratio = (target_time - t0) / (t1 - t0)
    return v0 + ratio * (v1 - v0)


def _safe_plot_filename(title: str) -> str:
    return title.lower().replace(" ", "_").replace("/", "_")


def _normalize_signal_value(topic_name: str, field_name: str, value: float) -> float:
    if _is_angle_degrees_signal(topic_name, field_name):
        return math.degrees(value)

    if _is_angular_rate_degrees_signal(topic_name, field_name):
        return math.degrees(value)

    if topic_name == "battery_status" and field_name == "remaining":
        return value * 10

    if topic_name == "battery_status" and field_name == "discharged_mah":
        return value / 100

    if topic_name == "battery_status" and field_name == "internal_resistance_estimate":
        return value * 1000

    if topic_name == "estimator_status" and field_name == "time_slip":
        return value * 1_000_000

    if topic_name == "vehicle_gps_position" and field_name == "alt":
        return value * 0.001

    return value


def _signal_unit(topic_name: str, field_name: str) -> Optional[str]:
    if _is_angle_degrees_signal(topic_name, field_name):
        return "deg"

    if _is_angular_rate_degrees_signal(topic_name, field_name):
        return "deg/s"

    if field_name in {"x", "y", "z", "alt", "altitude_msl_m", "baro_alt_meter", "current.alt", "dist_bottom", "current_distance", "variance", "eph", "epv", "hdop", "vdop"}:
        return "m"

    if field_name in {"vx", "vy", "vz", "vel_m_s", "height_rate", "height_rate_setpoint", "true_airspeed_m_s", "true_ground_minus_wind_m_s", "indicated_airspeed_m_s", "true_airspeed_sp", "s_variance_m_s"}:
        return "m/s"

    if field_name.startswith("accelerometer_m_s2") or field_name == "accel_vibration_metric":
        return "m/s^2"

    if field_name.startswith("magnetometer_ga"):
        return "gauss"

    if field_name in {"temperature", "air_temperature_celsius"} or field_name.endswith(".esc_temperature"):
        return "C"

    if field_name in {"voltage_v", "ocv_estimate", "voltage5v_v"} or field_name.startswith("sensors3v3"):
        return "V"

    if field_name == "current_a":
        return "A"

    if field_name == "discharged_mah":
        return "mAh / 100"

    if field_name == "remaining":
        return "0=empty, 10=full"

    if field_name == "internal_resistance_estimate":
        return "mOhm"

    if field_name.endswith(".esc_rpm"):
        return "RPM"

    if field_name == "time_slip":
        return "us"

    return None


def _signal_axis_label(topic_name: str, field_name: str) -> str:
    unit = _signal_unit(topic_name, field_name)
    return f"[{unit}]" if unit else ""


def _signal_label_with_unit(resolved_signal: dict) -> str:
    axis_label = resolved_signal.get("axis_label")
    if axis_label:
        return f"{resolved_signal['signal']} {axis_label}"

    return resolved_signal["signal"]


def _combined_y_axis_label(resolved_signals: list[dict]) -> str:
    axis_labels = {
        resolved.get("axis_label")
        for resolved in resolved_signals
        if resolved.get("axis_label")
    }

    if len(axis_labels) == 1:
        return next(iter(axis_labels))

    return ""


def _is_angle_degrees_signal(topic_name: str, field_name: str) -> bool:
    return (
        topic_name in {
            "vehicle_attitude",
            "vehicle_attitude_setpoint",
            "vehicle_attitude_groundtruth",
            "vehicle_visual_odometry",
        }
        and field_name in {"roll", "pitch", "yaw", "roll_d", "pitch_d", "yaw_d"}
    )


def _is_angular_rate_degrees_signal(topic_name: str, field_name: str) -> bool:
    if topic_name in {"vehicle_angular_velocity", "vehicle_angular_velocity_groundtruth"} and field_name.startswith("xyz["):
        return True

    if topic_name == "vehicle_rates_setpoint" and field_name in {"roll", "pitch", "yaw"}:
        return True

    if topic_name in {"vehicle_attitude", "vehicle_attitude_groundtruth", "vehicle_visual_odometry"} and field_name in {"rollspeed", "pitchspeed", "yawspeed"}:
        return True

    if topic_name == "vehicle_attitude_setpoint" and field_name == "yaw_sp_move_rate":
        return True

    if topic_name == "sensor_combined" and field_name.startswith("gyro_rad["):
        return True

    return False


def _timestamp_to_seconds(timestamp: Any) -> float:
    return float(_json_safe_value(timestamp)) / 1_000_000


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")

    if hasattr(value, "item"):
        return value.item()

    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )


def _load_pyplot() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _has_legend_entries(ax: Any) -> bool:
    if not hasattr(ax, "get_legend_handles_labels"):
        return True

    handles, labels = ax.get_legend_handles_labels()
    return bool(handles and labels)
