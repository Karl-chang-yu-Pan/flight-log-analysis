from __future__ import annotations

import math
from bisect import bisect_left
from pathlib import Path
from typing import Any

from pyulog import ULog

from ulog_plots import (
    _json_safe_value,
    _timestamp_to_seconds,
    prepare_ulog_for_plotting,
    resolve_signal,
)


DEFAULT_MAX_POINTS = 800

COLORS8 = ["#d55e00", "#009e73", "#55b4e9", "#000000", "#e69f00", "#0072b2", "#cc79a7", "#f0e442"]
COLORS3 = COLORS8[:3]
COLORS2 = COLORS8[:2]
COLOR_GRAY = "#464646"
GPS_PROJECTED_COLOR = "#56b4e9"
MISSION_SETPOINT_COLOR = "#cc79a7"

FLIGHT_MODE_STYLES = {
    0: ("Manual", "#cc0000"),
    1: ("Altitude", "#eecc00"),
    8: ("Altitude Cruise", "#eecc00"),
    2: ("Position", "#00cc33"),
    6: ("Position (Slow)", "#00cc33"),
    10: ("Acro", "#66cc00"),
    14: ("Offboard", "#00cccc"),
    15: ("Stabilized", "#0033cc"),
    3: ("Mission", "#6600cc"),
    4: ("Loiter", "#6600cc"),
    5: ("Return to Land", "#6600cc"),
    12: ("Descend", "#6600cc"),
    13: ("Terminate", "#6600cc"),
    17: ("Takeoff", "#6600cc"),
    18: ("Land", "#6600cc"),
    19: ("Follow Target", "#6600cc"),
    20: ("Precision Land", "#6600cc"),
    21: ("Orbit", "#6600cc"),
    22: ("VTOL Takeoff", "#6600cc"),
    23: ("External 1", "#9700cc"),
    24: ("External 2", "#9700cc"),
    25: ("External 3", "#9700cc"),
    26: ("External 4", "#9700cc"),
    27: ("External 5", "#9700cc"),
    28: ("External 6", "#9700cc"),
    29: ("External 7", "#9700cc"),
    30: ("External 8", "#9700cc"),
}

VTOL_MODE_STYLES = {
    1: ("Transition", "#cc0000"),
    2: ("Fixed-Wing", "#eecc00"),
    3: ("Multicopter", "#0033cc"),
}


LOCAL_POSITION_PLOT = {
    "id": "local_position_2d",
    "title": "Local Position 2D",
    "kind": "local_position",
    "traces": [
        {
            "key": "position",
            "label": "Estimated",
            "x": "vehicle_local_position.x",
            "y": "vehicle_local_position.y",
            "z": "vehicle_local_position.z",
            "color": COLORS2[0],
        },
        {
            "key": "setpoint",
            "label": "Setpoint",
            "x": "vehicle_local_position_setpoint.x",
            "y": "vehicle_local_position_setpoint.y",
            "z": "vehicle_local_position_setpoint.z",
            "color": COLORS2[1],
        },
        {
            "key": "groundtruth",
            "label": "Groundtruth",
            "x": "vehicle_local_position_groundtruth.x",
            "y": "vehicle_local_position_groundtruth.y",
            "z": "vehicle_local_position_groundtruth.z",
            "color": COLOR_GRAY,
        },
        {
            "key": "gps_projected",
            "label": "GPS (projected)",
            "source": "vehicle_gps_position",
            "color": GPS_PROJECTED_COLOR,
        },
        {
            "key": "position_setpoints",
            "label": "Position Setpoints",
            "source": "position_setpoint_triplet",
            "color": MISSION_SETPOINT_COLOR,
            "marker_only": True,
        },
    ],
}


TIMESERIES_PLOTS = [
    {
        "id": "altitude_estimate",
        "title": "Altitude Estimate",
        "series": [
            {"label": "GPS altitude", "signals": ["vehicle_gps_position.altitude_msl_m", "vehicle_gps_position.alt"]},
            {"label": "Barometer", "signals": ["vehicle_air_data.baro_alt_meter", "sensor_combined.baro_alt_meter"]},
            {"label": "Global position", "signals": ["vehicle_global_position.alt"]},
            {"label": "Altitude setpoint", "signals": ["position_setpoint_triplet.current.alt"]},
        ],
    },
    {
        "id": "attitude",
        "title": "Roll/Pitch/Yaw Angle",
        "series": [
            {"label": "Roll", "signals": ["vehicle_attitude.roll"]},
            {"label": "Pitch", "signals": ["vehicle_attitude.pitch"]},
            {"label": "Yaw", "signals": ["vehicle_attitude.yaw"]},
            {"label": "Roll setpoint", "signals": ["vehicle_attitude_setpoint.roll_d"]},
            {"label": "Pitch setpoint", "signals": ["vehicle_attitude_setpoint.pitch_d"]},
            {"label": "Yaw setpoint", "signals": ["vehicle_attitude_setpoint.yaw_d"]},
        ],
    },
    {
        "id": "angular_rate",
        "title": "Roll/Pitch/Yaw Angular Rate",
        "series": [
            {"label": "Roll rate", "signals": ["vehicle_angular_velocity.xyz[0]", "vehicle_attitude.rollspeed"]},
            {"label": "Pitch rate", "signals": ["vehicle_angular_velocity.xyz[1]", "vehicle_attitude.pitchspeed"]},
            {"label": "Yaw rate", "signals": ["vehicle_angular_velocity.xyz[2]", "vehicle_attitude.yawspeed"]},
            {"label": "Roll rate setpoint", "signals": ["vehicle_rates_setpoint.roll"]},
            {"label": "Pitch rate setpoint", "signals": ["vehicle_rates_setpoint.pitch"]},
            {"label": "Yaw rate setpoint", "signals": ["vehicle_rates_setpoint.yaw"]},
        ],
    },
    {
        "id": "local_position",
        "title": "Local Position X/Y/Z",
        "series": [
            {"label": "X", "signals": ["vehicle_local_position.x"]},
            {"label": "Y", "signals": ["vehicle_local_position.y"]},
            {"label": "Z", "signals": ["vehicle_local_position.z"]},
            {"label": "X setpoint", "signals": ["vehicle_local_position_setpoint.x"]},
            {"label": "Y setpoint", "signals": ["vehicle_local_position_setpoint.y"]},
            {"label": "Z setpoint", "signals": ["vehicle_local_position_setpoint.z"]},
        ],
    },
    {
        "id": "velocity",
        "title": "Velocity",
        "series": [
            {"label": "VX", "signals": ["vehicle_local_position.vx"]},
            {"label": "VY", "signals": ["vehicle_local_position.vy"]},
            {"label": "VZ", "signals": ["vehicle_local_position.vz"]},
            {"label": "VX setpoint", "signals": ["vehicle_local_position_setpoint.vx"]},
            {"label": "VY setpoint", "signals": ["vehicle_local_position_setpoint.vy"]},
            {"label": "VZ setpoint", "signals": ["vehicle_local_position_setpoint.vz"]},
        ],
    },
    {
        "id": "visual_odometry_position",
        "title": "Visual Odometry Position",
        "series": [
            {"label": "X", "signals": ["vehicle_visual_odometry.x"]},
            {"label": "Y", "signals": ["vehicle_visual_odometry.y"]},
            {"label": "Z", "signals": ["vehicle_visual_odometry.z"]},
            {"label": "Groundtruth X", "signals": ["vehicle_local_position_groundtruth.x"]},
            {"label": "Groundtruth Y", "signals": ["vehicle_local_position_groundtruth.y"]},
            {"label": "Groundtruth Z", "signals": ["vehicle_local_position_groundtruth.z"]},
        ],
    },
    {
        "id": "airspeed",
        "title": "Airspeed",
        "series": [
            {"label": "True airspeed", "signals": ["airspeed_validated.true_airspeed_m_s", "airspeed.true_airspeed"]},
            {"label": "Indicated airspeed", "signals": ["airspeed.indicated_airspeed_m_s"]},
            {"label": "GPS speed", "signals": ["vehicle_gps_position.vel_m_s"]},
            {"label": "Airspeed setpoint", "signals": ["tecs_status.true_airspeed_sp", "tecs_status.airspeed_sp"]},
        ],
    },
    {
        "id": "tecs",
        "title": "TECS",
        "series": [
            {"label": "Height rate", "signals": ["tecs_status.height_rate"]},
            {"label": "Height rate setpoint", "signals": ["tecs_status.height_rate_setpoint"]},
        ],
    },
    {
        "id": "manual_control",
        "title": "Manual Control Inputs",
        "y_range": [-1.1, 1.1],
        "series": [
            {"label": "Roll", "signals": ["manual_control_setpoint.roll", "manual_control_setpoint.y"]},
            {"label": "Pitch", "signals": ["manual_control_setpoint.pitch", "manual_control_setpoint.x"]},
            {"label": "Yaw", "signals": ["manual_control_setpoint.yaw", "manual_control_setpoint.r"]},
            {"label": "Throttle", "signals": ["manual_control_setpoint.throttle", "manual_control_setpoint.z"]},
            {"label": "Mode slot", "signals": ["manual_control_switches.mode_slot"]},
            {"label": "Kill switch", "signals": ["manual_control_switches.kill_switch"]},
        ],
    },
    {
        "id": "actuator_controls",
        "title": "Actuator Controls",
        "y_range": [-1.0, 1.0],
        "series": [
            {"label": "Torque roll", "signals": ["vehicle_torque_setpoint.xyz[0]", "actuator_controls_0.control[0]"]},
            {"label": "Torque pitch", "signals": ["vehicle_torque_setpoint.xyz[1]", "actuator_controls_0.control[1]"]},
            {"label": "Torque yaw", "signals": ["vehicle_torque_setpoint.xyz[2]", "actuator_controls_0.control[2]"]},
            {"label": "Thrust", "signals": ["vehicle_thrust_setpoint.xyz[2]", "actuator_controls_0.control[3]"]},
        ],
    },
    {
        "id": "motor_outputs",
        "title": "Motor Outputs",
        "y_range": [-1.0, 1.0],
        "series": [
            {"label": f"Motor {index + 1}", "signals": [f"actuator_motors.control[{index}]"]}
            for index in range(8)
        ],
    },
    {
        "id": "servo_outputs",
        "title": "Servo Outputs",
        "y_range": [-1.0, 1.0],
        "series": [
            {"label": f"Servo {index + 1}", "signals": [f"actuator_servos.control[{index}]"]}
            for index in range(8)
        ],
    },
    {
        "id": "raw_acceleration",
        "title": "Raw Acceleration",
        "series": [
            {"label": "X", "signals": ["sensor_combined.accelerometer_m_s2[0]"]},
            {"label": "Y", "signals": ["sensor_combined.accelerometer_m_s2[1]"]},
            {"label": "Z", "signals": ["sensor_combined.accelerometer_m_s2[2]"]},
        ],
    },
    {
        "id": "vibration",
        "title": "Vibration Metrics",
        "series": [
            {"label": f"IMU {index}", "signals": [f"vehicle_imu_status.accel_vibration_metric"]}
            for index in range(1)
        ],
    },
    {
        "id": "raw_gyro",
        "title": "Raw Angular Speed",
        "series": [
            {"label": "Roll rate", "signals": ["sensor_combined.gyro_rad[0]"]},
            {"label": "Pitch rate", "signals": ["sensor_combined.gyro_rad[1]"]},
            {"label": "Yaw rate", "signals": ["sensor_combined.gyro_rad[2]"]},
        ],
    },
    {
        "id": "magnetic_field",
        "title": "Raw Magnetic Field Strength",
        "series": [
            {"label": "X", "signals": ["vehicle_magnetometer.magnetometer_ga[0]", "sensor_combined.magnetometer_ga[0]"]},
            {"label": "Y", "signals": ["vehicle_magnetometer.magnetometer_ga[1]", "sensor_combined.magnetometer_ga[1]"]},
            {"label": "Z", "signals": ["vehicle_magnetometer.magnetometer_ga[2]", "sensor_combined.magnetometer_ga[2]"]},
        ],
    },
    {
        "id": "gps_uncertainty",
        "title": "GPS Uncertainty",
        "y_range": [0.0, 40.0],
        "series": [
            {"label": "EPH", "signals": ["vehicle_gps_position.eph"]},
            {"label": "EPV", "signals": ["vehicle_gps_position.epv"]},
            {"label": "HDOP", "signals": ["vehicle_gps_position.hdop"]},
            {"label": "VDOP", "signals": ["vehicle_gps_position.vdop"]},
            {"label": "Satellites", "signals": ["vehicle_gps_position.satellites_used"]},
        ],
    },
    {
        "id": "gps_noise_jamming",
        "title": "GPS Noise & Jamming",
        "series": [
            {"label": "Noise", "signals": ["vehicle_gps_position.noise_per_ms"]},
            {"label": "Jamming", "signals": ["vehicle_gps_position.jamming_indicator"]},
        ],
    },
    {
        "id": "power",
        "title": "Power",
        "series": [
            {"label": "Voltage", "signals": ["battery_status.voltage_v"]},
            {"label": "Current", "signals": ["battery_status.current_a"]},
            {"label": "Discharged", "signals": ["battery_status.discharged_mah"]},
            {"label": "Remaining", "signals": ["battery_status.remaining"]},
            {"label": "5V rail", "signals": ["system_power.voltage5v_v"]},
        ],
    },
    {
        "id": "temperature",
        "title": "Temperature",
        "series": [
            {"label": "Barometer", "signals": ["sensor_baro.temperature"]},
            {"label": "Airspeed", "signals": ["airspeed.air_temperature_celsius"]},
            {"label": "Battery", "signals": ["battery_status.temperature"]},
        ],
    },
    {
        "id": "estimator_flags",
        "title": "Estimator Flags",
        "series": [
            {"label": "Health flags", "signals": ["estimator_status.health_flags"]},
            {"label": "Timeout flags", "signals": ["estimator_status.timeout_flags"]},
            {"label": "Innovation check flags", "signals": ["estimator_status.innovation_check_flags"]},
        ],
    },
    {
        "id": "failsafe_flags",
        "title": "Failsafe Flags",
        "series": [
            {"label": "Failsafe", "signals": ["vehicle_status.failsafe"]},
            {"label": "User took over", "signals": ["failsafe_flags.failsafe_and_user_took_over"]},
            {"label": "Offboard lost", "signals": ["failsafe_flags.offboard_control_signal_lost"]},
        ],
    },
    {
        "id": "cpu_ram",
        "title": "CPU & RAM",
        "y_range": [0.0, 1.0],
        "series": [
            {"label": "CPU load", "signals": ["cpuload.load"]},
            {"label": "RAM usage", "signals": ["cpuload.ram_usage"]},
        ],
    },
]


def build_interactive_plot_payload(
    log_path: str | Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        return {
            "plots": [],
            "time_range_s": [0.0, 0.0],
            "warnings": [f"failed to parse ULog: {exc}"],
        }

    prepare_ulog_for_plotting(ulog)
    start_s, end_s = _log_time_range(ulog)
    overlays = _build_overlays(ulog, start_s, end_s)

    plots: list[dict[str, Any]] = []
    local_plot = _build_local_position_plot(ulog, start_s, end_s, max_points)
    if local_plot is not None:
        plots.append(local_plot)

    for definition in TIMESERIES_PLOTS:
        plot = _build_timeseries_plot(definition, ulog, start_s, end_s, overlays, max_points)
        if plot is not None:
            plots.append(plot)

    return {
        "plots": plots,
        "time_range_s": [_round_float(start_s), _round_float(end_s)],
        "warnings": [],
    }


def _build_timeseries_plot(
    definition: dict[str, Any],
    ulog: Any,
    start_s: float,
    end_s: float,
    overlays: list[dict[str, Any]],
    max_points: int,
) -> dict[str, Any] | None:
    series = []
    warnings = []
    for index, series_def in enumerate(definition["series"]):
        resolved = _resolve_first_available(ulog, series_def["signals"], start_s, end_s)
        if resolved is None:
            continue

        time_s, values = _downsample_pair(resolved["time_s"], resolved["values"], max_points)
        series.append(
            {
                "key": _series_key(definition["id"], series_def["label"]),
                "label": series_def["label"],
                "signal": resolved["signal"],
                "unit": resolved.get("unit"),
                "axis_label": resolved.get("axis_label"),
                "color": COLORS8[index % len(COLORS8)],
                "time_s": time_s,
                "values": values,
            }
        )

    if not series:
        return None

    return {
        "id": definition["id"],
        "title": definition["title"],
        "kind": "timeseries",
        "time_range_s": [_round_float(start_s), _round_float(end_s)],
        "y_range": definition.get("y_range"),
        "series": series,
        "overlays": overlays,
        "warnings": warnings,
    }


def _build_local_position_plot(
    ulog: Any,
    start_s: float,
    end_s: float,
    max_points: int,
) -> dict[str, Any] | None:
    traces = []

    for trace_def in LOCAL_POSITION_PLOT["traces"]:
        if trace_def.get("source") == "vehicle_gps_position":
            trace = _build_projected_gps_trace(ulog, trace_def, max_points)
            if trace is not None:
                traces.append(trace)
            continue

        if trace_def.get("source") == "position_setpoint_triplet":
            trace = _build_position_setpoints_trace(ulog, trace_def, max_points)
            if trace is not None:
                traces.append(trace)
            continue

        x_signal = resolve_signal(ulog, trace_def["x"], start_s, end_s)
        y_signal = resolve_signal(ulog, trace_def["y"], start_s, end_s)
        if "warning" in x_signal or "warning" in y_signal:
            continue

        aligned = _align_xy_with_time(x_signal, y_signal)
        if not aligned["time_s"]:
            continue

        time_s, x_values, y_values = _downsample_xy(
            aligned["time_s"],
            aligned["x"],
            aligned["y"],
            max_points,
        )

        z_payload = None
        if trace_def.get("z"):
            z_signal = resolve_signal(ulog, trace_def["z"], start_s, end_s)
            if "warning" not in z_signal:
                z_time, z_values = _downsample_pair(
                    z_signal["time_s"],
                    z_signal["values"],
                    max_points,
                )
                z_payload = {"time_s": z_time, "values": z_values, "unit": z_signal.get("unit")}

        traces.append(
            {
                "key": trace_def["key"],
                "label": trace_def["label"],
                "color": trace_def["color"],
                "x_signal": trace_def["x"],
                "y_signal": trace_def["y"],
                "z_signal": trace_def.get("z"),
                "time_s": time_s,
                "x": y_values,
                "y": x_values,
                "z": z_payload,
            }
        )

    if not traces:
        return None

    return {
        "id": LOCAL_POSITION_PLOT["id"],
        "title": LOCAL_POSITION_PLOT["title"],
        "kind": "local_position",
        "time_range_s": [_round_float(start_s), _round_float(end_s)],
        "scale_from": "position",
        "equal_aspect": True,
        "min_range": 5.0,
        "zoom_out_factor": 1.3,
        "traces": traces,
        "warnings": [],
    }


def _build_projected_gps_trace(
    ulog: Any,
    trace_def: dict[str, Any],
    max_points: int,
) -> dict[str, Any] | None:
    gps = _find_dataset(ulog, "vehicle_gps_position")
    if gps is None:
        return None

    data = getattr(gps, "data", {}) or {}
    timestamps = data.get("timestamp")
    fix_type = data.get("fix_type")
    lat, lon, _ = _lat_lon_alt_deg(ulog, gps)
    if timestamps is None or fix_type is None or lat is None or lon is None:
        return None

    anchor = _projection_anchor(ulog, lat, lon)
    time_s = []
    lat_rad = []
    lon_rad = []
    for timestamp, fix, lat_value, lon_value in zip(timestamps, fix_type, lat, lon):
        if float(_json_safe_value(fix)) <= 2:
            continue
        time_s.append(_timestamp_to_seconds(timestamp))
        lat_rad.append(math.radians(float(_json_safe_value(lat_value))))
        lon_rad.append(math.radians(float(_json_safe_value(lon_value))))

    if not time_s:
        return None

    x_values, y_values = _map_projection(lat_rad, lon_rad, anchor[0], anchor[1])
    time_s, x_values, y_values = _downsample_xy(time_s, x_values, y_values, max_points)
    return {
        "key": trace_def["key"],
        "label": trace_def["label"],
        "color": trace_def["color"],
        "time_s": time_s,
        "x": y_values,
        "y": x_values,
        "marker_only": False,
    }


def _build_position_setpoints_trace(
    ulog: Any,
    trace_def: dict[str, Any],
    max_points: int,
) -> dict[str, Any] | None:
    dataset = _find_dataset(ulog, "position_setpoint_triplet")
    gps = _find_dataset(ulog, "vehicle_gps_position")
    if dataset is None or gps is None:
        return None

    data = getattr(dataset, "data", {}) or {}
    timestamps = data.get("timestamp")
    lat = data.get("current.lat")
    lon = data.get("current.lon")
    alt = data.get("current.alt")
    gps_lat, gps_lon, _ = _lat_lon_alt_deg(ulog, gps)
    if timestamps is None or lat is None or lon is None or gps_lat is None or gps_lon is None:
        return None

    anchor = _projection_anchor(ulog, gps_lat, gps_lon)
    time_s = []
    lat_rad = []
    lon_rad = []
    alt_time_s = []
    z_values = []
    for index, (timestamp, lat_value, lon_value) in enumerate(zip(timestamps, lat, lon)):
        safe_lat = _json_safe_value(lat_value)
        safe_lon = _json_safe_value(lon_value)
        if safe_lat is None or safe_lon is None:
            continue
        try:
            if not math.isfinite(float(safe_lat)) or not math.isfinite(float(safe_lon)):
                continue
        except (TypeError, ValueError):
            continue

        time_s.append(_timestamp_to_seconds(timestamp))
        lat_rad.append(math.radians(float(safe_lat)))
        lon_rad.append(math.radians(float(safe_lon)))
        if alt is not None and index < len(alt):
            alt_time_s.append(_timestamp_to_seconds(timestamp))
            z_values.append(float(_json_safe_value(alt[index])))

    if not time_s:
        return None

    x_values, y_values = _map_projection(lat_rad, lon_rad, anchor[0], anchor[1])
    time_s, x_values, y_values = _downsample_xy(time_s, x_values, y_values, max_points)
    z_payload = None
    if len(z_values) == len(alt_time_s):
        z_time, z_values = _downsample_pair(alt_time_s, z_values, max_points)
        z_payload = {"time_s": z_time, "values": z_values, "unit": "m"}

    return {
        "key": trace_def["key"],
        "label": trace_def["label"],
        "color": trace_def["color"],
        "time_s": time_s,
        "x": y_values,
        "y": x_values,
        "z": z_payload,
        "marker_only": True,
    }


def _resolve_first_available(
    ulog: Any,
    signals: list[str],
    start_s: float,
    end_s: float,
) -> dict[str, Any] | None:
    for signal in signals:
        resolved = resolve_signal(ulog, signal, start_s, end_s)
        if "warning" not in resolved:
            return resolved

    return None


def _align_xy_with_time(x_signal: dict[str, Any], y_signal: dict[str, Any]) -> dict[str, list[float]]:
    aligned_time = []
    aligned_x = []
    aligned_y = []
    y_times = y_signal["time_s"]
    y_values = y_signal["values"]

    for time_s, x_value in zip(x_signal["time_s"], x_signal["values"]):
        if not y_times or time_s < y_times[0] or time_s > y_times[-1]:
            continue

        y_value = _interpolated_value(y_times, y_values, time_s)
        if y_value is None:
            continue

        aligned_time.append(time_s)
        aligned_x.append(float(x_value))
        aligned_y.append(float(y_value))

    return {
        "time_s": aligned_time,
        "x": aligned_x,
        "y": aligned_y,
    }


def _interpolated_value(times: list[float], values: list[float], target_time: float) -> float | None:
    index = bisect_left(times, target_time)
    if index == 0:
        return float(values[0])

    if index >= len(times):
        return float(values[-1])

    t0 = times[index - 1]
    t1 = times[index]
    v0 = float(values[index - 1])
    v1 = float(values[index])
    if t1 == t0:
        return v0

    ratio = (target_time - t0) / (t1 - t0)
    return v0 + ratio * (v1 - v0)


def _downsample_pair(
    time_s: list[float],
    values: list[float],
    max_points: int,
) -> tuple[list[float], list[float]]:
    if len(time_s) <= max_points:
        return _round_list(time_s), _round_list(values)

    step = max(1, math.ceil(len(time_s) / max_points))
    indices = list(range(0, len(time_s), step))
    if indices[-1] != len(time_s) - 1:
        indices.append(len(time_s) - 1)

    return (
        [_round_float(time_s[index]) for index in indices],
        [_round_float(values[index]) for index in indices],
    )


def _downsample_xy(
    time_s: list[float],
    x_values: list[float],
    y_values: list[float],
    max_points: int,
) -> tuple[list[float], list[float], list[float]]:
    if len(time_s) <= max_points:
        return _round_list(time_s), _round_list(x_values), _round_list(y_values)

    step = max(1, math.ceil(len(time_s) / max_points))
    indices = list(range(0, len(time_s), step))
    if indices[-1] != len(time_s) - 1:
        indices.append(len(time_s) - 1)

    return (
        [_round_float(time_s[index]) for index in indices],
        [_round_float(x_values[index]) for index in indices],
        [_round_float(y_values[index]) for index in indices],
    )


def _build_overlays(ulog: Any, start_s: float, end_s: float) -> list[dict[str, Any]]:
    overlays = _flight_review_style_background_overlays(ulog, start_s, end_s)
    overlays.extend(_dropout_overlays(ulog, start_s, end_s))
    overlays.extend(_changed_parameter_overlays(ulog, start_s, end_s))
    return overlays


def _flight_review_style_background_overlays(
    ulog: Any,
    start_s: float,
    end_s: float,
) -> list[dict[str, Any]]:
    overlays = []
    vehicle_status = _find_dataset(ulog, "vehicle_status")
    if vehicle_status is not None:
        overlays.extend(
            _state_background_overlays(
                vehicle_status,
                "nav_state",
                start_s,
                end_s,
                FLIGHT_MODE_STYLES,
                kind="mode_background",
                source="vehicle_status.nav_state",
                alpha=0.09,
            )
        )

    vtol_status = _find_dataset(ulog, "vtol_vehicle_status")
    if vtol_status is not None:
        overlays.extend(
            _state_background_overlays(
                vtol_status,
                "vehicle_vtol_state",
                start_s,
                end_s,
                VTOL_MODE_STYLES,
                kind="vtol_background",
                source="vtol_vehicle_status.vehicle_vtol_state",
                alpha=0.09,
                band="bottom",
            )
        )

    return overlays


def _state_background_overlays(
    dataset: Any,
    field_name: str,
    start_s: float,
    end_s: float,
    styles: dict[int, tuple[str, str]],
    *,
    kind: str,
    source: str,
    alpha: float,
    band: str | None = None,
) -> list[dict[str, Any]]:
    data = getattr(dataset, "data", {}) or {}
    timestamps = data.get("timestamp")
    values = data.get(field_name)
    if timestamps is None or values is None:
        return []

    intervals = _state_intervals(timestamps, values, start_s, end_s)
    overlays = []
    for interval_start, interval_end, value in intervals:
        try:
            label, color = styles[int(value)]
        except (KeyError, TypeError, ValueError):
            continue

        overlay = {
            "kind": kind,
            "source": source,
            "start_s": interval_start,
            "end_s": interval_end,
            "label": label,
            "color": color,
            "alpha": alpha,
        }
        if band:
            overlay["band"] = band
        overlays.append(overlay)

    return overlays


def _state_intervals(
    timestamps: Any,
    values: Any,
    start_s: float,
    end_s: float,
) -> list[tuple[float, float, Any]]:
    current_value = None
    current_start = start_s
    intervals = []

    for timestamp, value in zip(timestamps, values):
        time_s = _timestamp_to_seconds(timestamp)
        if time_s < start_s:
            current_value = _json_safe_value(value)
            continue
        if time_s > end_s:
            break

        value = _json_safe_value(value)
        if current_value is None:
            current_value = value
            current_start = max(start_s, time_s)
            continue

        if value != current_value:
            if time_s > current_start:
                intervals.append((current_start, time_s, current_value))
            current_value = value
            current_start = time_s

    if current_value is not None and end_s > current_start:
        intervals.append((current_start, end_s, current_value))

    return intervals


def _dropout_overlays(ulog: Any, start_s: float, end_s: float) -> list[dict[str, Any]]:
    overlays = []
    for dropout in getattr(ulog, "dropouts", []) or []:
        timestamp = getattr(dropout, "timestamp", None)
        duration_ms = getattr(dropout, "duration", None)
        if timestamp is None or duration_ms is None:
            continue

        dropout_start = _timestamp_to_seconds(timestamp)
        dropout_end = dropout_start + float(duration_ms) / 1000.0
        if dropout_end < start_s or dropout_start > end_s:
            continue

        overlays.append(
            {
                "kind": "dropout",
                "start_s": _round_float(max(start_s, dropout_start)),
                "end_s": _round_float(min(end_s, dropout_end)),
                "label": "Dropout",
                "color": "red",
                "alpha": 0.15,
            }
        )

    return overlays


def _changed_parameter_overlays(ulog: Any, start_s: float, end_s: float) -> list[dict[str, Any]]:
    overlays = []
    for timestamp, name, value in (getattr(ulog, "changed_parameters", []) or [])[:80]:
        time_s = _timestamp_to_seconds(timestamp)
        if time_s < start_s or time_s > end_s:
            continue

        overlays.append(
            {
                "kind": "parameter_change",
                "start_s": _round_float(time_s),
                "end_s": _round_float(time_s),
                "label": f"{name}={_json_safe_value(value)}",
                "color": "#756052",
                "alpha": 0.5,
            }
        )

    return overlays


def _log_time_range(ulog: Any) -> tuple[float, float]:
    start_timestamp = getattr(ulog, "start_timestamp", None)
    last_timestamp = getattr(ulog, "last_timestamp", None)
    if start_timestamp is not None and last_timestamp is not None:
        start_s = _timestamp_to_seconds(start_timestamp)
        end_s = _timestamp_to_seconds(last_timestamp)
        offset_s = (end_s - start_s) * 0.05
        return start_s - offset_s, end_s + offset_s

    starts = []
    ends = []
    for dataset in getattr(ulog, "data_list", []) or []:
        timestamps = (getattr(dataset, "data", {}) or {}).get("timestamp")
        if timestamps is None:
            continue
        try:
            if len(timestamps) == 0:
                continue
            starts.append(_timestamp_to_seconds(timestamps[0]))
            ends.append(_timestamp_to_seconds(timestamps[-1]))
        except (TypeError, IndexError, ValueError):
            continue

    if not starts or not ends:
        return 0.0, 0.0

    start_s = min(starts)
    end_s = max(ends)
    offset_s = (end_s - start_s) * 0.05
    return start_s - offset_s, end_s + offset_s


def _round_list(values: list[float]) -> list[float]:
    return [_round_float(value) for value in values]


def _round_float(value: Any) -> float:
    return round(float(_json_safe_value(value)), 6)


def _series_key(plot_id: str, label: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in label)
    return f"{plot_id}_{normalized.strip('_')}"


def _find_dataset(ulog: Any, topic_name: str) -> Any:
    try:
        return getattr(ulog, "get_dataset")(topic_name)
    except Exception:
        pass

    for dataset in getattr(ulog, "data_list", []) or []:
        if getattr(dataset, "name", None) == topic_name:
            return dataset

    return None


def _lat_lon_alt_deg(ulog: Any, dataset: Any) -> tuple[Any, Any, Any]:
    data = getattr(dataset, "data", {}) or {}
    info = getattr(ulog, "msg_info_dict", {}) or {}
    if int(info.get("ver_data_format", 0) or 0) >= 2:
        return data.get("latitude_deg"), data.get("longitude_deg"), data.get("altitude_msl_m")
    if "lat" in data and "lon" in data and "alt" in data:
        lat = [float(_json_safe_value(value)) / 1e7 for value in data["lat"]]
        lon = [float(_json_safe_value(value)) / 1e7 for value in data["lon"]]
        alt = [float(_json_safe_value(value)) / 1e3 for value in data["alt"]]
        return lat, lon, alt
    return None, None, None


def _projection_anchor(ulog: Any, lat: Any, lon: Any) -> tuple[float, float]:
    anchor_lat = math.radians(float(_json_safe_value(lat[0])))
    anchor_lon = math.radians(float(_json_safe_value(lon[0])))
    local_position = _find_dataset(ulog, "vehicle_local_position")
    if local_position is None:
        return anchor_lat, anchor_lon

    data = getattr(local_position, "data", {}) or {}
    ref_timestamps = data.get("ref_timestamp")
    ref_lat = data.get("ref_lat")
    ref_lon = data.get("ref_lon")
    if ref_timestamps is None or ref_lat is None or ref_lon is None:
        return anchor_lat, anchor_lon

    for index, timestamp in enumerate(ref_timestamps):
        try:
            if float(_json_safe_value(timestamp)) != 0:
                return (
                    math.radians(float(_json_safe_value(ref_lat[index]))),
                    math.radians(float(_json_safe_value(ref_lon[index]))),
                )
        except (TypeError, ValueError, IndexError):
            continue

    return anchor_lat, anchor_lon


def _map_projection(
    lat: list[float],
    lon: list[float],
    anchor_lat: float,
    anchor_lon: float,
) -> tuple[list[float], list[float]]:
    sin_anchor_lat = math.sin(anchor_lat)
    cos_anchor_lat = math.cos(anchor_lat)
    x_values = []
    y_values = []
    earth_radius_m = 6_371_000.0

    for lat_value, lon_value in zip(lat, lon):
        sin_lat = math.sin(lat_value)
        cos_lat = math.cos(lat_value)
        cos_d_lon = math.cos(lon_value - anchor_lon)
        arg = sin_anchor_lat * sin_lat + cos_anchor_lat * cos_lat * cos_d_lon
        arg = min(1.0, max(-1.0, arg))
        c = math.acos(arg)
        if abs(c) < 1e-12:
            k = 1.0
        else:
            k = c / math.sin(c)

        x = k * (cos_anchor_lat * sin_lat - sin_anchor_lat * cos_lat * cos_d_lon) * earth_radius_m
        y = k * cos_lat * math.sin(lon_value - anchor_lon) * earth_radius_m
        x_values.append(x)
        y_values.append(y)

    return x_values, y_values
