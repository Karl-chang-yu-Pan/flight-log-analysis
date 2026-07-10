from __future__ import annotations

import math
from bisect import bisect_left
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyulog import ULog

from flight_log_agent.ulog.plots import (
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
    4: ("Fixed-Wing", "#eecc00"),
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


SPECTROGRAM_PLOTS = [
    {
        "id": "angular_velocity_spectrogram",
        "title": "Angular Velocity Spectrogram",
        "candidates": [
            {
                "topic": "vehicle_angular_velocity",
                "fields": ["xyz[0]", "xyz[1]", "xyz[2]"],
                "labels": ["Rollspeed", "Pitchspeed", "Yawspeed"],
            },
        ],
    },
    {
        "id": "angular_acceleration_spectrogram",
        "title": "Angular Acceleration Spectrogram",
        "candidates": [
            {
                "topic": "vehicle_angular_acceleration",
                "fields": ["xyz[0]", "xyz[1]", "xyz[2]"],
                "labels": ["Roll accel", "Pitch accel", "Yaw accel"],
            },
        ],
    },
    {
        "id": "actuator_controls_spectrogram",
        "title": "Actuator Controls Spectrogram",
        "candidates": [
            {
                "topic": "vehicle_torque_setpoint",
                "fields": ["xyz[0]", "xyz[1]", "xyz[2]"],
                "labels": ["Roll", "Pitch", "Yaw"],
            },
            {
                "topic": "actuator_controls_0",
                "fields": ["control[0]", "control[1]", "control[2]"],
                "labels": ["Roll", "Pitch", "Yaw"],
            },
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

    plots.extend(_build_velocity_frame_plots(ulog, start_s, end_s, max_points))

    for definition in TIMESERIES_PLOTS:
        plot = _build_timeseries_plot(definition, ulog, start_s, end_s, overlays, max_points)
        if plot is not None:
            plots.append(plot)

    plots.extend(_build_actuator_control_plots(ulog, start_s, end_s, overlays, max_points))

    for definition in SPECTROGRAM_PLOTS:
        plot = _build_spectrogram_plot(definition, ulog, start_s, end_s, max_points)
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


def _build_velocity_frame_plots(
    ulog: Any,
    start_s: float,
    end_s: float,
    max_points: int,
) -> list[dict[str, Any]]:
    style_profile = _flight_style_profile(ulog)
    styles = style_profile["styles"]
    plots = []

    if "fixed_wing" in styles:
        plot = _build_fixed_wing_velocity_angle_plot(ulog, start_s, end_s, max_points, style_profile)
        if plot is not None:
            plots.append(plot)

    if "multirotor" in styles:
        plot = _build_multirotor_heading_velocity_plot(ulog, start_s, end_s, max_points, style_profile)
        if plot is not None:
            plots.append(plot)

    return plots


def _build_fixed_wing_velocity_angle_plot(
    ulog: Any,
    start_s: float,
    end_s: float,
    max_points: int,
    style_profile: dict[str, Any],
) -> dict[str, Any] | None:
    velocity = _raw_field_series(ulog, "vehicle_local_position", ["vx", "vy", "vz"], start_s, end_s)
    attitude = _raw_field_series(ulog, "vehicle_attitude", ["roll", "pitch", "yaw"], start_s, end_s)
    if velocity is None or attitude is None:
        return None

    time_s = []
    aoa = []
    sideslip = []
    total_speed = []
    for index, sample_time in enumerate(velocity["time_s"]):
        if not _style_valid_at_time(style_profile, "fixed_wing", sample_time):
            time_s.append(sample_time)
            aoa.append(0.0)
            sideslip.append(0.0)
            total_speed.append(0.0)
            continue

        roll = _interpolated_value(attitude["time_s"], attitude["roll"], sample_time)
        pitch = _interpolated_value(attitude["time_s"], attitude["pitch"], sample_time)
        yaw = _interpolated_value(attitude["time_s"], attitude["yaw"], sample_time)
        if roll is None or pitch is None or yaw is None:
            continue

        body = _ned_velocity_to_body(
            velocity["vx"][index],
            velocity["vy"][index],
            velocity["vz"][index],
            roll,
            pitch,
            yaw,
        )
        forward, right, down = body
        if not all(math.isfinite(value) for value in body):
            continue

        time_s.append(sample_time)
        aoa.append(math.degrees(math.atan2(down, forward)))
        sideslip.append(math.degrees(math.atan2(right, math.hypot(forward, down))))
        total_speed.append(math.sqrt(forward * forward + right * right + down * down))

    if not time_s:
        return None

    time_s, aoa, sideslip, total_speed = _downsample_multi(time_s, [aoa, sideslip, total_speed], max_points)
    return {
        "id": "fixed_wing_body_velocity_angles",
        "title": "Fixed-Wing Body Velocity Angles",
        "kind": "timeseries",
        "time_range_s": [_round_float(start_s), _round_float(end_s)],
        "series": [
            {
                "key": "fixed_wing_body_velocity_angles_aoa",
                "label": "AoA",
                "signal": "derived.vehicle_body_velocity_aoa",
                "unit": "deg",
                "axis_label": "[deg]",
                "color": COLORS2[0],
                "time_s": time_s,
                "values": aoa,
            },
            {
                "key": "fixed_wing_body_velocity_angles_sideslip",
                "label": "Sideslip",
                "signal": "derived.vehicle_body_velocity_sideslip",
                "unit": "deg",
                "axis_label": "[deg]",
                "color": COLORS2[1],
                "time_s": time_s,
                "values": sideslip,
            },
            {
                "key": "fixed_wing_body_velocity_angles_total_speed",
                "label": "Total Speed",
                "signal": "derived.vehicle_body_velocity_total_speed",
                "unit": "m/s",
                "axis_label": "[m/s]",
                "color": COLORS3[2],
                "time_s": time_s,
                "values": total_speed,
            },
        ],
        "overlays": [],
        "warnings": [],
    }


def _build_multirotor_heading_velocity_plot(
    ulog: Any,
    start_s: float,
    end_s: float,
    max_points: int,
    style_profile: dict[str, Any],
) -> dict[str, Any] | None:
    velocity = _raw_field_series(ulog, "vehicle_local_position", ["vx", "vy"], start_s, end_s)
    attitude = _raw_field_series(ulog, "vehicle_attitude", ["yaw"], start_s, end_s)
    if velocity is None or attitude is None:
        return None

    time_s = []
    forward_values = []
    right_values = []
    for index, sample_time in enumerate(velocity["time_s"]):
        if not _style_valid_at_time(style_profile, "multirotor", sample_time):
            time_s.append(sample_time)
            forward_values.append(0.0)
            right_values.append(0.0)
            continue

        yaw = _interpolated_value(attitude["time_s"], attitude["yaw"], sample_time)
        if yaw is None:
            continue

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        north = velocity["vx"][index]
        east = velocity["vy"][index]
        forward = cos_yaw * north + sin_yaw * east
        right = -sin_yaw * north + cos_yaw * east
        if not math.isfinite(forward) or not math.isfinite(right):
            continue

        time_s.append(sample_time)
        forward_values.append(forward)
        right_values.append(right)

    if not time_s:
        return None

    time_s, forward_values, right_values = _downsample_multi(time_s, [forward_values, right_values], max_points)
    return {
        "id": "multirotor_heading_velocity",
        "title": "Multirotor Heading-Frame Velocity",
        "kind": "timeseries",
        "time_range_s": [_round_float(start_s), _round_float(end_s)],
        "series": [
            {
                "key": "multirotor_heading_velocity_forward",
                "label": "Forward",
                "signal": "derived.heading_velocity_forward",
                "unit": "m/s",
                "axis_label": "[m/s]",
                "color": COLORS2[0],
                "time_s": time_s,
                "values": forward_values,
            },
            {
                "key": "multirotor_heading_velocity_right",
                "label": "Right",
                "signal": "derived.heading_velocity_right",
                "unit": "m/s",
                "axis_label": "[m/s]",
                "color": COLORS2[1],
                "time_s": time_s,
                "values": right_values,
            },
        ],
        "overlays": [],
        "warnings": [],
    }


def _build_actuator_control_plots(
    ulog: Any,
    start_s: float,
    end_s: float,
    overlays: list[dict[str, Any]],
    max_points: int,
) -> list[dict[str, Any]]:
    if _has_dataset(ulog, "actuator_motors") or _has_dataset(ulog, "actuator_servos"):
        candidates = [
            _build_dynamic_actuator_control_plot(ulog, 0, "actuator_controls", "Actuator Controls", start_s, end_s, overlays, max_points),
            _build_dynamic_actuator_control_plot(
                ulog,
                1,
                "actuator_controls_1",
                "Actuator Controls 1 (VTOL in Fixed-Wing mode)",
                start_s,
                end_s,
                overlays,
                max_points,
            ),
        ]
    else:
        candidates = [
            _build_legacy_actuator_control_plot(ulog, 0, "actuator_controls", "Actuator Controls", "Thrust (up)", start_s, end_s, overlays, max_points),
            _build_legacy_actuator_control_plot(
                ulog,
                1,
                "actuator_controls_1",
                "Actuator Controls 1 (VTOL in Fixed-Wing mode)",
                "Thrust (forward)",
                start_s,
                end_s,
                overlays,
                max_points,
            ),
        ]

    return [plot for plot in candidates if plot is not None]


def _build_legacy_actuator_control_plot(
    ulog: Any,
    instance: int,
    plot_id: str,
    title: str,
    thrust_label: str,
    start_s: float,
    end_s: float,
    overlays: list[dict[str, Any]],
    max_points: int,
) -> dict[str, Any] | None:
    dataset = _find_dataset(ulog, f"actuator_controls_{instance}")
    series_defs = [
        ("control[0]", "Roll", COLORS8[0], None),
        ("control[1]", "Pitch", COLORS8[1], None),
        ("control[2]", "Yaw", COLORS8[2], None),
        ("control[3]", thrust_label, COLORS8[3], None),
    ]
    return _build_actuator_control_plot_from_dataset(
        dataset,
        plot_id,
        title,
        series_defs,
        start_s,
        end_s,
        overlays,
        max_points,
    )


def _build_dynamic_actuator_control_plot(
    ulog: Any,
    instance: int,
    plot_id: str,
    title: str,
    start_s: float,
    end_s: float,
    overlays: list[dict[str, Any]],
    max_points: int,
) -> dict[str, Any] | None:
    torque_dataset = _find_dataset(ulog, "vehicle_torque_setpoint", instance)
    thrust_dataset = _find_dataset(ulog, "vehicle_thrust_setpoint", instance)
    if thrust_dataset is None and instance != 0:
        thrust_dataset = _find_dataset(ulog, "vehicle_thrust_setpoint", 0)

    series = []
    for field_name, label, color in (
        ("xyz[0]", "Roll", COLORS8[0]),
        ("xyz[1]", "Pitch", COLORS8[1]),
        ("xyz[2]", "Yaw", COLORS8[2]),
    ):
        item = _dataset_field_series(
            torque_dataset,
            field_name,
            f"{plot_id}_{_series_key_suffix(label)}",
            label,
            color,
            start_s,
            end_s,
            max_points,
        )
        if item is not None:
            series.append(item)

    thrust_defs = [("xyz[0]", "Thrust (forward)", COLORS8[4], None)]
    if instance == 0:
        thrust_defs.insert(0, ("xyz[2]", "Thrust (up)", COLORS8[3], lambda value: -value))

    for field_name, label, color, transform in thrust_defs:
        item = _dataset_field_series(
            thrust_dataset,
            field_name,
            f"{plot_id}_{_series_key_suffix(label)}",
            label,
            color,
            start_s,
            end_s,
            max_points,
            transform=transform,
        )
        if item is not None:
            series.append(item)

    return _actuator_control_plot_payload(plot_id, title, series, overlays)


def _build_actuator_control_plot_from_dataset(
    dataset: Any,
    plot_id: str,
    title: str,
    series_defs: list[tuple[str, str, str, Any]],
    start_s: float,
    end_s: float,
    overlays: list[dict[str, Any]],
    max_points: int,
) -> dict[str, Any] | None:
    series = []
    for field_name, label, color, transform in series_defs:
        item = _dataset_field_series(
            dataset,
            field_name,
            f"{plot_id}_{_series_key_suffix(label)}",
            label,
            color,
            start_s,
            end_s,
            max_points,
            transform=transform,
        )
        if item is not None:
            series.append(item)
    return _actuator_control_plot_payload(plot_id, title, series, overlays)


def _actuator_control_plot_payload(
    plot_id: str,
    title: str,
    series: list[dict[str, Any]],
    overlays: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not series:
        return None
    return {
        "id": plot_id,
        "title": title,
        "kind": "timeseries",
        "time_range_s": _series_time_range(series),
        "y_range": [-1.0, 1.0],
        "series": series,
        "overlays": overlays,
        "warnings": [],
    }


def _dataset_field_series(
    dataset: Any,
    field_name: str,
    key: str,
    label: str,
    color: str,
    start_s: float,
    end_s: float,
    max_points: int,
    *,
    transform: Any = None,
) -> dict[str, Any] | None:
    if dataset is None:
        return None
    data = getattr(dataset, "data", {}) or {}
    timestamps = data.get("timestamp")
    values = data.get(field_name)
    if timestamps is None or values is None:
        return None

    time_s = []
    series_values = []
    for timestamp, value in zip(timestamps, values):
        try:
            sample_time = _timestamp_to_seconds(timestamp)
            sample_value = float(_json_safe_value(value))
        except (TypeError, ValueError):
            continue
        if sample_time < start_s or sample_time > end_s or not math.isfinite(sample_value):
            continue
        if transform is not None:
            sample_value = transform(sample_value)
        time_s.append(sample_time)
        series_values.append(sample_value)

    if not time_s:
        return None
    time_s, series_values = _downsample_pair(time_s, series_values, max_points)
    return {
        "key": key,
        "label": label,
        "signal": f"{getattr(dataset, 'name', '')}.{field_name}",
        "unit": None,
        "axis_label": None,
        "color": color,
        "time_s": time_s,
        "values": series_values,
    }


def _series_time_range(series: list[dict[str, Any]]) -> list[float]:
    times = [time for item in series for time in item.get("time_s", [])]
    if not times:
        return [0.0, 0.0]
    return [_round_float(min(times)), _round_float(max(times))]


def _series_key_suffix(label: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_")


def _build_spectrogram_plot(
    definition: dict[str, Any],
    ulog: Any,
    start_s: float,
    end_s: float,
    max_points: int,
) -> dict[str, Any] | None:
    candidate = _first_spectrogram_candidate(ulog, definition["candidates"])
    if candidate is None:
        return None

    dataset = candidate["dataset"]
    data = getattr(dataset, "data", {}) or {}
    timestamp_key = "timestamp_sample" if "timestamp_sample" in data else "timestamp"
    timestamps = data.get(timestamp_key)
    if timestamps is None:
        return None

    try:
        data_len = len(timestamps)
    except TypeError:
        return None
    if data_len < 256:
        return None

    try:
        first_timestamp = float(_json_safe_value(timestamps[0]))
        last_timestamp = float(_json_safe_value(timestamps[data_len - 1]))
    except (TypeError, ValueError, IndexError):
        return None

    delta_t = ((last_timestamp - first_timestamp) * 1.0e-6) / data_len
    if delta_t <= 0:
        return None

    sampling_frequency = 1.0 / delta_t
    if sampling_frequency < 100 or sampling_frequency == float("inf"):
        return None

    arrays = []
    for field in candidate["fields"]:
        values = data.get(field)
        if values is None:
            return None
        try:
            array = np.asarray([float(_json_safe_value(value)) for value in values[:data_len]], dtype=float)
        except (TypeError, ValueError):
            return None
        if len(array) != data_len or not np.isfinite(array).any():
            return None
        arrays.append(np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0))

    frequency, times, values_db = _spectrogram_sum(arrays, first_timestamp, delta_t)
    if len(times) == 0:
        return None

    if len(times) > max_points:
        step = max(1, math.ceil(len(times) / max_points))
        times = times[::step]
        values_db = values_db[:, ::step]

    finite_values = values_db[np.isfinite(values_db)]
    if finite_values.size == 0:
        return None

    return {
        "id": definition["id"],
        "title": definition["title"],
        "kind": "spectrogram",
        "time_range_s": [_round_float(start_s), _round_float(end_s)],
        "frequency_range_hz": [_round_float(float(frequency[0])), _round_float(float(frequency[-1]))],
        "time_s": _round_list([float(value) for value in times]),
        "frequencies_hz": _round_list([float(value) for value in frequency]),
        "values_db": [
            _round_list([float(value) for value in row])
            for row in values_db
        ],
        "value_range_db": [_round_float(float(np.min(finite_values))), _round_float(float(np.max(finite_values)))],
        "sampling_frequency_hz": _round_float(sampling_frequency),
        "source": candidate["topic"],
        "fields": candidate["fields"],
        "labels": candidate["labels"],
        "warnings": [],
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


def _raw_field_series(
    ulog: Any,
    topic_name: str,
    field_names: list[str],
    start_s: float,
    end_s: float,
    topic_instance: int = 0,
) -> dict[str, list[float]] | None:
    dataset = _find_dataset(ulog, topic_name, topic_instance)
    if dataset is None:
        return None

    data = getattr(dataset, "data", {}) or {}
    timestamps = data.get("timestamp")
    if timestamps is None:
        return None

    fields = {field_name: data.get(field_name) for field_name in field_names}
    if any(values is None for values in fields.values()):
        return None

    result: dict[str, list[float]] = {"time_s": []}
    for field_name in field_names:
        result[field_name] = []

    for index, timestamp in enumerate(timestamps):
        try:
            time_s = _timestamp_to_seconds(timestamp)
        except (TypeError, ValueError):
            continue
        if time_s < start_s or time_s > end_s:
            continue

        row = {}
        valid = True
        for field_name, values in fields.items():
            try:
                value = float(_json_safe_value(values[index]))
            except (TypeError, ValueError, IndexError):
                valid = False
                break
            if not math.isfinite(value):
                valid = False
                break
            row[field_name] = value
        if not valid:
            continue

        result["time_s"].append(time_s)
        for field_name, value in row.items():
            result[field_name].append(value)

    return result if result["time_s"] else None


def _flight_style_profile(ulog: Any) -> dict[str, Any]:
    states = _flight_style_states(ulog)
    styles = _flight_styles_present(ulog)
    for _, state in states:
        if state == "transition":
            styles.update({"fixed_wing", "multirotor"})
        elif state in {"fixed_wing", "multirotor"}:
            styles.add(state)
    return {
        "styles": styles,
        "states": states,
    }


def _style_valid_at_time(profile: dict[str, Any], style: str, sample_time: float) -> bool:
    states = profile.get("states") or []
    if not states:
        return style in (profile.get("styles") or set())

    state = _state_at_time(states, sample_time)
    if state == "transition":
        return style in {"fixed_wing", "multirotor"}
    return state == style


def _state_at_time(states: list[tuple[float, str]], sample_time: float) -> str | None:
    if not states:
        return None
    current = states[0][1]
    for state_time, state in states:
        if sample_time < state_time:
            break
        current = state
    return current


def _flight_style_states(ulog: Any) -> list[tuple[float, str]]:
    vtol_status = _find_dataset(ulog, "vtol_vehicle_status")
    if vtol_status is not None:
        data = getattr(vtol_status, "data", {}) or {}
        states = _states_from_field(data, "vehicle_vtol_state", _style_from_vtol_state)
        if states:
            return states

    vehicle_status = _find_dataset(ulog, "vehicle_status")
    if vehicle_status is None:
        return []
    data = getattr(vehicle_status, "data", {}) or {}
    timestamps = data.get("timestamp")
    if timestamps is None:
        return []

    transition_values = data.get("in_transition_mode")
    vehicle_type_values = data.get("vehicle_type")
    rotary_values = data.get("is_rotary_wing")
    states = []
    for index, timestamp in enumerate(timestamps):
        try:
            sample_time = _timestamp_to_seconds(timestamp)
        except (TypeError, ValueError):
            continue

        transition = _indexed_truthy(transition_values, index)
        if transition:
            state = "transition"
        elif vehicle_type_values is not None:
            state = _style_from_vehicle_type(_indexed_value(vehicle_type_values, index))
        elif rotary_values is not None:
            state = "multirotor" if _indexed_truthy(rotary_values, index) else "fixed_wing"
        else:
            state = None
        if state is not None and (not states or states[-1][1] != state):
            states.append((sample_time, state))
    return states


def _states_from_field(data: dict[str, Any], field_name: str, mapper: Any) -> list[tuple[float, str]]:
    timestamps = data.get("timestamp")
    values = data.get(field_name)
    if timestamps is None or values is None:
        return []
    states = []
    for timestamp, value in zip(timestamps, values):
        state = mapper(value)
        if state is None:
            continue
        try:
            sample_time = _timestamp_to_seconds(timestamp)
        except (TypeError, ValueError):
            continue
        if not states or states[-1][1] != state:
            states.append((sample_time, state))
    return states


def _style_from_vtol_state(value: Any) -> str | None:
    parsed = _safe_int(value)
    if parsed == 1:
        return "transition"
    if parsed in {2, 4}:
        return "fixed_wing"
    if parsed == 3:
        return "multirotor"
    return None


def _style_from_vehicle_type(value: Any) -> str | None:
    parsed = _safe_int(value)
    if parsed == 1:
        return "multirotor"
    if parsed == 2:
        return "fixed_wing"
    return None


def _indexed_truthy(values: Any, index: int) -> bool:
    value = _indexed_value(values, index)
    if value is None:
        return False
    safe_value = _json_safe_value(value)
    if isinstance(safe_value, bool):
        return safe_value
    try:
        return float(safe_value) != 0.0
    except (TypeError, ValueError):
        return False


def _indexed_value(values: Any, index: int) -> Any:
    if values is None:
        return None
    try:
        return values[index]
    except (TypeError, IndexError):
        return None


def _flight_styles_present(ulog: Any) -> set[str]:
    styles: set[str] = set()

    vehicle_type = _find_dataset(ulog, "vehicle_type")
    if vehicle_type is not None:
        data = getattr(vehicle_type, "data", {}) or {}
        if _field_has_truthy_value(data.get("fixed_wing")):
            styles.add("fixed_wing")
        if _field_has_truthy_value(data.get("rotary_wing")):
            styles.add("multirotor")

    vehicle_status = _find_dataset(ulog, "vehicle_status")
    if vehicle_status is not None:
        data = getattr(vehicle_status, "data", {}) or {}
        for value in _iter_field_values(data, "vehicle_type"):
            style = _style_from_vehicle_type(value)
            if style is not None:
                styles.add(style)

    vtol_status = _find_dataset(ulog, "vtol_vehicle_status")
    if vtol_status is not None:
        data = getattr(vtol_status, "data", {}) or {}
        for value in _iter_field_values(data, "vehicle_vtol_state"):
            style = _style_from_vtol_state(value)
            if style == "transition":
                styles.update({"fixed_wing", "multirotor"})
            elif style is not None:
                styles.add(style)

    return styles


def _iter_field_values(data: Any, field_name: str) -> Iterable[Any]:
    if not isinstance(data, dict):
        return ()
    values = data.get(field_name)
    if values is None:
        return ()
    return values


def _field_has_truthy_value(values: Any) -> bool:
    if values is None:
        return False
    for value in values:
        safe_value = _json_safe_value(value)
        if isinstance(safe_value, bool) and safe_value:
            return True
        try:
            if float(safe_value) != 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _safe_int(value: Any) -> int | None:
    try:
        return int(_json_safe_value(value))
    except (TypeError, ValueError):
        return None


def _ned_velocity_to_body(
    north: float,
    east: float,
    down: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float]:
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    forward = cos_pitch * cos_yaw * north + cos_pitch * sin_yaw * east - sin_pitch * down
    right = (
        (sin_roll * sin_pitch * cos_yaw - cos_roll * sin_yaw) * north
        + (sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw) * east
        + sin_roll * cos_pitch * down
    )
    body_down = (
        (cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw) * north
        + (cos_roll * sin_pitch * sin_yaw - sin_roll * cos_yaw) * east
        + cos_roll * cos_pitch * down
    )
    return forward, right, body_down


def _first_spectrogram_candidate(ulog: Any, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in candidates:
        dataset = _find_dataset(ulog, candidate["topic"])
        if dataset is None:
            continue
        data = getattr(dataset, "data", {}) or {}
        if all(field in data for field in candidate["fields"]):
            return {**candidate, "dataset": dataset}
    return None


def _spectrogram_sum(
    arrays: list[np.ndarray],
    first_timestamp: float,
    delta_t: float,
    *,
    window_length: int = 256,
    noverlap: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = window_length - noverlap
    if step <= 0 or not arrays:
        return np.asarray([]), np.asarray([]), np.asarray([[]])

    data_len = min(len(array) for array in arrays)
    if data_len < window_length:
        return np.asarray([]), np.asarray([]), np.asarray([[]])

    starts = np.arange(0, data_len - window_length + 1, step)
    frequency = np.fft.rfftfreq(window_length, delta_t)
    window = np.hanning(window_length)
    window_power = float(np.sum(window * window)) or 1.0
    sampling_frequency = 1.0 / delta_t
    sum_psd = np.zeros((len(frequency), len(starts)))

    for array in arrays:
        trimmed = array[:data_len]
        for column, start_index in enumerate(starts):
            segment = np.asarray(trimmed[start_index:start_index + window_length], dtype=float)
            segment = segment - np.mean(segment)
            fft_values = np.fft.rfft(segment * window)
            psd = (np.abs(fft_values) ** 2) / (sampling_frequency * window_power)
            if len(psd) > 2:
                psd[1:-1] *= 2.0
            sum_psd[:, column] += psd

    sum_psd = np.maximum(sum_psd, np.finfo(float).tiny)
    values_db = 10.0 * np.log10(sum_psd)
    time_offsets = (starts + window_length / 2.0) * delta_t
    time_s = (first_timestamp * 1.0e-6) + time_offsets
    return frequency, time_s, values_db


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


def _downsample_multi(
    time_s: list[float],
    series_values: list[list[float]],
    max_points: int,
) -> tuple[list[float], ...]:
    if len(time_s) <= max_points:
        return (_round_list(time_s), *[_round_list(values) for values in series_values])

    step = max(1, math.ceil(len(time_s) / max_points))
    indices = list(range(0, len(time_s), step))
    if indices[-1] != len(time_s) - 1:
        indices.append(len(time_s) - 1)

    return (
        [_round_float(time_s[index]) for index in indices],
        *[
            [_round_float(values[index]) for index in indices]
            for values in series_values
        ],
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


def _has_dataset(ulog: Any, topic_name: str, topic_instance: int = 0) -> bool:
    return _find_dataset(ulog, topic_name, topic_instance) is not None


def _find_dataset(ulog: Any, topic_name: str, topic_instance: int = 0) -> Any:
    get_dataset = getattr(ulog, "get_dataset", None)
    if get_dataset is not None:
        if topic_instance != 0:
            try:
                dataset = get_dataset(topic_name, topic_instance)
                if _dataset_matches_instance(dataset, topic_instance):
                    return dataset
            except TypeError:
                pass
            except Exception:
                pass
        else:
            try:
                dataset = get_dataset(topic_name)
                if _dataset_matches_instance(dataset, topic_instance):
                    return dataset
            except Exception:
                pass

    for dataset in getattr(ulog, "data_list", []) or []:
        if getattr(dataset, "name", None) == topic_name and _dataset_matches_instance(dataset, topic_instance):
            return dataset

    return None


def _dataset_matches_instance(dataset: Any, topic_instance: int) -> bool:
    if dataset is None:
        return False
    return _safe_int(getattr(dataset, "multi_id", 0)) == topic_instance


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
