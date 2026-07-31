from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from pyulog.px4 import PX4ULog

from flight_log_agent.ulog.interactive_plot_data import (
    boolean_fields,
    dataset_instance,
    field_has_finite_values,
    field_has_nonzero_values,
    field_type,
    find_dataset,
    indexed_fields,
    iter_topic_datasets,
    series_from_arrays,
    series_from_field,
    timeseries_plot,
)
from flight_log_agent.ulog.interactive_plot_spectra import (
    expand_fifo_samples,
    positive_rfft_spectrum,
    summed_spectrogram,
)


FFT_MEAN_START_FREQUENCY_HZ = 40.0


@dataclass(frozen=True)
class PlotContext:
    ulog: Any
    start_s: float
    end_s: float
    overlays: list[dict[str, Any]]
    max_points: int
    colors: Sequence[str]

    @property
    def time_range_s(self) -> list[float]:
        return [self.start_s, self.end_s]


def build_visual_odometry_plots(context: PlotContext) -> list[dict[str, Any]]:
    dataset = find_dataset(context.ulog, "vehicle_visual_odometry")
    if dataset is None:
        return []

    plots = []
    position = _vector_plot(
        context,
        dataset,
        "visual_odometry_position",
        "Visual Odometry Position",
        (("x", "y", "z"), ("position[0]", "position[1]", "position[2]")),
        ("X", "Y", "Z"),
        unit="m",
        groundtruth_topic="vehicle_local_position_groundtruth",
        groundtruth_groups=(("x", "y", "z"), ("position[0]", "position[1]", "position[2]")),
        groundtruth_labels=("Groundtruth X", "Groundtruth Y", "Groundtruth Z"),
    )
    if position is not None:
        plots.append(position)

    velocity = _vector_plot(
        context,
        dataset,
        "visual_odometry_velocity",
        "Visual Odometry Velocity",
        (("vx", "vy", "vz"), ("velocity[0]", "velocity[1]", "velocity[2]")),
        ("X", "Y", "Z"),
        unit="m/s",
        groundtruth_topic="vehicle_local_position_groundtruth",
        groundtruth_groups=(("vx", "vy", "vz"), ("velocity[0]", "velocity[1]", "velocity[2]")),
        groundtruth_labels=("Groundtruth VX", "Groundtruth VY", "Groundtruth VZ"),
    )
    if velocity is not None:
        plots.append(velocity)

    attitude = _build_visual_odometry_attitude(context, dataset)
    if attitude is not None:
        plots.append(attitude)

    attitude_rate = _vector_plot(
        context,
        dataset,
        "visual_odometry_attitude_rate",
        "Visual Odometry Attitude Rate",
        (
            ("rollspeed", "pitchspeed", "yawspeed"),
            ("angular_velocity[0]", "angular_velocity[1]", "angular_velocity[2]"),
        ),
        ("Roll Rate", "Pitch Rate", "Yaw Rate"),
        unit="deg/s",
        transform=math.degrees,
        normalize=False,
        groundtruth_topic=(
            "vehicle_angular_velocity_groundtruth",
            "vehicle_attitude_groundtruth",
        ),
        groundtruth_groups=(
            ("xyz[0]", "xyz[1]", "xyz[2]"),
            ("rollspeed", "pitchspeed", "yawspeed"),
        ),
        groundtruth_labels=(
            "Roll Rate Groundtruth",
            "Pitch Rate Groundtruth",
            "Yaw Rate Groundtruth",
        ),
        groundtruth_transform=math.degrees,
        groundtruth_normalize=False,
    )
    if attitude_rate is not None:
        plots.append(attitude_rate)

    latency = _build_visual_odometry_latency(context, dataset)
    if latency is not None:
        plots.append(latency)
    return plots


def build_airspeed_plot(context: PlotContext) -> dict[str, Any] | None:
    validated = find_dataset(context.ulog, "airspeed_validated")
    legacy = find_dataset(context.ulog, "airspeed")
    if validated is None and legacy is None:
        return None

    series = []
    global_position = find_dataset(context.ulog, "vehicle_global_position")
    groundspeed = _derived_norm_series(
        context,
        global_position,
        ("vel_n", "vel_e"),
        key="airspeed_ground_speed_estimated",
        label="Ground Speed Estimated",
        signal="derived.vehicle_global_position.ground_speed",
        color=context.colors[0],
        unit="m/s",
    )
    if groundspeed is None:
        local_position = find_dataset(context.ulog, "vehicle_local_position")
        groundspeed = _derived_norm_series(
            context,
            local_position,
            ("vx", "vy"),
            key="airspeed_ground_speed_estimated",
            label="Ground Speed Estimated",
            signal="derived.vehicle_local_position.ground_speed",
            color=context.colors[0],
            unit="m/s",
        )
    if groundspeed is not None:
        series.append(groundspeed)

    selected_airspeed = _validated_airspeed_series(context, validated)
    if selected_airspeed is not None:
        selected_airspeed["color"] = context.colors[1 % len(context.colors)]
        series.append(selected_airspeed)
    elif legacy is not None:
        for field_name, label in (
            ("indicated_airspeed_m_s", "Indicated airspeed"),
            ("true_airspeed_m_s", "True airspeed"),
            ("true_airspeed", "True airspeed"),
        ):
            item = series_from_field(
                legacy,
                field_name,
                key=f"airspeed_{_slug(label)}",
                label=label,
                color=context.colors[1 % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
            )
            if item is not None:
                series.append(item)
                break

    gps = find_dataset(context.ulog, "vehicle_gps_position")
    gps_speed = series_from_field(
        gps,
        "vel_m_s",
        key="airspeed_gps_speed",
        label="GPS speed",
        color=context.colors[2 % len(context.colors)],
        start_s=context.start_s,
        end_s=context.end_s,
        max_points=context.max_points,
    )
    if gps_speed is not None:
        series.append(gps_speed)

    tecs = find_dataset(context.ulog, "tecs_status")
    for field_name in ("true_airspeed_sp", "airspeed_sp"):
        setpoint = series_from_field(
            tecs,
            field_name,
            key="airspeed_airspeed_setpoint",
            label="Airspeed setpoint",
            color=context.colors[3 % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            interpolation="step_after",
        )
        if setpoint is not None:
            series.append(setpoint)
            break

    return timeseries_plot(
        "airspeed",
        "Airspeed",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
    )


def build_manual_control_plot(context: PlotContext) -> dict[str, Any] | None:
    manual = find_dataset(context.ulog, "manual_control_setpoint")
    if manual is None:
        return _build_raw_rc_plot(context)

    series = []
    definitions = (
        (("roll", "y"), "Roll", None, "linear"),
        (("pitch", "x"), "Pitch", None, "linear"),
        (("yaw", "r"), "Yaw", None, "linear"),
        (("throttle", "z"), "Throttle", None, "linear"),
        (("aux1",), "Aux1", None, "linear"),
        (("aux2",), "Aux2", None, "linear"),
    )
    for candidates, label, transform, interpolation in definitions:
        item = _first_field_series(
            context,
            manual,
            candidates,
            key=f"manual_control_{_slug(label)}",
            label=label,
            color=context.colors[len(series) % len(context.colors)],
            transform=transform,
            interpolation=interpolation,
        )
        if item is not None:
            series.append(item)

    switches = find_dataset(context.ulog, "manual_control_switches") or manual
    mode_slot = series_from_field(
        switches,
        "mode_slot",
        key="manual_control_mode_slot",
        label="Mode slot",
        color=context.colors[len(series) % len(context.colors)],
        start_s=context.start_s,
        end_s=context.end_s,
        max_points=context.max_points,
        transform=lambda value: value / 6.0,
        interpolation="step_after",
        normalize=False,
    )
    if mode_slot is not None:
        series.append(mode_slot)
    kill_switch = series_from_field(
        switches,
        "kill_switch",
        key="manual_control_kill_switch",
        label="Kill switch",
        color=context.colors[len(series) % len(context.colors)],
        start_s=context.start_s,
        end_s=context.end_s,
        max_points=context.max_points,
        transform=lambda value: 1.0 if value == 1.0 else 0.0,
        interpolation="step_after",
        normalize=False,
    )
    if kill_switch is not None:
        series.append(kill_switch)

    return timeseries_plot(
        "manual_control",
        "Manual Control Inputs",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
        y_range=(-1.1, 1.1),
    )


def build_actuator_output_plots(context: PlotContext) -> list[dict[str, Any]]:
    modern_topics = (
        ("actuator_motors", "motor_outputs", "Motor Outputs", "Motor"),
        ("actuator_servos", "servo_outputs", "Servo Outputs", "Servo"),
    )
    modern_present = any(iter_topic_datasets(context.ulog, topic) for topic, _, _, _ in modern_topics)
    plots = []
    if modern_present:
        for topic_name, base_id, title, label_prefix in modern_topics:
            datasets = iter_topic_datasets(context.ulog, topic_name)
            for dataset in datasets:
                instance = dataset_instance(dataset)
                series = _indexed_dataset_series(
                    context,
                    dataset,
                    "control",
                    label_prefix,
                    plot_id=_instance_id(base_id, instance),
                )
                plot = timeseries_plot(
                    _instance_id(base_id, instance),
                    _instance_title(title, instance, len(datasets)),
                    series,
                    overlays=context.overlays,
                    time_range_s=context.time_range_s,
                    y_range=(-1.0, 1.0),
                )
                if plot is not None:
                    plots.append(plot)
        return plots

    datasets = iter_topic_datasets(context.ulog, "actuator_outputs")
    for dataset in datasets:
        instance = dataset_instance(dataset)
        available = indexed_fields(dataset, "output")
        noutputs = getattr(dataset, "data", {}).get("noutputs")
        max_outputs = None
        if noutputs is not None:
            try:
                finite_counts = np.asarray(noutputs, dtype=float)
                finite_counts = finite_counts[np.isfinite(finite_counts)]
                if finite_counts.size:
                    max_outputs = int(np.max(finite_counts))
            except (TypeError, ValueError):
                pass

        if max_outputs is not None:
            available = [(index, field) for index, field in available if index < max_outputs]
        changing = False
        for _, field_name in available:
            values = getattr(dataset, "data", {}).get(field_name)
            try:
                finite = np.asarray(values, dtype=float)
                finite = finite[np.isfinite(finite)]
            except (TypeError, ValueError):
                continue
            if finite.size > 1 and np.any(finite != finite[0]):
                changing = True
                break
        if not changing:
            continue

        plot_id = f"actuator_outputs_{instance}"
        series = _indexed_dataset_series(
            context,
            dataset,
            "output",
            "Output",
            plot_id=plot_id,
            fields=available,
            one_based_labels=False,
        )
        plot = timeseries_plot(
            plot_id,
            f"Actuator Outputs (instance {instance})",
            series,
            overlays=context.overlays,
            time_range_s=context.time_range_s,
        )
        if plot is not None:
            plots.append(plot)
    return plots


def build_motor_rpm_plots(context: PlotContext) -> list[dict[str, Any]]:
    plots = []
    datasets = iter_topic_datasets(context.ulog, "esc_status")
    for dataset in datasets:
        instance = dataset_instance(dataset)
        series = []
        for esc_index, field_name in indexed_fields(dataset, "esc", suffix=".esc_rpm"):
            if not field_has_nonzero_values(dataset, field_name, threshold=0.001):
                continue
            item = series_from_field(
                dataset,
                field_name,
                key=f"motor_rpm_{instance}_{esc_index}",
                label=f"ESC {esc_index} RPM",
                color=context.colors[len(series) % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
                unit="RPM",
                axis_label="[RPM]",
            )
            if item is not None:
                series.append(item)
        plot = timeseries_plot(
            _instance_id("motor_rpm", instance),
            _instance_title("Motor RPM", instance, len(datasets)),
            series,
            overlays=context.overlays,
            time_range_s=context.time_range_s,
        )
        if plot is not None:
            plots.append(plot)
    return plots


def build_vibration_plot(context: PlotContext) -> dict[str, Any] | None:
    series = []
    for dataset in iter_topic_datasets(context.ulog, "vehicle_imu_status"):
        instance = dataset_instance(dataset)
        item = series_from_field(
            dataset,
            "accel_vibration_metric",
            key=f"vibration_imu_{instance}",
            label=f"IMU {instance}",
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
        )
        if item is not None:
            series.append(item)
    return timeseries_plot(
        "vibration",
        "Vibration Metrics",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
        horizontal_bands=[
            {"min": None, "max": 4.905, "color": "#4caf50", "alpha": 0.10, "label": "Normal"},
            {"min": 4.905, "max": 9.81, "color": "#ff9800", "alpha": 0.10, "label": "Elevated"},
            {"min": 9.81, "max": None, "color": "#f44336", "alpha": 0.10, "label": "High"},
        ],
    )


def build_distance_sensor_plot(context: PlotContext) -> dict[str, Any] | None:
    series = []
    datasets = iter_topic_datasets(context.ulog, "distance_sensor")
    for dataset in datasets:
        instance = dataset_instance(dataset)
        suffix = f" {instance}" if len(datasets) > 1 else ""
        for field_name, label in (
            ("current_distance", f"Distance{suffix}"),
            ("variance", f"Variance{suffix}"),
        ):
            item = series_from_field(
                dataset,
                field_name,
                key=f"distance_sensor_{instance}_{field_name}",
                label=label,
                color=context.colors[len(series) % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
            )
            if item is not None:
                series.append(item)

    local_position = find_dataset(context.ulog, "vehicle_local_position")
    for field_name, label, interpolation in (
        ("dist_bottom", "Estimated Distance Bottom", "linear"),
        ("dist_bottom_valid", "Dist Bottom Valid", "step_after"),
    ):
        item = series_from_field(
            local_position,
            field_name,
            key=f"distance_sensor_{field_name}",
            label=label,
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            interpolation=interpolation,
        )
        if item is not None:
            series.append(item)
    return timeseries_plot(
        "distance_sensor",
        "Distance Sensor",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
    )


def build_thrust_magnetic_plot(context: PlotContext) -> dict[str, Any] | None:
    series = []
    magnetometer = (
        find_dataset(context.ulog, "vehicle_magnetometer")
        or find_dataset(context.ulog, "sensor_combined")
    )
    magnetic_norm = _derived_norm_series(
        context,
        magnetometer,
        ("magnetometer_ga[0]", "magnetometer_ga[1]", "magnetometer_ga[2]"),
        key="thrust_magnetic_field_norm",
        label="Norm of Magnetic Field",
        signal="derived.magnetometer.norm",
        color=context.colors[0],
        unit="gauss",
    )
    if magnetic_norm is not None:
        series.append(magnetic_norm)

    thrust_datasets = iter_topic_datasets(context.ulog, "vehicle_thrust_setpoint")
    modern_thrust_added = False
    for dataset in thrust_datasets:
        instance = dataset_instance(dataset)
        thrust = _derived_norm_series(
            context,
            dataset,
            ("xyz[0]", "xyz[1]", "xyz[2]"),
            key=f"thrust_magnetic_thrust_{instance}",
            label=_instance_title("Thrust", instance, len(thrust_datasets)),
            signal=f"derived.vehicle_thrust_setpoint[{instance}].norm",
            color=context.colors[len(series) % len(context.colors)],
            unit=None,
        )
        if thrust is not None and any(
            abs(value) > 0.001
            for value in thrust["values"]
            if math.isfinite(value)
        ):
            series.append(thrust)
            modern_thrust_added = True
    if not modern_thrust_added:
        legacy = find_dataset(context.ulog, "actuator_controls_0")
        thrust = series_from_field(
            legacy,
            "control[3]",
            key="thrust_magnetic_thrust",
            label="Thrust",
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            normalize=False,
        )
        if thrust is not None:
            series.append(thrust)

    return timeseries_plot(
        "thrust_magnetic_field",
        "Thrust and Magnetic Field",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
    )


def build_temperature_plot(context: PlotContext) -> dict[str, Any] | None:
    series = []
    simple_topics = (
        ("sensor_baro", "temperature", "Barometer"),
        ("sensor_accel", "temperature", "Accelerometer"),
        ("airspeed", "air_temperature_celsius", "Airspeed"),
        ("battery_status", "temperature", "Battery"),
    )
    for topic_name, field_name, label in simple_topics:
        datasets = iter_topic_datasets(context.ulog, topic_name)
        for dataset in datasets:
            instance = dataset_instance(dataset)
            item = series_from_field(
                dataset,
                field_name,
                key=f"temperature_{topic_name}_{instance}",
                label=_instance_title(label, instance, len(datasets)),
                color=context.colors[len(series) % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
            )
            if item is not None:
                series.append(item)

    esc_status_datasets = iter_topic_datasets(context.ulog, "esc_status")
    for dataset in esc_status_datasets:
        esc_instance = dataset_instance(dataset)
        for esc_index, field_name in indexed_fields(
            dataset,
            "esc",
            suffix=".esc_temperature",
        ):
            if not field_has_nonzero_values(dataset, field_name, threshold=0.001):
                continue
            item = series_from_field(
                dataset,
                field_name,
                key=f"temperature_esc_{esc_instance}_{esc_index}",
                label=(
                    f"ESC {esc_index} temperature"
                    if len(esc_status_datasets) <= 1
                    else f"ESC {esc_index} temperature (status {esc_instance})"
                ),
                color=context.colors[len(series) % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
                unit="C",
                axis_label="[C]",
            )
            if item is not None:
                series.append(item)
    return timeseries_plot(
        "temperature",
        "Temperature",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
    )


def build_estimator_flags_plot(context: PlotContext) -> dict[str, Any] | None:
    dataset = find_dataset(context.ulog, "estimator_status")
    if dataset is None:
        return None
    series = []

    typed_boolean_fields = boolean_fields(dataset)
    for field_name in typed_boolean_fields:
        if not field_has_nonzero_values(dataset, field_name):
            continue
        item = series_from_field(
            dataset,
            field_name,
            key=f"estimator_flags_{_slug(field_name)}",
            label=_humanize(field_name),
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            interpolation="step_after",
            normalize=False,
        )
        if item is not None:
            series.append(item)

    packed_fields = sorted(
        field_name
        for field_name in (getattr(dataset, "data", {}) or {})
        if str(field_name).endswith("_flags")
    )
    for field_name in packed_fields:
        if not field_has_nonzero_values(dataset, field_name):
            continue
        item = series_from_field(
            dataset,
            field_name,
            key=f"estimator_flags_{_slug(field_name)}",
            label=_humanize(field_name),
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            interpolation="step_after",
            normalize=False,
        )
        if item is not None:
            series.append(item)

    if not series and (packed_fields or typed_boolean_fields):
        representative = (
            packed_fields[0]
            if packed_fields
            else typed_boolean_fields[0]
        )
        clear = series_from_field(
            dataset,
            representative,
            key=f"estimator_flags_{_slug(representative)}",
            label=_humanize(representative),
            color=context.colors[0],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            interpolation="step_after",
            normalize=False,
        )
        if clear is not None:
            series.append(clear)
    return timeseries_plot(
        "estimator_flags",
        "Estimator Flags",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
    )


def build_failsafe_flags_plot(context: PlotContext) -> dict[str, Any] | None:
    series = []
    vehicle_status = find_dataset(context.ulog, "vehicle_status")
    for field_name, label in (
        ("failsafe", "Failsafe"),
        ("failsafe_and_user_took_over", "User took over"),
    ):
        item = series_from_field(
            vehicle_status,
            field_name,
            key=f"failsafe_flags_{_slug(label)}",
            label=label,
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            interpolation="step_after",
            normalize=False,
        )
        if item is not None:
            series.append(item)

    failsafe = find_dataset(context.ulog, "failsafe_flags")
    for field_name in boolean_fields(failsafe):
        if not field_has_nonzero_values(failsafe, field_name):
            continue
        item = series_from_field(
            failsafe,
            field_name,
            key=f"failsafe_flags_{_slug(field_name)}",
            label=_humanize(field_name),
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            interpolation="step_after",
            normalize=False,
        )
        if item is not None:
            series.append(item)

    if failsafe is not None:
        for field in getattr(failsafe, "field_data", []) or []:
            field_name = str(getattr(field, "field_name", "") or "")
            if (
                not field_name
                or field_type(failsafe, field_name) != "uint8_t"
                or field_name in boolean_fields(failsafe)
            ):
                continue
            values = getattr(failsafe, "data", {}).get(field_name)
            try:
                numeric = np.asarray(values, dtype=float)
                finite = numeric[np.isfinite(numeric)]
            except (TypeError, ValueError):
                continue
            if not finite.size or np.max(finite) <= 0 or len(np.unique(finite)) > 16:
                continue
            item = series_from_field(
                failsafe,
                field_name,
                key=f"failsafe_flags_{_slug(field_name)}",
                label=_humanize(field_name),
                color=context.colors[len(series) % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
                interpolation="step_after",
                normalize=False,
            )
            if item is not None:
                series.append(item)

    return timeseries_plot(
        "failsafe_flags",
        "Failsafe Flags",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
    )


def build_sampling_regularity_plot(context: PlotContext) -> dict[str, Any] | None:
    sensor = find_dataset(context.ulog, "sensor_combined")
    series = []
    if sensor is not None:
        timestamps = (getattr(sensor, "data", {}) or {}).get("timestamp")
        if timestamps is not None and len(timestamps) > 1:
            timestamps_array = np.asarray(timestamps, dtype=float)
            differences = np.diff(timestamps_array)
            item = series_from_arrays(
                timestamps_array[1:],
                differences,
                key="sampling_regularity_delta_t",
                label="delta t (between logged samples)",
                signal="derived.sensor_combined.timestamp_delta",
                color=context.colors[2 % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
                unit="us",
                axis_label="[us]",
            )
            if item is not None:
                series.append(item)

    estimator = find_dataset(context.ulog, "estimator_status")
    slip = series_from_field(
        estimator,
        "time_slip",
        key="sampling_regularity_estimator_time_slip",
        label="Estimator time slip (cumulative)",
        color=context.colors[1 % len(context.colors)],
        start_s=context.start_s,
        end_s=context.end_s,
        max_points=context.max_points,
    )
    if slip is not None:
        series.append(slip)
    return timeseries_plot(
        "sampling_regularity",
        "Sampling Regularity of Sensor Data",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
        y_range=(0.0, 25_000.0),
    )


def build_fifo_plots(context: PlotContext) -> list[dict[str, Any]]:
    plots = []
    definitions = (
        ("sensor_accel_fifo", "Acceleration", "m/s^2", None),
        ("sensor_gyro_fifo", "Gyro", "deg/s", math.degrees),
    )
    for topic_name, sensor_label, unit, transform in definitions:
        datasets = iter_topic_datasets(context.ulog, topic_name)
        for dataset in datasets:
            expanded = expand_fifo_samples(getattr(dataset, "data", {}) or {})
            if expanded is None:
                continue
            instance = dataset_instance(dataset)
            raw_series = []
            for axis_index, axis in enumerate(("x", "y", "z")):
                item = series_from_arrays(
                    expanded["timestamp"],
                    expanded[axis],
                    key=f"{topic_name}_{instance}_{axis}",
                    label=axis.upper(),
                    signal=f"derived.{topic_name}[{instance}].{axis}",
                    color=context.colors[axis_index % len(context.colors)],
                    start_s=context.start_s,
                    end_s=context.end_s,
                    max_points=context.max_points,
                    transform=transform,
                    unit=unit,
                    axis_label=f"[{unit}]",
                )
                if item is not None:
                    raw_series.append(item)
            raw_title = (
                f"Raw Acceleration (FIFO, IMU{instance})"
                if topic_name == "sensor_accel_fifo"
                else f"Raw Gyro (FIFO, IMU{instance})"
            )
            raw_plot = timeseries_plot(
                f"{topic_name}_raw_{instance}",
                raw_title,
                raw_series,
                overlays=context.overlays,
                time_range_s=context.time_range_s,
            )
            if raw_plot is not None:
                plots.append(raw_plot)

            spectrogram = summed_spectrogram(
                expanded["timestamp"],
                [expanded["x"], expanded["y"], expanded["z"]],
                max_time_bins=context.max_points,
            )
            if spectrogram is not None:
                spec_title = (
                    f"Acceleration Power Spectral Density (FIFO, IMU{instance})"
                    if topic_name == "sensor_accel_fifo"
                    else f"Gyro Power Spectral Density (FIFO, IMU{instance})"
                )
                plots.append(
                    _spectrogram_payload(
                        f"{topic_name}_spectrogram_{instance}",
                        spec_title,
                        spectrogram,
                        source=f"{topic_name}[{instance}]",
                        fields=["x", "y", "z"],
                        labels=["X", "Y", "Z"],
                    )
                )

            if topic_name == "sensor_accel_fifo":
                regularity = _fifo_sampling_regularity_plot(context, dataset)
                if regularity is not None:
                    plots.append(regularity)
    return plots


def build_fft_plots(context: PlotContext) -> list[dict[str, Any]]:
    definitions = (
        (
            "actuator_controls_fft",
            "Actuator Controls FFT",
            (("vehicle_torque_setpoint", ("xyz[0]", "xyz[1]", "xyz[2]")),),
            ("Roll", "Pitch", "Yaw"),
        ),
        (
            "angular_velocity_fft",
            "Angular Velocity FFT",
            (("vehicle_angular_velocity", ("xyz[0]", "xyz[1]", "xyz[2]")),),
            ("Rollspeed", "Pitchspeed", "Yawspeed"),
        ),
        (
            "angular_acceleration_fft",
            "Angular Acceleration FFT",
            (("vehicle_angular_acceleration", ("xyz[0]", "xyz[1]", "xyz[2]")),),
            ("Roll accel", "Pitch accel", "Yaw accel"),
        ),
    )
    legacy_controls = ("actuator_controls_0", ("control[0]", "control[1]", "control[2]"))
    plots = []
    for plot_id, title, raw_candidates, labels in definitions:
        candidates = list(raw_candidates)
        if plot_id == "actuator_controls_fft":
            candidates.append(legacy_controls)
        selected = _first_dataset_fields(context.ulog, candidates)
        if selected is None:
            continue
        dataset, fields = selected
        data = getattr(dataset, "data", {}) or {}
        timestamp_field = "timestamp_sample" if "timestamp_sample" in data else "timestamp"
        spectrum = positive_rfft_spectrum(
            data.get(timestamp_field),
            [data.get(field) for field in fields],
            max_points=context.max_points,
            summary_start_frequency_hz=FFT_MEAN_START_FREQUENCY_HZ,
        )
        if spectrum is None:
            continue
        series = []
        frequencies = spectrum["frequencies_hz"]
        for index, (amplitudes, label, field_name) in enumerate(
            zip(spectrum["amplitudes"], labels, fields)
        ):
            series.append(
                {
                    "key": f"{plot_id}_{_slug(label)}",
                    "label": label,
                    "signal": f"{getattr(dataset, 'name', '')}.{field_name}",
                    "unit": "amplitude",
                    "color": context.colors[index % len(context.colors)],
                    "frequencies_hz": _round_values(frequencies),
                    "values": _round_values(amplitudes),
                }
            )
        plots.append(
            {
                "id": plot_id,
                "title": title,
                "kind": "spectrum",
                "frequency_range_hz": [
                    round(float(frequencies[0]), 6),
                    round(float(frequencies[-1]), 6),
                ],
                "sampling_frequency_hz": round(
                    float(spectrum["sampling_frequency_hz"]),
                    6,
                ),
                "series": series,
                "frequency_markers": [],
                "horizontal_spans": [
                    {
                        "value": round(float(mean_amplitude), 6),
                        "start_x": FFT_MEAN_START_FREQUENCY_HZ,
                        "color": item["color"],
                        "series_key": item["key"],
                        "label": (
                            f"{item['label']} mean above "
                            f"{FFT_MEAN_START_FREQUENCY_HZ:g} Hz"
                        ),
                    }
                    for item, mean_amplitude in zip(
                        series,
                        spectrum["summary_mean_amplitudes"],
                    )
                    if mean_amplitude is not None
                ],
                "warnings": [],
            }
        )
    return plots


def _build_visual_odometry_attitude(
    context: PlotContext,
    dataset: Any,
) -> dict[str, Any] | None:
    series = []
    scalar_fields = _first_field_group(dataset, (("roll", "pitch", "yaw"),))
    if scalar_fields is not None:
        for index, (field_name, label) in enumerate(
            zip(scalar_fields, ("Roll", "Pitch", "Yaw"))
        ):
            item = series_from_field(
                dataset,
                field_name,
                key=f"visual_odometry_attitude_{_slug(label)}",
                label=label,
                color=context.colors[index % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
            )
            if item is not None:
                series.append(item)
    else:
        quaternion_fields = _first_field_group(
            dataset,
            (("q[0]", "q[1]", "q[2]", "q[3]"),),
        )
        data = getattr(dataset, "data", {}) or {}
        timestamps = data.get("timestamp")
        if quaternion_fields is not None and timestamps is not None:
            euler = _quaternion_to_euler(
                *(np.asarray(data[field], dtype=float) for field in quaternion_fields)
            )
            for index, (values, label) in enumerate(
                zip(euler, ("Roll", "Pitch", "Yaw"))
            ):
                item = series_from_arrays(
                    timestamps,
                    values,
                    key=f"visual_odometry_attitude_{_slug(label)}",
                    label=label,
                    signal=f"derived.vehicle_visual_odometry.{label.lower()}",
                    color=context.colors[index % len(context.colors)],
                    start_s=context.start_s,
                    end_s=context.end_s,
                    max_points=context.max_points,
                    unit="deg",
                    axis_label="[deg]",
                )
                if item is not None:
                    series.append(item)

    groundtruth = find_dataset(context.ulog, "vehicle_attitude_groundtruth")
    for field_name, label in zip(
        ("roll", "pitch", "yaw"),
        ("Roll Groundtruth", "Pitch Groundtruth", "Yaw Groundtruth"),
    ):
        item = series_from_field(
            groundtruth,
            field_name,
            key=f"visual_odometry_attitude_{_slug(label)}",
            label=label,
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
        )
        if item is not None:
            series.append(item)
    return timeseries_plot(
        "visual_odometry_attitude",
        "Visual Odometry Attitude",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
    )


def _build_visual_odometry_latency(
    context: PlotContext,
    dataset: Any,
) -> dict[str, Any] | None:
    data = getattr(dataset, "data", {}) or {}
    timestamp = data.get("timestamp")
    timestamp_sample = data.get("timestamp_sample")
    if timestamp is None or timestamp_sample is None:
        return None
    count = min(len(timestamp), len(timestamp_sample))
    latency_ms = (
        np.asarray(timestamp[:count], dtype=float)
        - np.asarray(timestamp_sample[:count], dtype=float)
    ) * 1.0e-3
    item = series_from_arrays(
        timestamp[:count],
        latency_ms,
        key="visual_odometry_latency",
        label="VIO Latency",
        signal="derived.vehicle_visual_odometry.latency",
        color=context.colors[0],
        start_s=context.start_s,
        end_s=context.end_s,
        max_points=context.max_points,
        unit="ms",
        axis_label="[ms]",
    )
    return timeseries_plot(
        "visual_odometry_latency",
        "Visual Odometry Latency",
        [item] if item is not None else [],
        overlays=context.overlays,
        time_range_s=context.time_range_s,
    )


def _vector_plot(
    context: PlotContext,
    dataset: Any,
    plot_id: str,
    title: str,
    field_groups: Sequence[Sequence[str]],
    labels: Sequence[str],
    *,
    unit: str | None,
    transform: Any = None,
    normalize: bool = True,
    groundtruth_topic: str | Sequence[str] | None = None,
    groundtruth_groups: Sequence[Sequence[str]] = (),
    groundtruth_labels: Sequence[str] = (),
    groundtruth_transform: Any = None,
    groundtruth_normalize: bool = True,
) -> dict[str, Any] | None:
    series = []
    fields = _first_field_group(dataset, field_groups)
    if fields is not None:
        for field_name, label in zip(fields, labels):
            item = series_from_field(
                dataset,
                field_name,
                key=f"{plot_id}_{_slug(label)}",
                label=label,
                color=context.colors[len(series) % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
                transform=transform,
                normalize=normalize,
                unit=unit,
                axis_label=f"[{unit}]" if unit else None,
            )
            if item is not None:
                series.append(item)

    groundtruth = None
    if groundtruth_topic:
        candidates = (
            (groundtruth_topic,)
            if isinstance(groundtruth_topic, str)
            else groundtruth_topic
        )
        for topic_name in candidates:
            groundtruth = find_dataset(context.ulog, topic_name)
            if groundtruth is not None:
                break
    groundtruth_fields = _first_field_group(groundtruth, groundtruth_groups)
    if groundtruth_fields is not None:
        for field_name, label in zip(groundtruth_fields, groundtruth_labels):
            item = series_from_field(
                groundtruth,
                field_name,
                key=f"{plot_id}_{_slug(label)}",
                label=label,
                color=context.colors[len(series) % len(context.colors)],
                start_s=context.start_s,
                end_s=context.end_s,
                max_points=context.max_points,
                transform=groundtruth_transform,
                normalize=groundtruth_normalize,
                unit=unit,
                axis_label=f"[{unit}]" if unit else None,
            )
            if item is not None:
                series.append(item)
    return timeseries_plot(
        plot_id,
        title,
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
    )


def _validated_airspeed_series(
    context: PlotContext,
    dataset: Any,
) -> dict[str, Any] | None:
    if dataset is None:
        return None
    data = getattr(dataset, "data", {}) or {}
    timestamps = data.get("timestamp")
    measured = data.get("true_airspeed_m_s")
    estimated = data.get("true_ground_minus_wind_m_s")
    valid = data.get("airspeed_sensor_measurement_valid")
    if timestamps is None or (measured is None and estimated is None):
        return None

    count = min(
        len(timestamps),
        *(len(values) for values in (measured, estimated, valid) if values is not None),
    )
    selected = []
    for index in range(count):
        is_valid = bool(valid[index]) if valid is not None else measured is not None
        primary = measured if is_valid else estimated
        fallback = estimated if is_valid else measured
        value = primary[index] if primary is not None else None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        if not math.isfinite(numeric) and fallback is not None:
            try:
                numeric = float(fallback[index])
            except (TypeError, ValueError):
                numeric = math.nan
        selected.append(numeric)

    return series_from_arrays(
        timestamps[:count],
        selected,
        key="airspeed_true_airspeed",
        label="True airspeed",
        signal="derived.airspeed_validated.selected_airspeed",
        color=context.colors[1 % len(context.colors)],
        start_s=context.start_s,
        end_s=context.end_s,
        max_points=context.max_points,
        unit="m/s",
        axis_label="[m/s]",
    )


def _build_raw_rc_plot(context: PlotContext) -> dict[str, Any] | None:
    dataset = find_dataset(context.ulog, "rc_channels")
    if dataset is None:
        return None
    series = []
    try:
        px4_ulog = PX4ULog(context.ulog)
    except Exception:
        px4_ulog = None
    channels = indexed_fields(dataset, "channels")
    channel_count = (getattr(dataset, "data", {}) or {}).get("channel_count")
    if channel_count is not None:
        try:
            finite_counts = np.asarray(channel_count, dtype=float)
            finite_counts = finite_counts[np.isfinite(finite_counts)]
            if finite_counts.size:
                active_count = max(0, int(np.max(finite_counts)))
                channels = [
                    (index, field_name)
                    for index, field_name in channels
                    if index < active_count
                ]
        except (TypeError, ValueError):
            pass
    for channel_index, field_name in channels:
        configured_names = None
        if px4_ulog is not None:
            try:
                configured_names = px4_ulog.get_configured_rc_input_names(channel_index)
            except Exception:
                configured_names = None
        label = f"Channel {channel_index}"
        if configured_names:
            label += f" ({', '.join(configured_names)})"
        item = series_from_field(
            dataset,
            field_name,
            key=f"raw_radio_control_channel_{channel_index}",
            label=label,
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            normalize=False,
        )
        if item is not None:
            series.append(item)
    return timeseries_plot(
        "raw_radio_control",
        "Raw Radio Control Inputs",
        series,
        overlays=context.overlays,
        time_range_s=context.time_range_s,
        y_range=(-1.1, 1.1),
    )


def _indexed_dataset_series(
    context: PlotContext,
    dataset: Any,
    prefix: str,
    label_prefix: str,
    *,
    plot_id: str,
    fields: Sequence[tuple[int, str]] | None = None,
    one_based_labels: bool = True,
) -> list[dict[str, Any]]:
    series = []
    for index, field_name in fields or indexed_fields(dataset, prefix):
        if not field_has_finite_values(dataset, field_name):
            continue
        display_index = index + 1 if one_based_labels else index
        item = series_from_field(
            dataset,
            field_name,
            key=f"{plot_id}_{index}",
            label=f"{label_prefix} {display_index}",
            color=context.colors[len(series) % len(context.colors)],
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            normalize=False,
        )
        if item is not None:
            series.append(item)
    return series


def _fifo_sampling_regularity_plot(
    context: PlotContext,
    dataset: Any,
) -> dict[str, Any] | None:
    data = getattr(dataset, "data", {}) or {}
    timestamps = data.get("timestamp")
    if timestamps is None or len(timestamps) < 2:
        return None
    timestamps_array = np.asarray(timestamps, dtype=float)
    differences = np.diff(timestamps_array)
    instance = dataset_instance(dataset)
    item = series_from_arrays(
        timestamps_array[1:],
        differences,
        key=f"sensor_accel_fifo_regularity_{instance}",
        label="delta t (between logged samples)",
        signal=f"derived.sensor_accel_fifo[{instance}].timestamp_delta",
        color=context.colors[2 % len(context.colors)],
        start_s=context.start_s,
        end_s=context.end_s,
        max_points=context.max_points,
        unit="us",
        axis_label="[us]",
    )
    return timeseries_plot(
        f"sensor_accel_fifo_regularity_{instance}",
        f"Sampling Regularity of Sensor Data (FIFO, IMU{instance})",
        [item] if item is not None else [],
        overlays=context.overlays,
        time_range_s=context.time_range_s,
        y_range=(0.0, 25_000.0),
    )


def _spectrogram_payload(
    plot_id: str,
    title: str,
    spectrogram: dict[str, Any],
    *,
    source: str,
    fields: list[str],
    labels: list[str],
) -> dict[str, Any]:
    frequencies = spectrogram["frequencies_hz"]
    times = spectrogram.get("time_s")
    if times is None:
        offsets = spectrogram.get("time_offsets_s") or []
        origin_s = float(spectrogram.get("first_timestamp_us", 0.0)) * 1.0e-6
        times = [origin_s + float(offset) for offset in offsets]
    values_db = np.asarray(spectrogram["values_db"], dtype=float)
    finite = values_db[np.isfinite(values_db)]
    return {
        "id": plot_id,
        "title": title,
        "kind": "spectrogram",
        "time_range_s": [
            round(float(times[0]), 6),
            round(float(times[-1]), 6),
        ],
        "frequency_range_hz": [
            round(float(frequencies[0]), 6),
            round(float(frequencies[-1]), 6),
        ],
        "time_s": _round_values(times),
        "frequencies_hz": _round_values(frequencies),
        "values_db": [_round_values(row) for row in values_db],
        "value_range_db": [
            round(float(np.min(finite)), 6),
            round(float(np.max(finite)), 6),
        ],
        "sampling_frequency_hz": round(
            float(spectrogram["sampling_frequency_hz"]),
            6,
        ),
        "source": source,
        "fields": fields,
        "labels": labels,
        "warnings": [],
    }


def _derived_norm_series(
    context: PlotContext,
    dataset: Any,
    fields: Sequence[str],
    *,
    key: str,
    label: str,
    signal: str,
    color: str,
    unit: str | None,
) -> dict[str, Any] | None:
    if dataset is None:
        return None
    data = getattr(dataset, "data", {}) or {}
    timestamps = data.get("timestamp")
    arrays = [data.get(field) for field in fields]
    if timestamps is None or any(values is None for values in arrays):
        return None
    count = min(len(timestamps), *(len(values) for values in arrays))
    stacked = np.vstack([np.asarray(values[:count], dtype=float) for values in arrays])
    norm = np.sqrt(np.sum(stacked * stacked, axis=0))
    return series_from_arrays(
        timestamps[:count],
        norm,
        key=key,
        label=label,
        signal=signal,
        color=color,
        start_s=context.start_s,
        end_s=context.end_s,
        max_points=context.max_points,
        unit=unit,
        axis_label=f"[{unit}]" if unit else None,
    )


def _first_field_series(
    context: PlotContext,
    dataset: Any,
    fields: Iterable[str],
    *,
    key: str,
    label: str,
    color: str,
    transform: Any = None,
    interpolation: str = "linear",
) -> dict[str, Any] | None:
    for field_name in fields:
        item = series_from_field(
            dataset,
            field_name,
            key=key,
            label=label,
            color=color,
            start_s=context.start_s,
            end_s=context.end_s,
            max_points=context.max_points,
            transform=transform,
            interpolation=interpolation,
        )
        if item is not None:
            return item
    return None


def _first_field_group(
    dataset: Any,
    groups: Sequence[Sequence[str]],
) -> tuple[str, ...] | None:
    if dataset is None:
        return None
    data = getattr(dataset, "data", {}) or {}
    for fields in groups:
        if all(field in data for field in fields):
            return tuple(fields)
    return None


def _first_dataset_fields(
    ulog: Any,
    candidates: Sequence[tuple[str, Sequence[str]]],
) -> tuple[Any, tuple[str, ...]] | None:
    for topic_name, fields in candidates:
        dataset = find_dataset(ulog, topic_name)
        resolved = _first_field_group(dataset, (fields,))
        if resolved is not None:
            return dataset, resolved
    return None


def _quaternion_to_euler(
    q0: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    q3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(len(q0), len(q1), len(q2), len(q3))
    w = np.asarray(q0[:count], dtype=float)
    x = np.asarray(q1[:count], dtype=float)
    y = np.asarray(q2[:count], dtype=float)
    z = np.asarray(q3[:count], dtype=float)
    norms = np.sqrt(w * w + x * x + y * y + z * z)
    norms = np.where(norms > np.finfo(float).eps, norms, 1.0)
    w, x, y, z = w / norms, x / norms, y / norms, z / norms
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def _instance_id(base: str, instance: int) -> str:
    return base if instance == 0 else f"{base}_{instance}"


def _instance_title(base: str, instance: int, instance_count: int) -> str:
    return base if instance == 0 and instance_count <= 1 else f"{base} {instance}"


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def _round_values(values: Iterable[Any]) -> list[float]:
    return [round(float(value), 6) for value in values]
