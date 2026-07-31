import math
from types import SimpleNamespace

import numpy as np

import flight_log_agent.ulog.interactive_plots as interactive_plots
from flight_log_agent.ulog.interactive_plot_families import (
    PlotContext,
    build_actuator_output_plots,
    build_airspeed_plot,
    build_estimator_flags_plot,
    build_failsafe_flags_plot,
    build_fft_plots,
    build_fifo_plots,
    build_manual_control_plot,
    build_motor_rpm_plots,
    build_temperature_plot,
    build_thrust_magnetic_plot,
    build_visual_odometry_plots,
)


COLORS = interactive_plots.COLORS8


def test_payload_invokes_dynamic_output_and_rpm_families(monkeypatch):
    timestamps = np.array([0, 1_000_000])
    motor_data = {"timestamp": timestamps}
    for index in range(10):
        motor_data[f"control[{index}]"] = np.array([0.1 * index, 0.2 * index])

    ulog = _ulog(
        _dataset("actuator_motors", motor_data),
        _dataset(
            "esc_status",
            {
                "timestamp": timestamps,
                "esc[0].esc_rpm": np.array([0.0, 1_200.0]),
                "esc[1].esc_rpm": np.array([0.0, 0.0]),
            },
        ),
    )
    monkeypatch.setattr(interactive_plots, "ULog", lambda _path: ulog)
    monkeypatch.setattr(
        interactive_plots,
        "prepare_ulog_for_plotting",
        lambda parsed: parsed,
    )

    payload = interactive_plots.build_interactive_plot_payload("flight.ulg")

    motor_outputs = next(
        plot for plot in payload["plots"] if plot["id"] == "motor_outputs"
    )
    motor_rpm = next(
        plot for plot in payload["plots"] if plot["id"] == "motor_rpm"
    )
    assert len(motor_outputs["series"]) == 10
    assert [item["label"] for item in motor_rpm["series"]] == ["ESC 0 RPM"]
    assert payload["warnings"] == []


def test_legacy_outputs_and_raw_rc_derive_their_active_width():
    timestamps = np.array([0, 1_000_000, 2_000_000])
    outputs = _dataset(
        "actuator_outputs",
        {
            "timestamp": timestamps,
            "noutputs": np.array([3, 3, 3]),
            "output[0]": np.array([1_000.0, 1_100.0, 1_200.0]),
            "output[1]": np.array([1_000.0, 1_050.0, 1_100.0]),
            "output[2]": np.array([900.0, 950.0, 1_000.0]),
            "output[3]": np.array([2_000.0, 2_100.0, 2_200.0]),
        },
    )
    rc_data = {
        "timestamp": timestamps,
        "channel_count": np.array([3, 3, 3]),
    }
    for index in range(6):
        rc_data[f"channels[{index}]"] = np.array([0.1, 0.2, 0.3])
    context = _context(_ulog(outputs, _dataset("rc_channels", rc_data)))

    output_plots = build_actuator_output_plots(context)
    manual_plot = build_manual_control_plot(context)

    assert len(output_plots) == 1
    assert len(output_plots[0]["series"]) == 3
    assert output_plots[0]["y_range"] is None
    assert manual_plot is not None
    assert [item["label"].split(" ", 2)[:2] for item in manual_plot["series"]] == [
        ["Channel", "0"],
        ["Channel", "1"],
        ["Channel", "2"],
    ]


def test_visual_odometry_supports_vector_schema_quaternion_and_old_groundtruth():
    timestamps = np.array([0, 1_000_000])
    half_angle = math.pi / 4.0
    visual_odometry = _dataset(
        "vehicle_visual_odometry",
        {
            "timestamp": timestamps,
            "timestamp_sample": np.array([0, 990_000]),
            "position[0]": np.array([1.0, 2.0]),
            "position[1]": np.array([3.0, 4.0]),
            "position[2]": np.array([5.0, 6.0]),
            "velocity[0]": np.array([0.1, 0.2]),
            "velocity[1]": np.array([0.3, 0.4]),
            "velocity[2]": np.array([0.5, 0.6]),
            "q[0]": np.array([1.0, math.cos(half_angle)]),
            "q[1]": np.array([0.0, 0.0]),
            "q[2]": np.array([0.0, 0.0]),
            "q[3]": np.array([0.0, math.sin(half_angle)]),
            "angular_velocity[0]": np.array([0.1, 0.2]),
            "angular_velocity[1]": np.array([0.2, 0.3]),
            "angular_velocity[2]": np.array([0.3, 0.4]),
        },
    )
    old_groundtruth = _dataset(
        "vehicle_attitude_groundtruth",
        {
            "timestamp": timestamps,
            "roll": np.array([0.0, 0.0]),
            "pitch": np.array([0.0, 0.0]),
            "yaw": np.array([0.0, 0.0]),
            "rollspeed": np.array([0.1, 0.1]),
            "pitchspeed": np.array([0.2, 0.2]),
            "yawspeed": np.array([0.3, 0.3]),
        },
    )

    plots = build_visual_odometry_plots(
        _context(_ulog(visual_odometry, old_groundtruth))
    )
    by_id = {plot["id"]: plot for plot in plots}

    assert set(by_id) == {
        "visual_odometry_position",
        "visual_odometry_velocity",
        "visual_odometry_attitude",
        "visual_odometry_attitude_rate",
        "visual_odometry_latency",
    }
    yaw = next(
        item
        for item in by_id["visual_odometry_attitude"]["series"]
        if item["label"] == "Yaw"
    )
    assert math.isclose(yaw["values"][-1], 90.0, abs_tol=1e-6)
    rate_labels = {
        item["label"]
        for item in by_id["visual_odometry_attitude_rate"]["series"]
    }
    assert "Roll Rate Groundtruth" in rate_labels
    assert by_id["visual_odometry_latency"]["series"][0]["values"] == [0.0, 10.0]


def test_airspeed_selection_uses_validity_per_sample_and_local_velocity_fallback():
    timestamps = np.array([0, 1_000_000, 2_000_000])
    ulog = _ulog(
        _dataset(
            "airspeed_validated",
            {
                "timestamp": timestamps,
                "true_airspeed_m_s": np.array([10.0, 11.0, 12.0]),
                "true_ground_minus_wind_m_s": np.array([20.0, 21.0, 22.0]),
                "airspeed_sensor_measurement_valid": np.array([1, 0, 1]),
            },
        ),
        _dataset(
            "vehicle_local_position",
            {
                "timestamp": timestamps,
                "vx": np.array([3.0, 0.0, 5.0]),
                "vy": np.array([4.0, 2.0, 12.0]),
            },
        ),
    )

    plot = build_airspeed_plot(_context(ulog))

    assert plot is not None
    ground_speed = next(
        item for item in plot["series"] if item["label"] == "Ground Speed Estimated"
    )
    airspeed = next(
        item for item in plot["series"] if item["label"] == "True airspeed"
    )
    assert ground_speed["values"] == [5.0, 2.0, 13.0]
    assert airspeed["values"] == [10.0, 21.0, 12.0]


def test_flag_fields_are_discovered_from_schema_and_rendered_as_steps():
    timestamps = np.array([0, 1_000_000, 2_000_000])
    estimator = _dataset(
        "estimator_status",
        {
            "timestamp": timestamps,
            "future_domain_flags": np.array([0, 32, 0], dtype=np.uint32),
        },
        field_types={"future_domain_flags": "uint32_t"},
    )
    failsafe = _dataset(
        "failsafe_flags",
        {
            "timestamp": timestamps,
            "future_fault": np.array([0, 1, 0], dtype=np.int8),
            "future_severity": np.array([0, 2, 1], dtype=np.uint8),
        },
        field_types={
            "future_fault": "bool",
            "future_severity": "uint8_t",
        },
    )
    context = _context(_ulog(estimator, failsafe))

    estimator_plot = build_estimator_flags_plot(context)
    failsafe_plot = build_failsafe_flags_plot(context)

    assert estimator_plot is not None
    assert any(
        item["label"] == "Future domain flags"
        and item["values"] == [0.0, 32.0, 0.0]
        and item["interpolation"] == "step_after"
        for item in estimator_plot["series"]
    )
    assert failsafe_plot is not None
    assert {
        item["label"]
        for item in failsafe_plot["series"]
    } == {"Future fault", "Future severity"}
    assert all(
        item["interpolation"] == "step_after"
        for item in failsafe_plot["series"]
    )

    all_clear = _dataset(
        "estimator_status",
        {
            "timestamp": timestamps,
            "future_fault": np.array([0, 0, 0], dtype=np.int8),
        },
        field_types={"future_fault": "bool"},
    )
    all_clear_plot = build_estimator_flags_plot(_context(_ulog(all_clear)))
    assert all_clear_plot is not None
    assert all_clear_plot["series"][0]["label"] == "Future fault"
    assert all_clear_plot["series"][0]["values"] == [0.0, 0.0, 0.0]


def test_temperature_labels_esc_status_instances_unambiguously():
    timestamps = np.array([0, 1_000_000])
    context = _context(
        _ulog(
            _dataset(
                "esc_status",
                {
                    "timestamp": timestamps,
                    "esc[0].esc_temperature": np.array([30.0, 31.0]),
                },
            ),
            _dataset(
                "esc_status",
                {
                    "timestamp": timestamps,
                    "esc[0].esc_temperature": np.array([40.0, 41.0]),
                },
                multi_id=2,
            ),
        )
    )

    plot = build_temperature_plot(context)

    assert plot is not None
    assert [item["label"] for item in plot["series"]] == [
        "ESC 0 temperature (status 0)",
        "ESC 0 temperature (status 2)",
    ]


def test_thrust_plot_falls_back_when_modern_topic_has_no_usable_thrust():
    timestamps = np.array([0, 1_000_000])
    context = _context(
        _ulog(
            _dataset(
                "vehicle_thrust_setpoint",
                {
                    "timestamp": timestamps,
                    "xyz[0]": np.array([0.0, 0.0]),
                    "xyz[1]": np.array([0.0, 0.0]),
                    "xyz[2]": np.array([0.0, 0.0]),
                },
            ),
            _dataset(
                "actuator_controls_0",
                {
                    "timestamp": timestamps,
                    "control[3]": np.array([0.2, 0.5]),
                },
            ),
        )
    )

    plot = build_thrust_magnetic_plot(context)

    assert plot is not None
    assert [item["label"] for item in plot["series"]] == ["Thrust"]
    assert plot["series"][0]["values"] == [0.2, 0.5]


def test_cpu_mean_spans_use_full_resolution_values(monkeypatch):
    timestamps = np.arange(10, dtype=float) * 1_000_000.0
    values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ulog = _ulog(_dataset("cpuload", {"timestamp": timestamps, "load": values}))
    monkeypatch.setattr(interactive_plots, "ULog", lambda _path: ulog)
    monkeypatch.setattr(
        interactive_plots,
        "prepare_ulog_for_plotting",
        lambda parsed: parsed,
    )

    payload = interactive_plots.build_interactive_plot_payload(
        "flight.ulg",
        max_points=4,
    )

    cpu = next(plot for plot in payload["plots"] if plot["id"] == "cpu_ram")
    assert cpu["horizontal_spans"][0]["value"] == 0.1
    assert (
        cpu["horizontal_spans"][0]["series_key"]
        == cpu["series"][0]["key"]
    )


def test_temperature_rpm_fft_and_fifo_plots_use_logged_fields_and_instances():
    sample_count = 512
    high_rate_timestamps = np.arange(sample_count, dtype=float) * 1_000.0
    tone = np.sin(2.0 * math.pi * 50.0 * high_rate_timestamps * 1.0e-6)
    esc_status = _dataset(
        "esc_status",
        {
            "timestamp": np.array([0, 1_000_000]),
            "esc[0].esc_rpm": np.array([0.0, 1_500.0]),
            "esc[1].esc_rpm": np.array([0.0, 0.0]),
            "esc[0].esc_temperature": np.array([0.0, 45.0]),
            "esc[1].esc_temperature": np.array([0.0, 0.0]),
        },
        multi_id=2,
    )
    angular_velocity = _dataset(
        "vehicle_angular_velocity",
        {
            "timestamp": high_rate_timestamps,
            "timestamp_sample": high_rate_timestamps,
            "xyz[0]": tone,
            "xyz[1]": tone * 0.5,
            "xyz[2]": tone * 0.25,
        },
    )
    packet_count = 128
    packet_timestamps = np.arange(packet_count, dtype=float) * 2_000.0 + 1_000.0
    fifo_data = {
        "timestamp": packet_timestamps,
        "timestamp_sample": packet_timestamps,
        "dt": np.full(packet_count, 1_000.0),
        "samples": np.full(packet_count, 2),
        "scale": np.ones(packet_count),
    }
    for axis_index, axis in enumerate(("x", "y", "z")):
        fifo_data[f"{axis}[0]"] = np.sin(
            2.0 * math.pi * 40.0 * (packet_timestamps - 1_000.0) * 1.0e-6
        ) * (axis_index + 1)
        fifo_data[f"{axis}[1]"] = np.sin(
            2.0 * math.pi * 40.0 * packet_timestamps * 1.0e-6
        ) * (axis_index + 1)
    context = _context(
        _ulog(
            esc_status,
            angular_velocity,
            _dataset("sensor_accel_fifo", fifo_data, multi_id=3),
        ),
        end_s=2.0,
    )

    rpm_plots = build_motor_rpm_plots(context)
    temperature_plot = build_temperature_plot(context)
    fft_plots = build_fft_plots(context)
    fifo_plots = build_fifo_plots(context)

    assert [item["label"] for item in rpm_plots[0]["series"]] == ["ESC 0 RPM"]
    assert temperature_plot is not None
    assert [
        item["label"] for item in temperature_plot["series"]
    ] == ["ESC 0 temperature"]
    angular_fft = next(
        plot for plot in fft_plots if plot["id"] == "angular_velocity_fft"
    )
    peak_index = int(np.argmax(angular_fft["series"][0]["values"]))
    assert math.isclose(
        angular_fft["series"][0]["frequencies_hz"][peak_index],
        50.0,
        abs_tol=2.0,
    )
    assert len(angular_fft["horizontal_spans"]) == 3
    assert all(
        span["start_x"] == 40.0
        and span["series_key"] in {
            item["key"] for item in angular_fft["series"]
        }
        and "mean above 40 Hz" in span["label"]
        for span in angular_fft["horizontal_spans"]
    )
    assert {
        plot["id"]
        for plot in fifo_plots
    } == {
        "sensor_accel_fifo_raw_3",
        "sensor_accel_fifo_spectrogram_3",
        "sensor_accel_fifo_regularity_3",
    }


def _context(ulog, *, end_s=3.0):
    return PlotContext(
        ulog=ulog,
        start_s=-0.1,
        end_s=end_s,
        overlays=[],
        max_points=800,
        colors=COLORS,
    )


def _dataset(name, data, *, multi_id=0, field_types=None):
    field_types = field_types or {}
    fields = [
        SimpleNamespace(field_name=field_name, type_str=field_type)
        for field_name, field_type in field_types.items()
    ]
    return SimpleNamespace(
        name=name,
        multi_id=multi_id,
        data=data,
        field_data=fields,
    )


def _ulog(*datasets):
    timestamps = [
        float(timestamp)
        for dataset in datasets
        for timestamp in dataset.data.get("timestamp", [])
    ]

    def get_dataset(topic_name, multi_id=0):
        for dataset in datasets:
            if dataset.name == topic_name and dataset.multi_id == multi_id:
                return dataset
        raise KeyError((topic_name, multi_id))

    return SimpleNamespace(
        data_list=list(datasets),
        msg_info_dict={},
        initial_parameters={},
        start_timestamp=min(timestamps) if timestamps else 0,
        last_timestamp=max(timestamps) if timestamps else 0,
        dropouts=[],
        changed_parameters=[],
        get_dataset=get_dataset,
    )
