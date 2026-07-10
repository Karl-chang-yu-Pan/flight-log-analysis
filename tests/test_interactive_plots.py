import math
from types import SimpleNamespace

import numpy as np

import flight_log_agent.ulog.interactive_plots as ulog_interactive_plots


def test_build_interactive_plot_payload_includes_local_position_xy(monkeypatch):
    ulog = _fake_ulog(
        {
            "vehicle_local_position": {
                "timestamp": [0, 1_000_000, 2_000_000],
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 2.0, 4.0],
                "z": [0.0, -1.0, -2.0],
                "vx": [0.0, 0.5, 1.0],
                "vy": [0.0, 0.25, 0.5],
                "vz": [0.0, -0.1, -0.2],
            },
            "vehicle_local_position_setpoint": {
                "timestamp": [0, 1_000_000, 2_000_000],
                "x": [0.5, 1.5, 2.5],
                "y": [1.0, 3.0, 5.0],
                "z": [-0.5, -1.5, -2.5],
                "vx": [0.1, 0.6, 1.1],
                "vy": [0.1, 0.35, 0.6],
                "vz": [0.0, -0.2, -0.3],
            },
            "vehicle_gps_position": {
                "timestamp": [0, 1_000_000, 2_000_000],
                "fix_type": [3, 3, 3],
                "lat": [0, 0, 0],
                "lon": [0, 10_000, 20_000],
                "alt": [0, 1000, 2000],
            },
            "position_setpoint_triplet": {
                "timestamp": [0, 1_000_000, 2_000_000],
                "current.lat": [0.0, 0.0, 0.0],
                "current.lon": [0.0, 0.001, 0.002],
                "current.alt": [0.0, 1.0, 2.0],
            },
        }
    )
    monkeypatch.setattr(ulog_interactive_plots, "ULog", lambda path: ulog)
    monkeypatch.setattr(ulog_interactive_plots, "prepare_ulog_for_plotting", lambda parsed: parsed)

    payload = ulog_interactive_plots.build_interactive_plot_payload("flight.ulg")

    assert payload["time_range_s"] == [-0.1, 2.1]
    local_position = payload["plots"][0]
    assert local_position["id"] == "local_position_2d"
    assert local_position["kind"] == "local_position"
    assert local_position["scale_from"] == "position"
    assert local_position["min_range"] == 5.0
    assert local_position["zoom_out_factor"] == 1.3
    assert [trace["key"] for trace in local_position["traces"]] == [
        "position",
        "setpoint",
        "gps_projected",
        "position_setpoints",
    ]
    assert [trace["label"] for trace in local_position["traces"]] == [
        "Estimated",
        "Setpoint",
        "GPS (projected)",
        "Position Setpoints",
    ]
    assert local_position["traces"][0]["x"] == [0.0, 2.0, 4.0]
    assert local_position["traces"][0]["y"] == [0.0, 1.0, 2.0]
    assert local_position["traces"][1]["z"]["values"] == [-0.5, -1.5, -2.5]
    assert local_position["traces"][2]["color"] == "#56b4e9"
    assert local_position["traces"][3]["marker_only"] is True
    assert local_position["traces"][3]["color"] == "#cc79a7"

    local_timeseries = next(plot for plot in payload["plots"] if plot["id"] == "local_position")
    assert [series["label"] for series in local_timeseries["series"]] == [
        "X",
        "Y",
        "Z",
        "X setpoint",
        "Y setpoint",
        "Z setpoint",
    ]


def test_build_interactive_plot_payload_downsamples_series(monkeypatch):
    timestamps = [index * 1_000_000 for index in range(10)]
    ulog = _fake_ulog(
        {
            "vehicle_attitude": {
                "timestamp": timestamps,
                "roll": [float(index) for index in range(10)],
            },
        }
    )
    monkeypatch.setattr(ulog_interactive_plots, "ULog", lambda path: ulog)
    monkeypatch.setattr(ulog_interactive_plots, "prepare_ulog_for_plotting", lambda parsed: parsed)

    payload = ulog_interactive_plots.build_interactive_plot_payload("flight.ulg", max_points=4)

    attitude = next(plot for plot in payload["plots"] if plot["id"] == "attitude")
    roll = attitude["series"][0]
    assert roll["time_s"] == [0.0, 3.0, 6.0, 9.0]
    assert roll["values"] == [0.0, 171.887339, 343.774677, 515.662016]


def test_build_interactive_plot_payload_exposes_flight_review_fixed_y_ranges(monkeypatch):
    ulog = _fake_ulog(
        {
            "manual_control_setpoint": {
                "timestamp": [0, 1_000_000],
                "roll": [-0.5, 0.5],
            },
            "vehicle_gps_position": {
                "timestamp": [0, 1_000_000],
                "eph": [0.5, 1.0],
            },
            "cpuload": {
                "timestamp": [0, 1_000_000],
                "load": [0.2, 0.4],
            },
            "vehicle_status": {
                "timestamp": [0, 1_000_000],
                "nav_state": [0, 2],
            },
            "vtol_vehicle_status": {
                "timestamp": [0, 1_000_000],
                "vehicle_vtol_state": [1, 2],
            },
        }
    )
    monkeypatch.setattr(ulog_interactive_plots, "ULog", lambda path: ulog)
    monkeypatch.setattr(ulog_interactive_plots, "prepare_ulog_for_plotting", lambda parsed: parsed)

    payload = ulog_interactive_plots.build_interactive_plot_payload("flight.ulg")

    ranges = {
        plot["id"]: plot.get("y_range")
        for plot in payload["plots"]
    }
    assert ranges["manual_control"] == [-1.1, 1.1]
    assert ranges["gps_uncertainty"] == [0.0, 40.0]
    assert ranges["cpu_ram"] == [0.0, 1.0]

    manual_control = next(plot for plot in payload["plots"] if plot["id"] == "manual_control")
    assert {
        (overlay["kind"], overlay["label"], overlay["color"], overlay["alpha"])
        for overlay in manual_control["overlays"]
    } >= {
        ("mode_background", "Manual", "#cc0000", 0.09),
        ("vtol_background", "Transition", "#cc0000", 0.09),
    }


def test_build_interactive_plot_payload_adds_fixed_wing_body_velocity_angles(monkeypatch):
    ulog = _fake_ulog(
        {
            "vehicle_status": {
                "timestamp": np.array([0, 1_000_000]),
                "vehicle_type": np.array([2, 2]),
            },
            "vehicle_local_position": {
                "timestamp": np.array([0, 1_000_000]),
                "vx": np.array([10.0, 10.0]),
                "vy": np.array([0.0, 0.0]),
                "vz": np.array([1.0, 1.0]),
            },
            "vehicle_attitude": {
                "timestamp": np.array([0, 1_000_000]),
                "roll": np.array([0.0, 0.0]),
                "pitch": np.array([0.0, 0.0]),
                "yaw": np.array([0.0, 0.0]),
            },
        }
    )
    monkeypatch.setattr(ulog_interactive_plots, "ULog", lambda path: ulog)
    monkeypatch.setattr(ulog_interactive_plots, "prepare_ulog_for_plotting", lambda parsed: parsed)

    payload = ulog_interactive_plots.build_interactive_plot_payload("flight.ulg")

    plot = next(plot for plot in payload["plots"] if plot["id"] == "fixed_wing_body_velocity_angles")
    assert [series["label"] for series in plot["series"]] == ["AoA", "Sideslip", "Total Speed"]
    assert plot["series"][0]["unit"] == "deg"
    assert math.isclose(
        plot["series"][0]["values"][0],
        math.degrees(math.atan2(1.0, 10.0)),
        abs_tol=1e-6,
    )
    assert plot["series"][1]["values"] == [0.0, 0.0]
    assert plot["series"][2]["unit"] == "m/s"
    assert all(
        math.isclose(value, math.sqrt(101.0), abs_tol=1e-6)
        for value in plot["series"][2]["values"]
    )


def test_build_interactive_plot_payload_adds_multirotor_heading_velocity(monkeypatch):
    ulog = _fake_ulog(
        {
            "vehicle_status": {
                "timestamp": [0, 1_000_000],
                "vehicle_type": [1, 1],
            },
            "vehicle_local_position": {
                "timestamp": [0, 1_000_000],
                "vx": [1.0, 1.0],
                "vy": [0.0, 0.0],
            },
            "vehicle_attitude": {
                "timestamp": [0, 1_000_000],
                "yaw": [math.pi / 2.0, math.pi / 2.0],
            },
        }
    )
    monkeypatch.setattr(ulog_interactive_plots, "ULog", lambda path: ulog)
    monkeypatch.setattr(ulog_interactive_plots, "prepare_ulog_for_plotting", lambda parsed: parsed)

    payload = ulog_interactive_plots.build_interactive_plot_payload("flight.ulg")

    plot = next(plot for plot in payload["plots"] if plot["id"] == "multirotor_heading_velocity")
    forward = next(series for series in plot["series"] if series["label"] == "Forward")
    right = next(series for series in plot["series"] if series["label"] == "Right")
    assert all(math.isclose(value, 0.0, abs_tol=1e-6) for value in forward["values"])
    assert all(math.isclose(value, -1.0, abs_tol=1e-6) for value in right["values"])


def test_build_interactive_plot_payload_masks_vtol_velocity_plots(monkeypatch):
    ulog = _fake_ulog(
        {
            "vtol_vehicle_status": {
                "timestamp": [0, 1_000_000, 2_000_000],
                "vehicle_vtol_state": [3, 1, 2],
            },
            "vehicle_local_position": {
                "timestamp": [0, 1_000_000, 2_000_000],
                "vx": [1.0, 2.0, 3.0],
                "vy": [0.0, 0.0, 0.0],
                "vz": [0.0, 1.0, 1.0],
            },
            "vehicle_attitude": {
                "timestamp": [0, 1_000_000, 2_000_000],
                "roll": [0.0, 0.0, 0.0],
                "pitch": [0.0, 0.0, 0.0],
                "yaw": [0.0, 0.0, 0.0],
            },
        }
    )
    monkeypatch.setattr(ulog_interactive_plots, "ULog", lambda path: ulog)
    monkeypatch.setattr(ulog_interactive_plots, "prepare_ulog_for_plotting", lambda parsed: parsed)

    payload = ulog_interactive_plots.build_interactive_plot_payload("flight.ulg")

    multirotor = next(plot for plot in payload["plots"] if plot["id"] == "multirotor_heading_velocity")
    forward = next(series for series in multirotor["series"] if series["label"] == "Forward")
    assert forward["values"] == [1.0, 2.0, 0.0]

    fixed_wing = next(plot for plot in payload["plots"] if plot["id"] == "fixed_wing_body_velocity_angles")
    aoa = next(series for series in fixed_wing["series"] if series["label"] == "AoA")
    total_speed = next(series for series in fixed_wing["series"] if series["label"] == "Total Speed")
    assert aoa["values"][0] == 0.0
    assert math.isclose(aoa["values"][1], math.degrees(math.atan2(1.0, 2.0)), abs_tol=1e-6)
    assert math.isclose(aoa["values"][2], math.degrees(math.atan2(1.0, 3.0)), abs_tol=1e-6)
    assert total_speed["values"][0] == 0.0
    assert math.isclose(total_speed["values"][1], math.sqrt(5.0), abs_tol=1e-6)
    assert math.isclose(total_speed["values"][2], math.sqrt(10.0), abs_tol=1e-6)


def test_build_interactive_plot_payload_splits_vtol_actuator_controls(monkeypatch):
    ulog = _fake_ulog(
        {
            "actuator_controls_0": {
                "timestamp": [0, 1_000_000],
                "control[0]": [0.1, 0.2],
                "control[1]": [0.3, 0.4],
                "control[2]": [0.5, 0.6],
                "control[3]": [0.7, 0.8],
            },
            "actuator_controls_1": {
                "timestamp": [0, 1_000_000],
                "control[0]": [-0.1, -0.2],
                "control[1]": [-0.3, -0.4],
                "control[2]": [-0.5, -0.6],
                "control[3]": [0.9, 1.0],
            },
        }
    )
    monkeypatch.setattr(ulog_interactive_plots, "ULog", lambda path: ulog)
    monkeypatch.setattr(ulog_interactive_plots, "prepare_ulog_for_plotting", lambda parsed: parsed)

    payload = ulog_interactive_plots.build_interactive_plot_payload("flight.ulg")

    controls = next(plot for plot in payload["plots"] if plot["id"] == "actuator_controls")
    fixed_wing_controls = next(plot for plot in payload["plots"] if plot["id"] == "actuator_controls_1")
    assert controls["title"] == "Actuator Controls"
    assert fixed_wing_controls["title"] == "Actuator Controls 1 (VTOL in Fixed-Wing mode)"
    assert [series["label"] for series in controls["series"]] == ["Roll", "Pitch", "Yaw", "Thrust (up)"]
    assert [series["label"] for series in fixed_wing_controls["series"]] == [
        "Roll",
        "Pitch",
        "Yaw",
        "Thrust (forward)",
    ]


def test_build_interactive_plot_payload_adds_high_rate_spectrogram(monkeypatch):
    timestamps = [index * 1_000 for index in range(512)]
    values = [
        math.sin(2.0 * math.pi * 50.0 * (timestamp * 1.0e-6))
        for timestamp in timestamps
    ]
    ulog = _fake_ulog(
        {
            "vehicle_angular_velocity": {
                "timestamp_sample": timestamps,
                "timestamp": timestamps,
                "xyz[0]": values,
                "xyz[1]": [0.0 for _ in timestamps],
                "xyz[2]": [0.0 for _ in timestamps],
            },
        }
    )
    monkeypatch.setattr(ulog_interactive_plots, "ULog", lambda path: ulog)
    monkeypatch.setattr(ulog_interactive_plots, "prepare_ulog_for_plotting", lambda parsed: parsed)

    payload = ulog_interactive_plots.build_interactive_plot_payload("flight.ulg")

    plot = next(plot for plot in payload["plots"] if plot["id"] == "angular_velocity_spectrogram")
    assert plot["kind"] == "spectrogram"
    assert plot["sampling_frequency_hz"] >= 100.0
    assert plot["frequency_range_hz"][0] == 0.0
    assert len(plot["time_s"]) > 0
    assert len(plot["frequencies_hz"]) == len(plot["values_db"])
    assert all(len(row) == len(plot["time_s"]) for row in plot["values_db"])


def _fake_ulog(topic_data):
    data_list = [
        SimpleNamespace(name=name, data=data)
        for name, data in topic_data.items()
    ]

    def get_dataset(topic):
        for dataset in data_list:
            if dataset.name == topic:
                return dataset
        raise KeyError(topic)

    timestamps = [
        timestamp
        for data in topic_data.values()
        for timestamp in data.get("timestamp", [])
    ]

    return SimpleNamespace(
        data_list=data_list,
        msg_info_dict={},
        start_timestamp=min(timestamps) if timestamps else 0,
        last_timestamp=max(timestamps) if timestamps else 0,
        dropouts=[],
        changed_parameters=[],
        get_dataset=get_dataset,
    )
