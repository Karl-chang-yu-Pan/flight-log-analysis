from types import SimpleNamespace

from flight_log_agent.analysis.log_evidence import ULogEvidenceIndex


def build_ulog():
    return SimpleNamespace(
        initial_parameters={"TEST_PARAM": 42},
        data_list=[
            SimpleNamespace(
                name="vehicle_status",
                multi_id=0,
                data={
                    "timestamp": [1_000_000, 2_000_000, 3_000_000],
                    "nav_state": [3, 3, 5],
                },
            ),
            SimpleNamespace(
                name="airspeed_wind",
                multi_id=0,
                data={
                    "timestamp": [1_000_000, 2_000_000],
                    "windspeed_north": [1.0, 2.0],
                },
            ),
            SimpleNamespace(
                name="airspeed_wind",
                multi_id=1,
                data={
                    "timestamp": [1_500_000, 2_500_000],
                    "windspeed_north": [3.0, 4.0],
                },
            ),
        ],
    )


def test_evidence_index_resolves_complete_exact_series_and_parameters():
    index = ULogEvidenceIndex(build_ulog())

    signal = index.resolve_signal("vehicle_status.nav_state")
    parameter = index.resolve_parameter("TEST_PARAM")

    assert signal.status == "observed"
    assert signal.series is not None
    assert [(sample.time_s, sample.value) for sample in signal.series.samples] == [
        (1.0, 3),
        (2.0, 3),
        (3.0, 5),
    ]
    assert parameter.status == "observed"
    assert parameter.value == 42


def test_evidence_index_requires_explicit_instance_when_exact_field_is_ambiguous():
    index = ULogEvidenceIndex(build_ulog())

    ambiguous = index.resolve_signal("airspeed_wind.windspeed_north")
    exact = index.resolve_signal("airspeed_wind[1].windspeed_north")

    assert ambiguous.status == "ambiguous"
    assert ambiguous.candidates == [
        "airspeed_wind[0].windspeed_north",
        "airspeed_wind[1].windspeed_north",
    ]
    assert exact.status == "observed"
    assert exact.series is not None
    assert exact.series.multi_id == 1


def test_evidence_index_supports_windows_and_deterministic_alignment():
    index = ULogEvidenceIndex(build_ulog())

    window = index.samples("vehicle_status.nav_state", start_s=1.5, end_s=2.5)
    _, prior = index.sample_at("vehicle_status.nav_state", 2.6, policy="prior")
    _, nearest = index.sample_at("vehicle_status.nav_state", 2.6, policy="nearest")

    assert window.series is not None
    assert [(sample.time_s, sample.value) for sample in window.series.samples] == [(2.0, 3)]
    assert prior is not None and (prior.time_s, prior.value) == (2.0, 3)
    assert nearest is not None and (nearest.time_s, nearest.value) == (3.0, 5)


def test_evidence_index_does_not_use_suffix_or_fuzzy_signal_matching():
    index = ULogEvidenceIndex(build_ulog())

    resolution = index.resolve_signal("status.nav_state")

    assert resolution.status == "unavailable"
    assert resolution.reason == "exact topic, instance, and field are not logged"
