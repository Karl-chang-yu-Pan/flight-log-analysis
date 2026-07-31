from __future__ import annotations

import copy

import numpy as np
import pytest

from flight_log_agent.ulog.interactive_plot_spectra import (
    expand_fifo_samples,
    positive_rfft_spectrum,
    sampling_period_seconds,
    summed_spectrogram,
)


def test_sampling_period_uses_median_positive_microsecond_difference():
    timestamps = np.asarray([0, 1_000, 2_000, 2_000, 1_002_000], dtype=float)

    assert sampling_period_seconds(timestamps) == pytest.approx(0.001)
    assert sampling_period_seconds([0, 11_000, 22_000]) is None
    assert sampling_period_seconds([1_000, 1_000]) is None
    assert sampling_period_seconds([0, np.nan, 1_000]) is None
    assert sampling_period_seconds([0, 1_000, np.nan, 2_000, 3_000]) is None


def test_positive_rfft_is_bounded_and_retains_each_input_peak():
    sample_count = 4_096
    timestamps = np.arange(sample_count, dtype=float) * 1_000.0
    time_s = timestamps * 1.0e-6
    first = 2.0 * np.sin(2.0 * np.pi * 50.0 * time_s)
    second = -0.75 * np.sin(2.0 * np.pi * 125.0 * time_s)
    original_second = second.copy()

    result = positive_rfft_spectrum(
        timestamps,
        [first, second],
        max_points=64,
        summary_start_frequency_hz=40.0,
    )

    assert result is not None
    assert result["sampling_frequency_hz"] == pytest.approx(1_000.0)
    assert len(result["frequencies_hz"]) <= 64
    assert len(result["amplitudes"]) == 2
    assert all(
        len(amplitude) == len(result["frequencies_hz"])
        for amplitude in result["amplitudes"]
    )
    peak_frequencies = [
        result["frequencies_hz"][int(np.argmax(amplitude))]
        for amplitude in result["amplitudes"]
    ]
    assert peak_frequencies[0] == pytest.approx(50.0, abs=0.25)
    assert peak_frequencies[1] == pytest.approx(125.0, abs=0.25)
    full_frequencies = np.fft.rfftfreq(sample_count, 0.001)
    summary_mask = (
        (full_frequencies >= 40.0)
        & (full_frequencies < 500.0)
    )
    expected_first = np.abs(np.fft.rfft(first)) / sample_count
    expected_first[1:-1] *= 2.0
    assert result["summary_start_frequency_hz"] == 40.0
    assert result["summary_mean_amplitudes"][0] == pytest.approx(
        np.mean(expected_first[summary_mask])
    )
    assert np.array_equal(second, original_second)


def test_spectral_helpers_reject_invalid_or_slow_input():
    timestamps = np.arange(512, dtype=float) * 20_000.0
    values = np.zeros(512)

    assert positive_rfft_spectrum(timestamps, [values]) is None
    assert summed_spectrogram(timestamps, [values]) is None

    fast_timestamps = np.arange(512, dtype=float) * 1_000.0
    values[20] = np.nan
    assert positive_rfft_spectrum(fast_timestamps, [values]) is None
    assert summed_spectrogram(fast_timestamps, [values]) is None


def test_summed_spectrogram_matches_hann_density_and_bounds_time_bins():
    sample_count = 1_024
    timestamps = 5_000_000.0 + np.arange(sample_count, dtype=float) * 1_000.0
    time_s = np.arange(sample_count, dtype=float) * 0.001
    first = np.sin(2.0 * np.pi * 50.0 * time_s)
    second = 0.5 * np.sin(2.0 * np.pi * 120.0 * time_s)

    result = summed_spectrogram(
        timestamps,
        [first, second],
        max_time_bins=3,
    )

    assert result is not None
    assert result["sampling_frequency_hz"] == pytest.approx(1_000.0)
    assert result["values_db"].shape == (129, 3)
    assert len(result["time_s"]) == 3
    assert result["time_s"][0] == pytest.approx(5.128)

    window = np.hanning(256)
    window_power = np.sum(window * window)
    expected_psd = np.zeros(129)
    for values in (first, second):
        segment = values[:256] - np.mean(values[:256])
        component = np.abs(np.fft.rfft(segment * window)) ** 2
        component /= 1_000.0 * window_power
        component[1:-1] *= 2.0
        expected_psd += component
    expected_db = 10.0 * np.log10(
        np.maximum(expected_psd, np.finfo(float).tiny)
    )
    assert result["values_db"][:, 0] == pytest.approx(expected_db)


def test_summed_spectrogram_uses_logged_window_centers_after_a_dropout():
    timestamps = np.arange(512, dtype=float) * 1_000.0
    timestamps[256:] += 1_000_000.0
    values = np.sin(2.0 * np.pi * 50.0 * np.arange(512) * 0.001)

    result = summed_spectrogram(
        timestamps,
        [values],
        window_length=256,
        noverlap=0,
    )

    assert result is not None
    assert result["time_s"] == pytest.approx(
        [
            timestamps[128] * 1.0e-6,
            timestamps[384] * 1.0e-6,
        ]
    )


def test_expand_fifo_samples_discovers_fields_scales_and_preserves_signs():
    data = {
        "timestamp_sample": np.asarray([100.0, 300.0]),
        "dt": np.asarray([10.0, 20.0]),
        "samples": np.asarray([2, 3]),
        "scale": np.asarray([0.5, -2.0]),
        "x[0]": np.asarray([-2.0, 1.0]),
        "x[1]": np.asarray([-4.0, 2.0]),
        "x[2]": np.asarray([999.0, 3.0]),
        "y[0]": np.asarray([2.0, -1.0]),
        "y[1]": np.asarray([4.0, -2.0]),
        "y[2]": np.asarray([999.0, -3.0]),
        "z[0]": np.asarray([-6.0, 4.0]),
        "z[1]": np.asarray([-8.0, 5.0]),
        "z[2]": np.asarray([999.0, 6.0]),
    }
    original = copy.deepcopy(data)

    expanded = expand_fifo_samples(data)

    assert expanded is not None
    assert expanded["timestamp"] == pytest.approx([90, 100, 260, 280, 300])
    assert expanded["timestamp_sample"] == pytest.approx(
        [90, 100, 260, 280, 300]
    )
    assert expanded["x"] == pytest.approx([-1, -2, -2, -4, -6])
    assert expanded["y"] == pytest.approx([1, 2, 2, 4, 6])
    assert expanded["z"] == pytest.approx([-3, -4, -8, -10, -12])
    assert data.keys() == original.keys()
    for key in data:
        assert np.array_equal(data[key], original[key])


def test_expand_fifo_samples_rejects_missing_actual_axis_field():
    data = {
        "timestamp_sample": [100],
        "dt": [10],
        "samples": [2],
        "scale": [1],
        "x[0]": [1],
        "x[1]": [2],
        "y[0]": [3],
        "y[1]": [4],
        "z[0]": [5],
    }

    assert expand_fifo_samples(data) is None
