from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


MINIMUM_SAMPLING_RATE_HZ = 100.0


def sampling_period_seconds(
    timestamps_us: Any,
    *,
    minimum_rate_hz: float = MINIMUM_SAMPLING_RATE_HZ,
) -> float | None:
    """Return a robust sample period for microsecond timestamps.

    Logging dropouts do not bias the result because only the median positive
    timestamp difference is used. Invalid timestamps and data sampled below
    ``minimum_rate_hz`` are rejected.
    """

    if not np.isfinite(minimum_rate_hz) or minimum_rate_hz <= 0:
        raise ValueError("minimum_rate_hz must be finite and positive")

    timestamps = _one_dimensional_float_array(timestamps_us)
    if (
        timestamps is None
        or timestamps.size < 2
        or not np.all(np.isfinite(timestamps))
    ):
        return None

    differences_us = np.diff(timestamps)
    positive_differences_us = differences_us[
        np.isfinite(differences_us) & (differences_us > 0)
    ]
    if positive_differences_us.size == 0:
        return None

    period_s = float(np.median(positive_differences_us)) * 1.0e-6
    if not np.isfinite(period_s) or period_s <= 0:
        return None

    sampling_rate_hz = 1.0 / period_s
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz < minimum_rate_hz:
        return None
    return period_s


def positive_rfft_spectrum(
    timestamps_us: Any,
    arrays: Sequence[Any],
    *,
    max_points: int = 2_400,
    minimum_rate_hz: float = MINIMUM_SAMPLING_RATE_HZ,
    summary_start_frequency_hz: float | None = None,
) -> dict[str, Any] | None:
    """Compute bounded, one-sided amplitude spectra for equally sampled data.

    All returned series share one frequency axis. Downsampling retains the
    largest peak from every input array and then chooses representative local
    maxima, so narrow peaks are not lost to fixed-stride sampling.
    """

    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    if (
        summary_start_frequency_hz is not None
        and (
            not np.isfinite(summary_start_frequency_hz)
            or summary_start_frequency_hz < 0
        )
    ):
        raise ValueError(
            "summary_start_frequency_hz must be finite and non-negative"
        )

    period_s = sampling_period_seconds(
        timestamps_us,
        minimum_rate_hz=minimum_rate_hz,
    )
    timestamps = _one_dimensional_float_array(timestamps_us)
    finite_arrays = _finite_arrays(arrays)
    if (
        period_s is None
        or timestamps is None
        or not finite_arrays
        or any(array.size != timestamps.size for array in finite_arrays)
        or timestamps.size < 2
    ):
        return None

    sample_count = timestamps.size
    frequencies_hz = np.fft.rfftfreq(sample_count, period_s)
    amplitudes: list[np.ndarray] = []
    for array in finite_arrays:
        amplitude = np.abs(np.fft.rfft(array)) / sample_count
        if sample_count % 2 == 0:
            amplitude[1:-1] *= 2.0
        else:
            amplitude[1:] *= 2.0
        amplitudes.append(amplitude)

    result = {
        "sampling_frequency_hz": 1.0 / period_s,
    }
    if summary_start_frequency_hz is not None:
        summary_mask = frequencies_hz >= summary_start_frequency_hz
        if sample_count % 2 == 0:
            summary_mask &= frequencies_hz < (0.5 / period_s)
        result["summary_start_frequency_hz"] = summary_start_frequency_hz
        result["summary_mean_amplitudes"] = [
            (
                float(np.mean(amplitude[summary_mask]))
                if np.any(summary_mask)
                else None
            )
            for amplitude in amplitudes
        ]

    selected_indices = _spectrum_indices(amplitudes, max_points)
    result["frequencies_hz"] = frequencies_hz[selected_indices]
    result["amplitudes"] = [
        amplitude[selected_indices]
        for amplitude in amplitudes
    ]
    return result


def summed_spectrogram(
    timestamps_us: Any,
    arrays: Sequence[Any],
    *,
    max_time_bins: int = 800,
    window_length: int = 256,
    noverlap: int = 128,
    minimum_rate_hz: float = MINIMUM_SAMPLING_RATE_HZ,
) -> dict[str, Any] | None:
    """Return the summed density PSD of multiple arrays in decibels.

    This mirrors the existing NumPy Hann-window implementation: each window is
    mean-centered, overlapping rFFTs are density-scaled, and component PSDs are
    summed before conversion to decibels.
    """

    if max_time_bins < 1:
        raise ValueError("max_time_bins must be positive")
    if window_length < 2:
        raise ValueError("window_length must be at least 2")
    if noverlap < 0 or noverlap >= window_length:
        raise ValueError("noverlap must be between 0 and window_length")

    period_s = sampling_period_seconds(
        timestamps_us,
        minimum_rate_hz=minimum_rate_hz,
    )
    timestamps = _one_dimensional_float_array(timestamps_us)
    finite_arrays = _finite_arrays(arrays)
    if (
        period_s is None
        or timestamps is None
        or not finite_arrays
        or any(array.size != timestamps.size for array in finite_arrays)
        or timestamps.size < window_length
    ):
        return None

    step = window_length - noverlap
    starts = np.arange(0, timestamps.size - window_length + 1, step)
    if starts.size == 0:
        return None

    frequencies_hz = np.fft.rfftfreq(window_length, period_s)
    window = np.hanning(window_length)
    window_power = float(np.sum(window * window))
    if not np.isfinite(window_power) or window_power <= 0:
        return None

    sampling_frequency_hz = 1.0 / period_s
    summed_psd = np.zeros((frequencies_hz.size, starts.size), dtype=float)
    for array in finite_arrays:
        for column, start_index in enumerate(starts):
            segment = array[start_index:start_index + window_length]
            centered_segment = segment - np.mean(segment)
            fft_values = np.fft.rfft(centered_segment * window)
            psd = (np.abs(fft_values) ** 2) / (
                sampling_frequency_hz * window_power
            )
            if psd.size > 2:
                psd[1:-1] *= 2.0
            summed_psd[:, column] += psd

    summed_psd = np.maximum(summed_psd, np.finfo(float).tiny)
    values_db = 10.0 * np.log10(summed_psd)
    center_indices = np.minimum(
        starts + window_length // 2,
        timestamps.size - 1,
    )
    absolute_time_s = timestamps[center_indices] * 1.0e-6

    if starts.size > max_time_bins:
        selected_columns = np.unique(
            np.linspace(0, starts.size - 1, max_time_bins, dtype=int)
        )
        absolute_time_s = absolute_time_s[selected_columns]
        values_db = values_db[:, selected_columns]

    return {
        "sampling_frequency_hz": sampling_frequency_hz,
        "frequencies_hz": frequencies_hz,
        "time_s": absolute_time_s,
        "values_db": values_db,
    }


def expand_fifo_samples(data: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    """Expand packed FIFO x[N]/y[N]/z[N] fields without mutating ``data``.

    Each packet's ``timestamp_sample`` identifies its newest sample. Earlier
    samples are placed at ``dt`` microsecond intervals and every raw axis value
    is multiplied by that packet's scale, including negative values.
    """

    required_metadata = ("timestamp_sample", "dt", "samples", "scale")
    if any(key not in data for key in required_metadata):
        return None

    timestamps = _one_dimensional_float_array(data["timestamp_sample"])
    delta_us = _one_dimensional_float_array(data["dt"])
    sample_counts = _one_dimensional_float_array(data["samples"])
    scales = _one_dimensional_float_array(data["scale"])
    if any(
        array is None
        for array in (timestamps, delta_us, sample_counts, scales)
    ):
        return None

    metadata_arrays = (timestamps, delta_us, sample_counts, scales)
    packet_count = timestamps.size
    if packet_count == 0 or any(array.size != packet_count for array in metadata_arrays):
        return None
    if (
        not np.all(np.isfinite(timestamps))
        or not np.all(np.isfinite(delta_us))
        or not np.all(np.isfinite(sample_counts))
        or not np.all(np.isfinite(scales))
        or np.any(delta_us < 0)
        or np.any(sample_counts < 0)
        or np.any(sample_counts != np.floor(sample_counts))
    ):
        return None

    counts = sample_counts.astype(np.int64)
    available_indices = _fifo_axis_indices(data)
    if available_indices is None:
        return None
    required_indices = set(range(int(np.max(counts, initial=0))))
    if not required_indices.issubset(available_indices):
        return None

    axis_fields: dict[str, dict[int, np.ndarray]] = {}
    for axis in ("x", "y", "z"):
        axis_fields[axis] = {}
        for sample_index in required_indices:
            values = _one_dimensional_float_array(data[f"{axis}[{sample_index}]"])
            if (
                values is None
                or values.size != packet_count
                or not np.all(np.isfinite(values))
            ):
                return None
            axis_fields[axis][sample_index] = values

    total_samples = int(np.sum(counts))
    expanded_timestamps = np.empty(total_samples, dtype=float)
    expanded_axes = {
        axis: np.empty(total_samples, dtype=float)
        for axis in ("x", "y", "z")
    }

    output_index = 0
    for packet_index, count in enumerate(counts):
        count_int = int(count)
        for sample_index in range(count_int):
            expanded_timestamps[output_index] = (
                timestamps[packet_index]
                - (count_int - sample_index - 1) * delta_us[packet_index]
            )
            for axis in ("x", "y", "z"):
                expanded_axes[axis][output_index] = (
                    axis_fields[axis][sample_index][packet_index]
                    * scales[packet_index]
                )
            output_index += 1

    return {
        "timestamp": expanded_timestamps.copy(),
        "timestamp_sample": expanded_timestamps.copy(),
        **expanded_axes,
    }


def _one_dimensional_float_array(values: Any) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1:
        return None
    return array


def _finite_arrays(arrays: Sequence[Any]) -> list[np.ndarray] | None:
    converted: list[np.ndarray] = []
    try:
        iterator = iter(arrays)
    except TypeError:
        return None
    for values in iterator:
        array = _one_dimensional_float_array(values)
        if array is None or not np.all(np.isfinite(array)):
            return None
        converted.append(array)
    return converted or None


def _spectrum_indices(amplitudes: Sequence[np.ndarray], max_points: int) -> np.ndarray:
    point_count = amplitudes[0].size
    if point_count <= max_points:
        return np.arange(point_count)

    mandatory_indices = {0, point_count - 1}
    mandatory_indices.update(int(np.argmax(amplitude)) for amplitude in amplitudes)
    if len(mandatory_indices) >= max_points:
        ranked_peaks = sorted(
            (
                (float(amplitude[index]), index)
                for amplitude in amplitudes
                for index in [int(np.argmax(amplitude))]
            ),
            reverse=True,
        )
        selected = {0, point_count - 1}
        for _, index in ranked_peaks:
            if len(selected) >= max_points:
                break
            selected.add(index)
        return np.asarray(sorted(selected), dtype=int)

    normalized = []
    for amplitude in amplitudes:
        maximum = float(np.max(amplitude))
        normalized.append(amplitude / maximum if maximum > 0 else amplitude)
    peak_score = np.max(np.vstack(normalized), axis=0)

    remaining = max_points - len(mandatory_indices)
    candidates: list[int] = []
    for bucket in np.array_split(np.arange(point_count), remaining):
        if bucket.size:
            candidates.append(int(bucket[int(np.argmax(peak_score[bucket]))]))

    selected = set(mandatory_indices)
    for index in candidates:
        if len(selected) >= max_points:
            break
        selected.add(index)

    if len(selected) < max_points:
        for index in np.linspace(0, point_count - 1, max_points, dtype=int):
            selected.add(int(index))
            if len(selected) >= max_points:
                break
    return np.asarray(sorted(selected), dtype=int)


def _fifo_axis_indices(data: Mapping[str, Any]) -> set[int] | None:
    indices_by_axis: dict[str, set[int]] = {
        axis: set()
        for axis in ("x", "y", "z")
    }
    pattern = re.compile(r"^([xyz])\[(\d+)\]$")
    for key in data:
        match = pattern.fullmatch(str(key))
        if match is not None:
            indices_by_axis[match.group(1)].add(int(match.group(2)))

    common_indices = set.intersection(*indices_by_axis.values())
    return common_indices or None
