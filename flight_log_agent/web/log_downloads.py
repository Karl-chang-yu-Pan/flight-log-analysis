from __future__ import annotations

import math
import numbers
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pyulog import ULog

from flight_log_agent.ulog.preparse_view import build_parameter_rows


KML_NAMESPACE = "http://www.opengis.net/kml/2.2"


def build_parameter_download(log_path: Path, *, non_default_only: bool) -> bytes:
    ulog = ULog(str(log_path), None, disable_str_exceptions=True)
    rows = build_parameter_rows(ulog)
    return serialize_parameter_rows(rows, non_default_only=non_default_only)


def serialize_parameter_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    non_default_only: bool,
) -> bytes:
    lines = []
    for row in rows:
        if non_default_only and row.get("default_status") != "non_default":
            continue
        name = _single_line(row.get("name"))
        if not name:
            continue
        value = row.get("raw_value")
        mav_param_type = 6 if isinstance(value, numbers.Integral) else 9
        lines.append(f"1\t1\t{name}\t{_single_line(value)}\t{mav_param_type}\n")
    return "".join(lines).encode("utf-8")


def build_kml_download(log_path: Path) -> bytes:
    ulog = ULog(
        str(log_path),
        ["vehicle_gps_position"],
        disable_str_exceptions=True,
    )
    try:
        dataset = ulog.get_dataset("vehicle_gps_position")
    except Exception as exc:
        raise ValueError("log has no vehicle_gps_position data") from exc
    coordinates = extract_gps_coordinates(getattr(dataset, "data", {}) or {})
    return serialize_kml_track(coordinates)


def extract_gps_coordinates(data: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    if all(name in data for name in ("latitude_deg", "longitude_deg", "altitude_msl_m")):
        latitudes = data["latitude_deg"]
        longitudes = data["longitude_deg"]
        altitudes = data["altitude_msl_m"]
        latitude_scale = longitude_scale = altitude_scale = 1.0
    elif all(name in data for name in ("lat", "lon", "alt")):
        latitudes = data["lat"]
        longitudes = data["lon"]
        altitudes = data["alt"]
        latitude_scale = longitude_scale = 1e-7
        altitude_scale = 1e-3
    else:
        raise ValueError("GPS data does not contain latitude, longitude, and altitude")

    fix_types = data.get("fix_type")
    count = min(len(latitudes), len(longitudes), len(altitudes))
    if fix_types is not None:
        count = min(count, len(fix_types))

    coordinates = []
    for index in range(count):
        if fix_types is not None and _float(fix_types[index]) <= 2:
            continue
        latitude = _float(latitudes[index]) * latitude_scale
        longitude = _float(longitudes[index]) * longitude_scale
        altitude = _float(altitudes[index]) * altitude_scale
        if not all(math.isfinite(value) for value in (latitude, longitude, altitude)):
            continue
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            continue
        coordinates.append((longitude, latitude, altitude))

    if not coordinates:
        raise ValueError("log has no valid GPS position data")
    return coordinates


def serialize_kml_track(coordinates: Iterable[tuple[float, float, float]]) -> bytes:
    points = list(coordinates)
    if not points:
        raise ValueError("KML track requires at least one coordinate")

    ET.register_namespace("", KML_NAMESPACE)
    kml = ET.Element(ET.QName(KML_NAMESPACE, "kml"))
    document = ET.SubElement(kml, ET.QName(KML_NAMESPACE, "Document"))
    ET.SubElement(document, ET.QName(KML_NAMESPACE, "name")).text = "Flight track"
    placemark = ET.SubElement(document, ET.QName(KML_NAMESPACE, "Placemark"))
    ET.SubElement(placemark, ET.QName(KML_NAMESPACE, "name")).text = "Vehicle path"
    line_string = ET.SubElement(placemark, ET.QName(KML_NAMESPACE, "LineString"))
    ET.SubElement(line_string, ET.QName(KML_NAMESPACE, "tessellate")).text = "1"
    ET.SubElement(line_string, ET.QName(KML_NAMESPACE, "altitudeMode")).text = "absolute"
    ET.SubElement(line_string, ET.QName(KML_NAMESPACE, "coordinates")).text = "\n".join(
        f"{longitude:.9f},{latitude:.9f},{altitude:.3f}"
        for longitude, latitude, altitude in points
    )
    return ET.tostring(kml, encoding="utf-8", xml_declaration=True)


def _single_line(value: Any) -> str:
    return (
        str(value if value is not None else "")
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("GPS data contains a non-numeric value") from exc
