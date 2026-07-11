from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_AIRFRAME_IMAGE_KEY = "AirframeUnknown"
AirframeMetadata = dict[str, dict[str, str]]


def load_airframe_metadata(paths: Iterable[Path]) -> AirframeMetadata:
    """Merge generated PX4 airframe metadata in low-to-high priority order."""
    airframes: AirframeMetadata = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue

        for group in root.findall(".//airframe_group"):
            group_name = _clean(group.get("name") or group.get("type") or group.get("id"))
            image_key = _clean_image_key(group.get("image"))
            for airframe in group.findall("airframe"):
                airframe_id = normalize_airframe_id(airframe.get("id"))
                if not airframe_id:
                    continue

                values = {
                    "name": _clean(airframe.get("name")),
                    "group": group_name,
                    "image": image_key,
                }
                type_node = airframe.find("type")
                if type_node is not None:
                    values["type"] = _clean(type_node.text)

                entry = airframes.setdefault(airframe_id, {})
                entry.update({key: value for key, value in values.items() if value})

    return airframes


def metadata_for_airframe(
    airframe_id: Any,
    airframes: Mapping[str, Mapping[str, str]],
) -> Mapping[str, str]:
    normalized_id = normalize_airframe_id(airframe_id)
    return airframes.get(normalized_id, {}) if normalized_id else {}


def resolve_airframe_image_key(
    airframe_id: Any,
    airframes: Mapping[str, Mapping[str, str]],
    image_root: Path | None,
) -> str:
    metadata = metadata_for_airframe(airframe_id, airframes)
    image_key = _clean_image_key(metadata.get("image"))
    if image_key and _image_asset_exists(image_root, image_key):
        return image_key
    return DEFAULT_AIRFRAME_IMAGE_KEY


def normalize_airframe_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().rstrip("\x00")


def _image_asset_exists(image_root: Path | None, image_key: str) -> bool:
    if image_root is None:
        return False
    root = image_root.resolve()
    candidate = (root / f"{image_key}.svg").resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate.is_file()


def _clean_image_key(value: Any) -> str:
    image_key = _clean(value)
    if image_key.lower().endswith(".svg"):
        image_key = image_key[:-4]
    if not image_key or Path(image_key).name != image_key or image_key in {".", ".."}:
        return ""
    return image_key


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().rstrip("\x00")
