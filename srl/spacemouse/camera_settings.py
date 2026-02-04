import json
import os
import carb

SETTINGS_FILENAME = "camera_settings.json"
PREFERRED_CAMERA_ATTRS = [
    "focalLength",
    "horizontalAperture",
    "verticalAperture",
    "clippingRange",
    "exposure:iso",
    "exposure:shutter",
    "exposure:fStop",
    "exposure:aperture",
    "exposure:focusDistance",
]


def _get_camera_settings_path() -> str:
    tokens = carb.tokens.get_tokens_interface()
    user_config_dir = tokens.resolve("${user_config_dir}")
    return os.path.join(user_config_dir, "srl.spacemouse", SETTINGS_FILENAME)


def _load_camera_settings(path: str) -> dict:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except Exception:
        pass
    return {}


def _save_camera_settings(path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
    except Exception:
        pass


def _is_json_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_json_value(value):
    if _is_json_number(value) or isinstance(value, bool):
        return value
    if isinstance(value, (tuple, list)):
        if all(_is_json_number(v) or isinstance(v, bool) for v in value):
            return list(value)
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            seq = list(value)
            if all(_is_json_number(v) or isinstance(v, bool) for v in seq):
                return seq
        except Exception:
            pass
    return None


def apply_saved_camera_settings(cam_prim):
    path = _get_camera_settings_path()
    saved = _load_camera_settings(path)
    if not isinstance(saved, dict):
        return
    attrs = {attr.GetName(): attr for attr in cam_prim.GetAttributes()}
    for name, value in saved.items():
        if name.startswith("xformOp:") or name == "xformOpOrder":
            continue
        attr = attrs.get(name)
        if attr is None:
            continue
        if isinstance(value, list):
            attr.Set(tuple(value))
        else:
            attr.Set(value)


def get_camera_settings_snapshot(cam_prim) -> dict:
    saved = {}
    for name in PREFERRED_CAMERA_ATTRS:
        attr = cam_prim.GetAttribute(name)
        if not attr:
            continue
        value = attr.Get()
        json_value = _to_json_value(value)
        if json_value is not None:
            saved[name] = json_value
    return saved


def save_current_camera_settings(cam_prim):
    path = _get_camera_settings_path()
    _save_camera_settings(path, get_camera_settings_snapshot(cam_prim))


def save_camera_settings_if_changed(cam_prim, last_snapshot: dict) -> dict:
    current = get_camera_settings_snapshot(cam_prim)
    if current != last_snapshot:
        path = _get_camera_settings_path()
        _save_camera_settings(path, current)
        return current
    return last_snapshot


def clear_camera_settings():
    path = _get_camera_settings_path()
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass
