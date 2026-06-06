"""Convert EEGLAB channel locations between coordinate systems."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def convertlocs(chans: Any, command: str = "auto", *args: Any, verbose: str = "off", **kwargs: Any) -> Any:
    """Return channel locations with requested coordinate fields filled in.

    Args:
        chans: EEG dictionary, channel-location dictionary, or channel-location
            sequence.
        command: EEGLAB conversion command such as ``"cart2all"``,
            ``"sph2topo"``, ``"topo2cart"``, or ``"auto"``.
        *args: Optional EEGLAB-style key/value arguments. Only ``verbose`` is
            accepted for parity with EEGLAB.
        verbose: ``"on"`` or ``"off"``. Present for API parity.
        **kwargs: Optional keyword arguments. Only ``verbose`` is accepted.

    Returns:
        A deep-copied object in the same broad shape as ``chans``.
    """
    if args:
        if len(args) % 2:
            raise ValueError("convertlocs optional arguments must be key/value pairs")
        for index in range(0, len(args), 2):
            if str(args[index]).lower() != "verbose":
                raise ValueError(f"Unsupported convertlocs option: {args[index]}")
            verbose = args[index + 1]
    if kwargs:
        unknown = set(kwargs) - {"verbose"}
        if unknown:
            raise ValueError(f"Unsupported convertlocs option: {sorted(unknown)[0]}")
        verbose = kwargs.get("verbose", verbose)
    del verbose

    is_eeg = isinstance(chans, dict) and "chanlocs" in chans
    output = copy.deepcopy(chans)
    locs = chanlocs_as_list(output.get("chanlocs") if is_eeg else output)
    mode = _normalize_command(command, locs)
    _apply_conversion(locs, mode)
    if is_eeg:
        output["chanlocs"] = locs
        return output
    if isinstance(chans, dict):
        return locs[0] if locs else {}
    return locs


def _normalize_command(command: str, locs: list[dict[str, Any]]) -> str:
    mode = str(command or "auto").lower().replace("->", "2").replace("xyz", "cart").replace("polar", "topo")
    if mode != "auto":
        return mode
    if any(_has_coordinate(loc, "X", "Y", "Z") for loc in locs):
        return "cart2all"
    if any(_has_coordinate(loc, "sph_theta", "sph_phi") for loc in locs):
        return "sph2all"
    if any(_has_coordinate(loc, "sph_theta_besa", "sph_phi_besa") for loc in locs):
        return "sphbesa2all"
    return "topo2all"


def _apply_conversion(locs: list[dict[str, Any]], mode: str) -> None:
    if mode == "cart2all":
        _apply_conversion(locs, "cart2sph")
        _apply_conversion(locs, "sph2topo")
        _apply_conversion(locs, "sph2sphbesa")
        return
    if mode == "sph2all":
        _apply_conversion(locs, "sph2topo")
        _apply_conversion(locs, "sph2sphbesa")
        _apply_conversion(locs, "sph2cart")
        return
    if mode == "topo2all":
        _apply_conversion(locs, "topo2sph")
        _apply_conversion(locs, "sph2all")
        return
    if mode == "sphbesa2all":
        _apply_conversion(locs, "sphbesa2sph")
        _apply_conversion(locs, "sph2all")
        return
    if mode == "cart2topo":
        _apply_conversion(locs, "cart2sph")
        _apply_conversion(locs, "sph2topo")
        return
    if mode == "cart2sphbesa":
        _apply_conversion(locs, "cart2sph")
        _apply_conversion(locs, "sph2sphbesa")
        return
    if mode == "topo2cart":
        _apply_conversion(locs, "topo2sph")
        _apply_conversion(locs, "sph2cart")
        return
    if mode == "topo2sphbesa":
        _apply_conversion(locs, "topo2sph")
        _apply_conversion(locs, "sph2sphbesa")
        return
    if mode == "sphbesa2topo":
        _apply_conversion(locs, "sphbesa2sph")
        _apply_conversion(locs, "sph2topo")
        return
    if mode == "sphbesa2cart":
        _apply_conversion(locs, "sphbesa2sph")
        _apply_conversion(locs, "sph2cart")
        return

    for loc in locs:
        if mode == "cart2sph":
            _cart_to_sph(loc)
        elif mode == "sph2cart":
            _sph_to_cart(loc)
        elif mode == "sph2topo":
            _sph_to_topo(loc)
        elif mode == "topo2sph":
            _topo_to_sph(loc)
        elif mode == "sphbesa2sph":
            _sph_besa_to_sph(loc)
        elif mode == "sph2sphbesa":
            _sph_to_sph_besa(loc)
        else:
            raise ValueError(f"Unsupported channel-location conversion: {mode}")


def _cart_to_sph(loc: dict[str, Any]) -> None:
    if not _has_coordinate(loc, "X", "Y", "Z"):
        return
    x = float(loc["X"])
    y = float(loc["Y"])
    z = float(loc["Z"])
    hypot_xy = float(np.hypot(x, y))
    loc["sph_theta"] = _wrap_degrees(np.degrees(np.arctan2(y, x)))
    loc["sph_phi"] = float(np.degrees(np.arctan2(z, hypot_xy)))
    loc["sph_radius"] = float(np.sqrt(x * x + y * y + z * z))


def _sph_to_cart(loc: dict[str, Any]) -> None:
    if not _has_coordinate(loc, "sph_theta", "sph_phi"):
        return
    theta = np.radians(float(loc["sph_theta"]))
    phi = np.radians(float(loc["sph_phi"]))
    radius = float(loc.get("sph_radius", 1.0) or 1.0)
    loc["X"] = float(radius * np.cos(phi) * np.cos(theta))
    loc["Y"] = float(radius * np.cos(phi) * np.sin(theta))
    loc["Z"] = float(radius * np.sin(phi))


def _sph_to_topo(loc: dict[str, Any]) -> None:
    if not _has_coordinate(loc, "sph_theta", "sph_phi"):
        return
    loc["theta"] = _wrap_degrees(-float(loc["sph_theta"]))
    loc["radius"] = float(0.5 - float(loc["sph_phi"]) / 180.0)
    loc.setdefault("sph_radius", 1.0)


def _topo_to_sph(loc: dict[str, Any]) -> None:
    if not _has_coordinate(loc, "theta", "radius"):
        return
    loc["sph_theta"] = _wrap_degrees(-float(loc["theta"]))
    loc["sph_phi"] = float((0.5 - float(loc["radius"])) * 180.0)
    loc.setdefault("sph_radius", 1.0)


def _sph_besa_to_sph(loc: dict[str, Any]) -> None:
    if not _has_coordinate(loc, "sph_theta_besa", "sph_phi_besa"):
        return
    theta_besa = float(loc["sph_theta_besa"])
    phi_besa = float(loc["sph_phi_besa"])
    angle = 90.0 - phi_besa if theta_besa >= 0 else -90.0 - phi_besa
    loc["sph_theta"] = _wrap_degrees(-angle)
    loc["sph_phi"] = 90.0 - abs(theta_besa)
    loc.setdefault("sph_radius", 1.0)


def _sph_to_sph_besa(loc: dict[str, Any]) -> None:
    if not _has_coordinate(loc, "sph_theta", "sph_phi"):
        return
    theta = float(loc["sph_theta"])
    phi = float(loc["sph_phi"])
    radius = 0.5 - phi / 180.0
    angle = -theta
    loc["sph_theta_besa"] = float(180.0 * radius if angle >= 0 else -180.0 * radius)
    loc["sph_phi_besa"] = float(90.0 - angle if angle >= 0 else -90.0 - angle)


def _has_coordinate(loc: dict[str, Any], *fields: str) -> bool:
    return all(not _is_empty(loc.get(field)) and np.isfinite(float(loc[field])) for field in fields)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, (list, tuple, dict)) and len(value) == 0


def _wrap_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    if np.isclose(wrapped, -180.0) and value > 0:
        return 180.0
    return float(wrapped)


__all__ = ["convertlocs"]
