"""Apply packaged EGI montage locations to an EEG dataset."""

from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from typing import Any

from eegprep.functions.sigprocfunc.readlocs import readlocs


EGI_MONTAGE_BY_CHANNELS = {
    32: "egi-gsn-hydrocell-32.sfp",
    33: "egi-gsn-hydrocell-32.sfp",
    64: "egi-gsn-65-v2.sfp",
    65: "egi-gsn-65-v2.sfp",
    128: "egi-gsn-hydrocell-129.locs",
    129: "egi-gsn-hydrocell-129.locs",
    256: "egi-gsn-hydrocell-257.locs",
    257: "egi-gsn-hydrocell-257.locs",
}


def readegilocs(EEG: dict[str, Any], fileloc: str | None = None) -> dict[str, Any]:
    """Return ``EEG`` with EGI channel locations from packaged montages."""
    output = deepcopy(EEG)
    nbchan = int(output.get("nbchan", 0))
    resource = fileloc or EGI_MONTAGE_BY_CHANNELS.get(nbchan)
    if not resource:
        return output
    path = _montage_path(resource)
    locs = readlocs(path)
    chaninfo: dict[str, Any] = {"filename": str(path)}
    if nbchan == 256:
        chaninfo["nodatchans"] = locs[-1:]
        locs = locs[:-1]
    elif nbchan == 257:
        chaninfo["nodatchans"] = []
    elif nbchan in {32, 64, 128}:
        chaninfo["nodatchans"] = locs[:3] + locs[-1:]
        locs = locs[3:-1]
    elif nbchan in {33, 65, 129}:
        chaninfo["nodatchans"] = locs[:3]
        locs = locs[3:]
    output["chanlocs"] = locs[:nbchan]
    output["urchanlocs"] = deepcopy(output["chanlocs"])
    output["chaninfo"] = chaninfo
    return output


def _montage_path(fileloc: str):
    path = files("eegprep").joinpath("resources").joinpath("montages").joinpath(fileloc)
    if path.is_file():
        return path
    return fileloc


__all__ = ["readegilocs"]
