"""matlab.engine stand-in talking to tests/matlab/eegprep_batch_engine_serve.m through files.

The MATLAB batch licensing token licenses the matlab-batch launcher only, never
the Engine API, so matlab.engine.start_matlab() cannot work in CI. install()
registers a fake ``matlab.engine`` module whose start_matlab() returns a
BatchEngine: the tests and eeglabcompat.get_eeglab() keep importing
matlab.engine as before and never learn the difference.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import sys
import time
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import scipy.io

# The slowest reference call in CI (std_precomp) is about 6 minutes. A wedged
# eval, such as a menu dialog, never writes a response; leave headroom for that
# call, then fail and let restart.sh replace the server so later tests are not
# stuck behind it.
TIMEOUT = 900
RESTART_READY_TIMEOUT = 300


class MatlabExecutionError(Exception):
    pass


def _bash_executable() -> str:
    """Return a bash that can run ``restart.sh`` inside the batch directory.

    Windows puts ``System32\\bash.exe`` on ``PATH``. That file is the WSL
    launcher: it does not execute a Windows ``restart.sh`` against the batch
    directory, so the restart marker never appears. Git Bash does.
    """
    if os.name != "nt":
        return "bash"
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    for candidate in (
        Path(program_files) / "Git" / "bin" / "bash.exe",
        Path(program_files) / "Git" / "usr" / "bin" / "bash.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("bash")
    if found is not None and "system32" not in Path(found).as_posix().lower():
        return found
    raise FileNotFoundError(
        "restart.sh needs Git Bash; Windows System32 bash.exe is the WSL launcher and cannot run it"
    )


def _tag(tag, value=0.0):
    return {"eegprep_rpc_tag__": tag, "value": value}


def _encode(value):
    """Python argument -> what savemat writes as MATLAB expects (ints as double, None as [])."""
    if value is None:
        return _tag("empty")
    if isinstance(value, bool):
        return _tag("logical", float(value))
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()} if value else _tag("empty")
    if isinstance(value, (list, tuple)):
        if not value:
            return _tag("cell", np.empty((1, 0), dtype=object))
        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in value):
            return np.asarray(value, dtype=float).reshape(1, -1)
        cells = np.empty((1, len(value)), dtype=object)
        for i, v in enumerate(value):
            cells[0, i] = _encode(v)
        return _tag("cell", cells)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            cells = np.empty(value.shape, dtype=object)
            for i in np.ndindex(value.shape):
                cells[i] = _encode(value[i])
            return _tag("cell", cells)
        if value.dtype == bool:
            return _tag("logical", value.astype(float))
    return value


def _decode(value):
    """loadmat result -> what matlab.engine would have returned."""
    if not isinstance(value, np.ndarray):
        return value
    if value.dtype.names:
        if value.shape == (1, 1):
            return {n: _decode(value[n][0, 0]) for n in value.dtype.names}
        out = np.empty(value.shape, dtype=object)
        for i in np.ndindex(value.shape):
            out[i] = {n: _decode(value[n][i]) for n in value.dtype.names}
        return out.tolist()
    if value.dtype.kind in "US":
        return "" if value.size == 0 else str(value.item()) if value.size == 1 else [str(v) for v in value.ravel()]
    if value.dtype == object:
        if value.size == 0:
            return []
        flat = [_decode(v) for v in value.ravel(order="F")]
        if 1 in value.shape[:2]:
            return flat
        return np.array(flat, dtype=object).reshape(value.shape, order="F").tolist()
    if value.size == 1:
        scalar = value.item()
        return bool(scalar) if value.dtype == bool else int(scalar) if value.dtype.kind in "iu" else scalar
    return value


class _Workspace:
    def __init__(self, engine):
        self._engine = engine

    def __getitem__(self, name):
        return self._engine.eval(name, nargout=1)

    def __setitem__(self, name, value):
        self._engine.assignin("base", name, value, nargout=0)


class BatchEngine:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._counter = itertools.count(1)
        self.workspace = _Workspace(self)

    def eval(self, statement, nargout=0, **_):  # noqa: A003
        return self._request("eval", statement, [], nargout)

    def quit(self):
        """The served session keeps running for the next client."""

    exit = quit

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args, nargout=1, **kwargs):
            extra = [item for k, v in kwargs.items() if k not in ("background", "stdout", "stderr") for item in (k, v)]
            return self._request("call", name, [*args, *extra], nargout)

        return method

    def _request(self, kind, name, args, nargout):
        stem = f"{os.getpid()}_{next(self._counter):06d}"
        request, partial = self.directory / f"req_{stem}.mat", self.directory / f"req_{stem}.part"
        response = self.directory / f"res_{stem}.mat"
        cells = np.empty((1, len(args)), dtype=object)
        for i, v in enumerate(args):
            cells[0, i] = _encode(v)
        scipy.io.savemat(partial, {"kind": kind, "name": name, "nargout": float(nargout), "args": cells}, format="5")
        os.replace(partial, request)  # the server only sees complete files
        try:
            self._wait_for_response(response, kind, name)
        except Exception:
            # The server is still inside this eval. Drop it before the next test.
            self._restart_if_wedged()
            raise
        try:
            raw = scipy.io.loadmat(response, squeeze_me=False)
        finally:
            response.unlink(missing_ok=True)
        if "error" in raw:
            err = raw["error"]
            raise MatlabExecutionError(f"{_decode(err['message'][0, 0])}\n{_decode(err['stack'][0, 0])}")
        outputs = raw.get("outputs")
        if outputs is None or outputs.size == 0 or nargout == 0:
            return None
        decoded = [_decode(v) for v in outputs.ravel(order="F")]
        return decoded[0] if nargout == 1 else tuple(decoded[:nargout])

    def _wait_for_response(self, response, kind, name):
        deadline = time.monotonic() + TIMEOUT
        while not response.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"MATLAB did not answer {kind} {name!r} within {TIMEOUT}s")
            time.sleep(0.05)

    def _restart_if_wedged(self):
        script = self.directory / "restart.sh"
        if not script.is_file():
            return
        ready = self.directory / "ready"
        ready.unlink(missing_ok=True)
        # check=True: a restart that did not run must not look like a timeout.
        # as_posix() keeps Git Bash from treating backslashes in the script path as escapes.
        subprocess.run(
            [_bash_executable(), script.as_posix()],
            check=True,
            cwd=self.directory,
            timeout=120,
        )
        deadline = time.monotonic() + RESTART_READY_TIMEOUT
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)


def install(directory):
    """Register fake ``matlab`` / ``matlab.engine`` modules backed by the served directory."""
    engine = types.ModuleType("matlab.engine")
    engine.__spec__ = ModuleSpec("matlab.engine", None)  # find_spec("matlab.engine") must not raise
    engine.start_matlab = lambda *_, **__: BatchEngine(directory)
    engine.MatlabExecutionError = MatlabExecutionError
    package = types.ModuleType("matlab")
    package.__spec__ = ModuleSpec("matlab", None)
    package.__path__ = []
    package.engine = engine
    sys.modules["matlab"], sys.modules["matlab.engine"] = package, engine


if __name__ == "__main__":  # self-check against a fake server, no MATLAB needed
    import tempfile
    import threading

    d = Path(tempfile.mkdtemp())

    def serve():
        while not (d / "quit").exists():
            for req in sorted(d.glob("req_*.mat")):
                r = scipy.io.loadmat(req)
                n = int(r["nargout"][0, 0])
                res = req.with_name(req.name.replace("req_", "res_"))
                if str(r["name"][0]) == "boom":
                    scipy.io.savemat(res, {"error": {"identifier": "x", "message": "Unknown mode", "stack": ""}})
                else:
                    outs = np.empty((1, n), dtype=object)
                    for i in range(n):
                        outs[0, i] = r["args"][0, i] if i < r["args"].shape[1] else np.array([[42.0]])
                    scipy.io.savemat(res, {"outputs": outs})
                req.unlink()
            time.sleep(0.005)

    threading.Thread(target=serve, daemon=True).start()
    install(d)
    import matlab.engine

    eng = matlab.engine.start_matlab()
    assert eng.which("icadefs") == "icadefs"
    assert eng.sqrt(16) == 16.0
    assert eng.f([1, 2, 3]).shape == (1, 3)
    assert eng.g("a", 2.0, nargout=2) == ("a", 2.0)
    assert eng.addpath("x", nargout=0) is None
    assert eng.workspace["X"] == 42.0
    assert eng.h({"a": 1.0, "b": "s"}) == {"a": 1.0, "b": "s"}
    try:
        eng.boom()
        raise AssertionError("no error")
    except matlab.engine.MatlabExecutionError as exc:
        assert "Unknown mode" in str(exc)
    (d / "quit").touch()
    print("ok")
