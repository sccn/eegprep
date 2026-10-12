"""MATLAB ``.fig`` export from Matplotlib figures.

The structural test always runs. ``test_openfig_round_trip`` opens the file in
MATLAB and is skipped when no engine is available; run it before merging.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import loadmat

from eegprep.functions.sigprocfunc.savefig import savefig
from tests.fixtures import matlab_engine_available

IMAGE = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
LINE_X = np.array([1.0, 2.0, 3.0])
LINE_Y = np.array([1.0, 4.0, 9.0])


def _sample_figure():
    fig, axes = plt.subplots(1, 2, num="IRLS export", figsize=(8, 4))
    axes[0].plot(LINE_X, LINE_Y, color="blue", linestyle="--", linewidth=2.0, marker="o")
    axes[0].set_title("Hello\nworld")
    axes[0].set_xlabel("time")
    axes[0].set_yscale("log")
    axes[0].grid(True)
    axes[0].text(2.0, 4.0, "marker")
    axes[1].imshow(IMAGE, aspect="auto")
    axes[1].set_ylabel("channels")
    fig.suptitle("Export")
    return fig, axes


def _matlab_text(value) -> str:
    array = np.asarray(value)
    if array.dtype.kind not in "US":
        return "" if array.size == 0 else str(array.reshape(-1)[0])
    if array.ndim == 2 and array.dtype.itemsize == np.dtype("U1").itemsize:
        return "\n".join("".join(row).rstrip() for row in array.astype(str))
    return "".join(array.astype(str).ravel())


def _children(node):
    children = node.children
    if isinstance(children, np.ndarray):
        if children.size == 0:
            return []
        return list(np.ravel(children))
    return [] if children is None else [children]


def _properties(node):
    props = node.properties
    if isinstance(props, np.ndarray):
        return props.reshape(-1)[0]
    return props


def _walk(node):
    yield node
    for child in _children(node):
        yield from _walk(child)


def _load_root(path):
    return loadmat(path, squeeze_me=True, struct_as_record=False)["hgS_070000"]


def _by_type(root, kind):
    return [node for node in _walk(root) if _matlab_text(node.type) == kind]


def test_savefig_stores_figure_axes_line_image_and_text(tmp_path):
    fig, axes = _sample_figure()
    written = savefig(fig, tmp_path / "irls")
    try:
        assert written.endswith(".fig")
        assert open(written, "rb").read(6) == b"MATLAB"

        root = _load_root(written)
        assert _matlab_text(root.type) == "figure"
        figure_props = _properties(root)
        assert _matlab_text(figure_props.Name) == "IRLS export"
        assert _matlab_text(figure_props.NumberTitle) == "off"
        assert np.asarray(figure_props.Colormap, dtype=float).shape[1] == 3

        axes_nodes = _by_type(root, "axes")
        assert len(axes_nodes) == 2
        line_axes = next(node for node in axes_nodes if _by_type(node, "graph2d.lineseries"))
        image_axes = next(node for node in axes_nodes if _by_type(node, "image"))
        line_props = _properties(_by_type(line_axes, "graph2d.lineseries")[0])
        image_props = _properties(_by_type(image_axes, "image")[0])
        texts = [_matlab_text(_properties(node).String) for node in _by_type(root, "text")]

        np.testing.assert_allclose(np.ravel(line_props.XData), LINE_X)
        np.testing.assert_allclose(np.ravel(line_props.YData), LINE_Y)
        np.testing.assert_allclose(np.ravel(line_props.Color), [0, 0, 1])
        assert _matlab_text(line_props.LineStyle) == "--"
        assert float(np.ravel(line_props.LineWidth)[0]) == pytest.approx(2.0)
        assert _matlab_text(line_props.Marker) == "o"
        assert _matlab_text(_properties(line_axes).YScale) == "log"
        assert _matlab_text(_properties(line_axes).XGrid) == "on"
        np.testing.assert_allclose(np.ravel(_properties(line_axes).Position), axes[0].get_position().bounds)

        np.testing.assert_allclose(np.asarray(image_props.CData, dtype=float), IMAGE)
        assert _matlab_text(image_props.CDataMapping) == "scaled"
        assert _matlab_text(_properties(image_axes).YDir) == "reverse"
        assert "Hello\nworld" in texts
        assert "marker" in texts
        assert "channels" in texts
        assert "Export" in texts
        marker = next(node for node in _by_type(root, "text") if _matlab_text(_properties(node).String) == "marker")
        assert _matlab_text(_properties(marker).Units) == "data"
        np.testing.assert_allclose(np.ravel(_properties(marker).Position)[:2], [2.0, 4.0])
    finally:
        plt.close(fig)


def test_matplotlib_fig_format_uses_the_same_writer(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [2.0, 3.0], color="black")
    path = tmp_path / "from-matplotlib.fig"
    try:
        fig.savefig(path, format="fig")
        root = _load_root(path)
        line = _by_type(root, "graph2d.lineseries")[0]
        np.testing.assert_allclose(np.ravel(_properties(line).YData), [2.0, 3.0])
    finally:
        plt.close(fig)


@pytest.mark.matlab
def test_openfig_round_trip(tmp_path):
    """MATLAB openfig recovers line, image, and text stored by savefig."""
    if not matlab_engine_available():
        pytest.skip("MATLAB engine not available")
    matlab_engine = importlib.import_module("matlab.engine")

    fig, _axes = _sample_figure()
    fig_path = tmp_path / "roundtrip.fig"
    out_path = tmp_path / "roundtrip.mat"
    savefig(fig, fig_path)
    plt.close(fig)
    fig_literal = fig_path.as_posix().replace("'", "''")
    out_literal = out_path.as_posix().replace("'", "''")
    engine = matlab_engine.start_matlab()
    try:
        engine.eval(
            f"""
            set(0, 'DefaultFigureVisible', 'off');
            h = openfig('{fig_literal}', 'invisible');
            ln = findobj(h, 'Type', 'line');
            im = findobj(h, 'Type', 'image');
            tx = findall(h, 'Type', 'text');
            xdata = get(ln, 'XData');
            ydata = get(ln, 'YData');
            color = get(ln, 'Color');
            linestyle = get(ln, 'LineStyle');
            cdata = get(im, 'CData');
            ydir = get(ancestor(im, 'axes'), 'YDir');
            ntext = numel(tx);
            labels = cell(1, ntext);
            for k = 1:ntext
                labels{{k}} = get(tx(k), 'String');
            end
            close(h);
            save('{out_literal}', 'xdata', 'ydata', 'color', 'linestyle', 'cdata', 'ydir', 'labels');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    saved = loadmat(out_path, squeeze_me=True)
    np.testing.assert_allclose(np.ravel(saved["xdata"]), LINE_X)
    np.testing.assert_allclose(np.ravel(saved["ydata"]), LINE_Y)
    np.testing.assert_allclose(np.ravel(saved["color"]), [0, 0, 1])
    assert _matlab_text(saved["linestyle"]) == "--"
    np.testing.assert_allclose(np.asarray(saved["cdata"], dtype=float), IMAGE)
    assert _matlab_text(saved["ydir"]) == "reverse"
    labels = saved["labels"]
    texts = [_matlab_text(item) for item in np.ravel(labels)]
    assert "Hello\nworld" in texts
    assert "marker" in texts
    assert "Export" in texts
