"""Tests for the plot tier.

Needs the ``plot`` extra, so the whole module skips without it.

The assertions here are deliberately literal rather than recomputed from the code under
test. An earlier round of tests in this package compared parsed objects to themselves and
survived every mutation that mattered; the spacing, ordering and demeaning rules below
are each pinned to a number worked out by hand.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

# Before importing numpy: the base install has nothing, and a module-scope numpy import
# would make this file fail collection there rather than skip.
pytest.importorskip("matplotlib", reason="needs the plot extra")
np = pytest.importorskip("numpy", reason="needs the plot extra")

from eegprep_lean.plot import (  # noqa: E402
    FALLBACK_SPACING,
    SPACING_HEADROOM,
    default_spacing,
    plot_window,
    to_png,
)
from eegprep_lean.window import Window  # noqa: E402

N_SAMPLES = 400
RATE = 250.0


def square(amplitude: float, n_samples: int = N_SAMPLES) -> np.ndarray:
    """A two-valued channel, so its percentile amplitude is exactly ``2 * amplitude``."""
    return np.where(np.arange(n_samples) % 2 == 0, amplitude, -amplitude).astype(np.float64)


def make_window(
    data: np.ndarray,
    *,
    channels: tuple[int, ...] | None = None,
    start_sample: int = 2500,
    physical: bool = True,
) -> Window:
    return Window(
        data=data,
        channels=channels if channels is not None else tuple(range(data.shape[0])),
        start_sample=start_sample,
        rate=RATE,
        group_name="eeg_250hz",
        physical=physical,
    )


def traces(ax) -> list:
    """The trace lines, in drawing order, excluding the scale bar."""
    return [line for line in ax.lines if (line.get_gid() or "").startswith("trace-")]


class TestGeometry:
    def test_one_trace_per_channel_tagged_by_channel(self) -> None:
        ax = plot_window(make_window(np.vstack([square(10)] * 3)))

        assert [line.get_gid() for line in traces(ax)] == ["trace-0", "trace-1", "trace-2"]

    def test_the_first_channel_is_drawn_at_the_top(self) -> None:
        """Every EEG viewer counts downward. Drawing channel 0 at the bottom would put
        the montage upside down while looking entirely plausible."""
        ax = plot_window(make_window(np.vstack([square(10)] * 3)))

        centers = [float(np.mean(line.get_ydata())) for line in traces(ax)]

        assert centers[0] > centers[1] > centers[2]

    def test_x_is_absolute_time_in_the_recording(self) -> None:
        """Sample 2500 at 250 Hz is ten seconds in. Times relative to the window would
        misplace every event a caller draws on top."""
        ax = plot_window(make_window(np.vstack([square(10)] * 2)))

        x = traces(ax)[0].get_xdata()

        assert float(x[0]) == pytest.approx(10.0)
        assert float(x[-1]) == pytest.approx(10.0 + 399 / 250.0)

    def test_rows_follow_the_channels_the_window_holds_in_order(self) -> None:
        """A window of channels 2 and 0 is labeled 2 then 0, not 0 then 1."""
        window = make_window(np.vstack([square(10), square(10)]), channels=(2, 0))

        ax = plot_window(window)

        assert [label.get_text() for label in ax.get_yticklabels()] == ["2", "0"]
        assert [line.get_gid() for line in traces(ax)] == ["trace-2", "trace-0"]


class TestSpacing:
    def test_spacing_follows_the_median_channel_not_the_largest(self) -> None:
        """Amplitudes 10, 10 and 1000 give per-channel extents of 20, 20 and 2000. The
        median is 20, so the spacing is 24; scaling to the maximum would give 2400 and
        flatten the two ordinary channels to invisible lines."""
        data = np.vstack([square(10), square(10), square(1000)])

        assert default_spacing(data) == pytest.approx(20.0 * SPACING_HEADROOM)
        assert SPACING_HEADROOM == 1.2

    def test_traces_do_not_overlap_at_the_default_spacing(self) -> None:
        ax = plot_window(make_window(np.vstack([square(10)] * 3)))

        spans = [(float(np.min(line.get_ydata())), float(np.max(line.get_ydata()))) for line in traces(ax)]

        for upper, lower in zip(spans[:-1], spans[1:], strict=True):
            assert lower[1] < upper[0], "adjacent traces overlap at the default spacing"

    def test_flat_channels_still_get_separate_rows(self) -> None:
        """A data-derived spacing is zero here, which would stack every trace on one
        line and look like a single flat channel."""
        ax = plot_window(make_window(np.zeros((3, N_SAMPLES))))

        centers = [float(np.mean(line.get_ydata())) for line in traces(ax)]

        assert default_spacing(np.zeros((3, N_SAMPLES))) == FALLBACK_SPACING
        assert centers == [0.0, -FALLBACK_SPACING, -2 * FALLBACK_SPACING]

    def test_an_explicit_spacing_is_used_exactly(self) -> None:
        ax = plot_window(make_window(np.vstack([square(10)] * 3)), spacing=50.0)

        centers = [float(np.mean(line.get_ydata())) for line in traces(ax)]

        assert centers == [0.0, -50.0, -100.0]


class TestDemeaning:
    OFFSET = 10_000.0

    def test_a_large_dc_offset_is_removed_for_display(self) -> None:
        """Level-0 channels sit at offsets in the thousands while the signal spans tens,
        so without this every trace is a flat line at its own level."""
        ax = plot_window(make_window(np.vstack([square(10) + self.OFFSET] * 2)))

        first = traces(ax)[0].get_ydata()

        assert float(np.mean(first)) == pytest.approx(0.0)
        assert float(np.ptp(first)) == pytest.approx(20.0)

    def test_demeaning_does_not_touch_the_window(self) -> None:
        """Display only. A plot that mutated its input would leave the caller holding
        numbers that no longer mean what the reader returned."""
        data = np.vstack([square(10) + self.OFFSET] * 2)
        window = make_window(data.copy())

        plot_window(window)

        assert np.array_equal(window.data, data)

    def test_demean_false_draws_the_values_as_they_are(self) -> None:
        ax = plot_window(make_window(np.vstack([square(10) + self.OFFSET] * 2)), demean=False, spacing=50.0)

        assert float(np.mean(traces(ax)[0].get_ydata())) == pytest.approx(self.OFFSET)


class TestStatingAmplitudeWithoutNamingAUnit:
    def _scalebar(self, ax):
        return next(line for line in ax.lines if line.get_gid() == "scalebar")

    def _scalebar_text(self, ax) -> str:
        return next(text for text in ax.texts if text.get_gid() == "scalebar-label").get_text()

    def test_the_scalebar_is_exactly_one_spacing_tall(self) -> None:
        """It is the only statement of amplitude on the plot, because the axis cannot
        carry a unit. A bar of the wrong height misreports every deflection."""
        ax = plot_window(make_window(np.vstack([square(10)] * 3)), spacing=50.0)

        y = self._scalebar(ax).get_ydata()

        assert float(y[1]) - float(y[0]) == pytest.approx(50.0)

    def test_the_label_distinguishes_converted_values_from_stored_counts(self) -> None:
        """Stored counts and physical values are different quantities; one word for both
        would let a reader take int16 counts for signal amplitude."""
        data = np.vstack([square(10)] * 2)

        assert "units" in self._scalebar_text(plot_window(make_window(data, physical=True)))
        assert "counts" in self._scalebar_text(plot_window(make_window(data, physical=False)))

    def test_nothing_on_the_plot_claims_a_unit(self) -> None:
        """The array declares scale and offset but not what they produce, so microvolts
        would be a guess that is right often enough to go unquestioned."""
        ax = plot_window(make_window(np.vstack([square(10)] * 2)))

        written = " ".join([ax.get_ylabel(), ax.get_xlabel(), ax.get_title(), self._scalebar_text(ax)]).lower()

        for unit in ("uv", "µv", "μv", "volt", "mv"):
            assert unit not in written, f"the plot claims {unit!r}, which is not knowable here"


class TestLabels:
    def test_supplied_labels_replace_the_channel_indices(self) -> None:
        ax = plot_window(make_window(np.vstack([square(10)] * 2)), labels=["Cz", "Pz"])

        assert [label.get_text() for label in ax.get_yticklabels()] == ["Cz", "Pz"]

    def test_the_wrong_number_of_labels_is_refused(self) -> None:
        """Labels are positional, so a short list would rename every trace below the
        first missing one rather than leaving it blank."""
        with pytest.raises(ValueError, match="labels for 3 channels"):
            plot_window(make_window(np.vstack([square(10)] * 3)), labels=["Cz", "Pz"])


class TestRendering:
    def test_to_png_returns_png_bytes(self) -> None:
        """The browser path: Pyodide has no window, so a figure reaches the page as
        bytes."""
        ax = plot_window(make_window(np.vstack([square(10)] * 3)))

        png = to_png(ax.figure)

        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 1000

    def test_drawing_never_imports_pyplot(self) -> None:
        """pyplot selects a backend at import, and in a browser that backend wants the
        DOM. Run in a subprocess because another test importing pyplot first would make
        an in-process check pass for the wrong reason.
        """
        import eegprep_lean

        package_parent = pathlib.Path(eegprep_lean.__file__).resolve().parent.parent
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(package_parent), env.get("PYTHONPATH", "")])
        script = (
            "import sys, numpy as np\n"
            "from eegprep_lean.plot import plot_window, to_png\n"
            "from eegprep_lean.window import Window\n"
            "w = Window(data=np.zeros((2, 8)), channels=(0, 1), start_sample=0, rate=250.0,"
            " group_name='g', physical=True)\n"
            "to_png(plot_window(w).figure)\n"
            "assert 'matplotlib' in sys.modules, 'matplotlib was never imported'\n"
            "sys.exit(17 if 'matplotlib.pyplot' in sys.modules else 0)\n"
        )

        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)

        assert result.returncode == 0, f"pyplot was imported (rc={result.returncode}): {result.stderr}"


@pytest.mark.network
class TestAgainstTheLiveArchive:
    def test_a_real_window_renders(self) -> None:
        """The whole chain in one: index, sharded store, physical conversion, plot.

        Synthetic data cannot catch this. The DC offsets that make demeaning necessary
        are a property of real level-0 arrays, and a plot of real channels is where a
        spacing rule chosen against tidy test signals shows whether it works.
        """
        pytest.importorskip("zarr", reason="needs the zarr extra as well")
        import asyncio

        from eegprep_lean import read_index
        from eegprep_lean.window import read_window

        async def run():
            index = await read_index("nm000103")
            return await read_window(index, index.stores[0], start_sample=2500, n_samples=500, channels=list(range(8)))

        window = asyncio.run(run())
        ax = plot_window(window)
        png = to_png(ax.figure)

        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(traces(ax)) == 8
        # Real level-0 channels sit at offsets in the thousands and differ from each
        # other by more than the signal spans. If they were drawn undemeaned, the traces
        # would be further apart than any sane spacing and the ylim would be enormous.
        bottom, top = ax.get_ylim()
        assert top - bottom < 8 * default_spacing(np.asarray(window.data)) * 3
