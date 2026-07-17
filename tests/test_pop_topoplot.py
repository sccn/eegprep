import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_topoplot import (
    _latency_positions,
    _parse_items_text,
    plot_channel_locations,
    pop_topoplot,
    pop_topoplot_dialog_spec,
)
from eegprep.functions.sigprocfunc.topoplot import topoplot
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg_with_ica


def test_topoplot_blank_channel_locations_by_label_and_number():
    chanlocs = [
        {"labels": "Fz", "theta": 0, "radius": 0.3},
        {"labels": "Cz", "theta": 0, "radius": 0.0},
        {"labels": "Pz", "theta": 180, "radius": 0.3},
    ]

    label_fig, *_ = topoplot([], chanlocs, style="blank", electrodes="labelpoint")
    number_fig, *_ = topoplot([], chanlocs, style="blank", electrodes="numpoint")

    assert [text.get_text() for text in label_fig.axes[0].texts] == ["Fz", "Cz", "Pz"]
    assert [text.get_text() for text in number_fig.axes[0].texts] == ["1", "2", "3"]
    plt.close(label_fig)
    plt.close(number_fig)


def test_plot_channel_locations_returns_valid_console_command():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20)

    figure, command = plot_channel_locations(eeg, mode="numbers", return_com=True)

    assert command == "topoplot([], EEG['chanlocs'], style='blank', electrodes='numpoint')"
    assert figure.axes[0].get_title() == "Channel locations"
    plt.close(figure)


def test_plot_channel_locations_rejects_unknown_mode():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20)

    try:
        plot_channel_locations(eeg, mode="bad")
    except ValueError as exc:
        assert "mode must be" in str(exc)
    else:
        raise AssertionError("expected invalid channel-location mode ValueError")


def test_pop_topoplot_plots_sample_data_erp_maps_headlessly():
    eeg = pop_loadset(SAMPLE_DATASET_PATH)

    figures, command = pop_topoplot(
        eeg,
        typeplot=1,
        items=[0, 100],
        topotitle="sample maps",
        rowcols=[1, 2],
        return_com=True,
        electrodes="off",
        colorbar="off",
    )

    assert len(figures) == 1
    assert [axis.get_title() for axis in figures[0].axes] == ["0 ms", "100 ms"]
    assert "pop_topoplot(EEG, typeplot=1" in command
    assert "items=[0, 100]" in command
    assert "electrodes='off'" in command
    plt.close(figures[0])


def test_pop_topoplot_multi_map_pages_include_shared_colorbar_by_default():
    eeg = create_test_eeg_with_ica(n_channels=6, n_samples=30)

    figures = pop_topoplot(
        eeg,
        typeplot=1,
        items=[0, 20],
        topotitle="shared scale",
        rowcols=[1, 2],
        electrodes="off",
    )

    assert len(figures) == 1
    assert [axis.get_title() for axis in figures[0].axes[:2]] == ["0 ms", "20 ms"]
    assert len(figures[0].axes) == 3
    plt.close(figures[0])


def test_pop_topoplot_component_pages_scale_each_map_to_own_absmax():
    eeg = create_test_eeg_with_ica(n_channels=6, n_samples=30, n_components=2)
    eeg["icawinv"] = np.column_stack([np.arange(1, 7) * 1.0, np.arange(1, 7) * 10.0])

    figures = pop_topoplot(
        eeg,
        typeplot=0,
        items=[1, 2],
        topotitle="component scale",
        rowcols=[1, 2],
        electrodes="off",
    )

    expected = []
    for index in range(2):
        _, zi, *_ = topoplot(eeg["icawinv"][:, index], eeg["chanlocs"], noplot="on")
        limit = float(np.nanmax(np.abs(zi))) * 1.05  # topoplot widens the color axis by EEGLAB's 5% margin
        expected.append((-limit, limit))
    clims = [axis.images[0].get_clim() for axis in figures[0].axes[:2]]
    np.testing.assert_allclose(clims[0], expected[0], rtol=1e-6)
    np.testing.assert_allclose(clims[1], expected[1], rtol=1e-6)
    assert not np.allclose(clims[0], clims[1])
    assert len(figures[0].axes) == 3
    plt.close(figures[0])


def test_pop_topoplot_component_colorbar_uses_polarity_labels():
    eeg = create_test_eeg_with_ica(n_channels=6, n_samples=30, n_components=3)

    comp_figs = pop_topoplot(eeg, typeplot=0, items=[1, 2], topotitle="ic", rowcols=[1, 2], electrodes="off")
    comp_figs[0].canvas.draw()
    comp_labels = [text.get_text() for text in comp_figs[0].axes[-1].get_yticklabels()]
    assert comp_labels == ["-", "0", "+"]
    plt.close(comp_figs[0])

    erp_figs = pop_topoplot(eeg, typeplot=1, items=[0, 20], topotitle="erp", rowcols=[1, 2], electrodes="off")
    erp_figs[0].canvas.draw()
    erp_labels = [text.get_text() for text in erp_figs[0].axes[-1].get_yticklabels()]
    assert erp_labels != ["-", "0", "+"]
    plt.close(erp_figs[0])


def test_pop_topoplot_plots_component_maps_with_inverted_and_blank_items():
    eeg = create_test_eeg_with_ica(n_channels=6, n_samples=50, n_components=3)

    figures, command = pop_topoplot(
        eeg,
        typeplot=0,
        items=[1, -2, float("nan")],
        topotitle="IC maps",
        rowcols=[],
        return_com=True,
        colorbar="off",
    )

    assert len(figures) == 1
    assert [axis.get_title() for axis in figures[0].axes[:2]] == ["IC 1", "IC -2"]
    assert "items=[1, -2, float('nan')]" in command
    plt.close(figures[0])


def test_pop_topoplot_item_text_parses_eeglab_colon_ranges():
    assert _parse_items_text("-100:50:0") == [-100.0, -50.0, 0.0]
    assert _parse_items_text("0.5:0.25:1") == [0.5, 0.75, 1.0]
    tenth_steps = _parse_items_text("0:0.1:1")
    assert len(tenth_steps) == 11
    assert tenth_steps[-1] == 1.0
    parsed = _parse_items_text("1:2 NaN 5")
    assert parsed[:2] == [1.0, 2.0]
    assert np.isnan(parsed[2])
    assert parsed[3] == 5.0


def test_pop_topoplot_latency_positions_use_matlab_rounding():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=4)
    eeg["xmin"] = 0.0
    eeg["xmax"] = 3.0
    eeg["pnts"] = 4

    assert _latency_positions(eeg, np.array([500.0, 2500.0])).tolist() == [1, 3]


def test_pop_topoplot_gui_parses_signed_decimal_step_ranges():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=5, n_trials=2)
    eeg["xmin"] = -0.1
    eeg["xmax"] = 0.1
    eeg["times"] = np.linspace(-100, 100, eeg["pnts"])

    class Renderer:
        def run(self, spec, initial_values=None):
            return {
                "items": "-100:50.0:0",
                "topotitle": "range",
                "rowcols": "[1 3]",
                "options": "'electrodes', 'off', 'colorbar', 'off'",
            }

    figures, command = pop_topoplot(eeg, typeplot=1, renderer=Renderer(), return_com=True)

    assert [axis.get_title() for axis in figures[0].axes[:3]] == ["-100 ms", "-50 ms", "0 ms"]
    assert "items=[-100, -50, 0]" in command
    plt.close(figures[0])


def test_pop_topoplot_all_blank_items_do_not_crash():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20, n_components=2)

    figures, command = pop_topoplot(
        eeg,
        typeplot=1,
        items=[float("nan")],
        topotitle="blank",
        rowcols=[1, 1],
        return_com=True,
    )

    assert len(figures) == 1
    assert "items=[float('nan')]" in command
    plt.close(figures[0])


def test_pop_topoplot_rejects_finite_latency_when_epoch_range_collapsed():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=5)
    eeg["xmin"] = 0.0
    eeg["xmax"] = 0.0

    try:
        pop_topoplot(eeg, typeplot=1, items=[0])
    except ValueError as exc:
        assert "outside the epoch time range" in str(exc)
    else:
        raise AssertionError("expected collapsed epoch range ValueError")

    figures = pop_topoplot(eeg, typeplot=1, items=[float("nan")], rowcols=[1, 1], colorbar="off")
    assert len(figures) == 1
    plt.close(figures[0])


def test_pop_topoplot_gui_parses_eeglab_style_options():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20, n_components=4)

    class Renderer:
        def run(self, spec, initial_values=None):
            assert pop_topoplot_dialog_spec(eeg, typeplot=0).title == spec.title
            return {
                "items": "1:2",
                "topotitle": "components",
                "rowcols": "[1 2]",
                "plotdip": False,
                "options": "'electrodes', 'off', 'colorbar', 'off'",
            }

    figures, command = pop_topoplot(eeg, typeplot=0, renderer=Renderer(), return_com=True)

    assert len(figures) == 1
    assert "typeplot=0" in command
    assert "items=[1, 2]" in command
    assert "colorbar='off'" in command
    plt.close(figures[0])


def test_pop_topoplot_rejects_missing_ica_or_chanlocs():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20, n_components=4)
    eeg["icawinv"] = np.array([])

    try:
        pop_topoplot(eeg, typeplot=0, items=[1])
    except ValueError as exc:
        assert "no ICA data" in str(exc)
    else:
        raise AssertionError("expected missing ICA ValueError")

    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20)
    eeg["chanlocs"] = []
    try:
        pop_topoplot(eeg, typeplot=1, items=[0])
    except ValueError as exc:
        assert "channel location" in str(exc)
    else:
        raise AssertionError("expected missing channel location ValueError")


def test_pop_topoplot_gui_preflights_missing_inputs_before_dialog():
    class Renderer:
        def run(self, spec, initial_values=None):
            raise AssertionError("renderer should not run when preflight fails")

    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20, n_components=4)
    eeg["icawinv"] = None

    try:
        pop_topoplot(eeg, typeplot=0, renderer=Renderer())
    except ValueError as exc:
        assert "no ICA data" in str(exc)
    else:
        raise AssertionError("expected missing ICA ValueError")

    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20)
    eeg["chanlocs"] = []
    try:
        pop_topoplot(eeg, typeplot=1, renderer=Renderer())
    except ValueError as exc:
        assert "channel location" in str(exc)
    else:
        raise AssertionError("expected missing channel location ValueError")
