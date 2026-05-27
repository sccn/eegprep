import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_topoplot import plot_channel_locations, pop_topoplot, pop_topoplot_dialog_spec
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
