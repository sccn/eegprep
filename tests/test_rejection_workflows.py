import copy
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.io

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.popfunc.eeg_rejsuperpose import eeg_rejsuperpose
from eegprep.functions.popfunc.pop_autorej import pop_autorej
from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh
from eegprep.functions.popfunc.pop_jointprob import pop_jointprob
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_rejchan import pop_rejchan
from eegprep.functions.popfunc.pop_rejcont import pop_rejcont
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt
from eegprep.functions.popfunc.pop_rejmenu import pop_rejmenu
from eegprep.functions.popfunc.pop_rejspec import pop_rejspec
from eegprep.functions.popfunc.pop_rejtrend import pop_rejtrend
from eegprep.functions.popfunc.pop_selectcomps import pop_selectcomps
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg


def _epoched_eeg() -> dict:
    rng = np.random.default_rng(4)
    eeg = create_test_eeg(n_channels=4, n_samples=80, n_trials=5, srate=100)
    data = rng.normal(0, 0.05, (4, 80, 5))
    data[0, 10:20, 1] = 25
    data[1, :, 2] += np.linspace(0, 8, 80)
    data[2, :, 3] += 4 * np.sin(2 * np.pi * 25 * np.arange(80) / 100)
    eeg["data"] = data
    eeg["icaweights"] = np.eye(4)
    eeg["icasphere"] = np.eye(4)
    eeg["icawinv"] = np.eye(4)
    eeg["icachansind"] = np.arange(4)
    eeg["icaact"] = None
    eeg["reject"] = {}
    return eeg


def test_pop_eegthresh_marks_epochs_and_emits_replayable_python():
    eeg = _epoched_eeg()

    out, com = pop_eegthresh(eeg, 1, [1], -10, 10, 0, 0.79, 0, 0, return_com=True)

    assert out["reject"]["rejthresh"].tolist() == [False, True, False, False, False]
    assert out["reject"]["rejthreshE"][0].tolist() == [False, True, False, False, False]
    assert _console_python_command(com) == (
        "EEG = pop_eegthresh(EEG, icacomp=1, elecrange=[1], negthresh=[-10], "
        "posthresh=[10], starttime=[0], endtime=[0.79], superpose=0, reject=0)"
    )


def test_rejection_statistics_store_data_and_component_marks():
    eeg = _epoched_eeg()

    prob_out, _local, _global, prob_count = pop_jointprob(eeg, 1, [1, 2, 3, 4], 1.2, 1.2, 0, 0)
    kurt_out, _local, _global, kurt_count = pop_rejkurt(eeg, 1, [1, 2, 3, 4], 1.2, 1.2, 0, 0)
    trend_out = pop_rejtrend(eeg, 1, [2], 80, 0.2, 0.3, 0, 0)
    spec_out, spec_indices = pop_rejspec(
        eeg,
        1,
        "elecrange",
        [3],
        "threshold",
        [-10, 10],
        "freqlimits",
        [20, 30],
        "eegplotreject",
        0,
    )
    comp_out, _local, _global, comp_count = pop_jointprob(eeg, 0, [1, 2, 3, 4], 1.2, 1.2, 0, 0)

    assert prob_count >= 1
    assert kurt_count >= 0
    assert prob_out["reject"]["rejjpE"].shape == (4, 5)
    assert kurt_out["reject"]["rejkurtE"].shape == (4, 5)
    assert trend_out["reject"]["rejconst"][2]
    assert spec_indices
    assert spec_out["specdata"].shape[:2] == (4, 40)
    assert comp_count >= 1
    assert "icarejjp" in comp_out["reject"]


def test_eeg_rejsuperpose_and_pop_rejepoch_remove_marked_epochs():
    eeg = _epoched_eeg()
    eeg["reject"]["rejmanual"] = np.array([False, True, False, False, True])
    eeg["reject"]["rejmanualE"] = np.zeros((4, 5), dtype=bool)
    eeg["reject"]["rejmanualE"][0, 1] = True
    eeg["reject"]["rejmanualE"][1, 4] = True

    marked, com = eeg_rejsuperpose(eeg, 1, 1, 0, 0, 0, 0, 0, 0, return_com=True)

    assert marked["reject"]["rejglobal"].tolist() == [False, True, False, False, True]
    assert marked["reject"]["rejglobalE"].shape == (4, 5)
    removed, reject_com = pop_rejepoch(copy.deepcopy(marked), marked["reject"]["rejglobal"], 0, return_com=True)
    assert removed["trials"] == 3
    assert _console_python_command(com) == "EEG = eeg_rejsuperpose(EEG, 1, 1, 0, 0, 0, 0, 0, 0)"
    assert _console_python_command(reject_com) == "EEG = pop_rejepoch(EEG, tmprej=[2, 5], confirm=0)"


@pytest.mark.matlab
def test_eeg_rejsuperpose_matches_eeglab_for_deterministic_marks(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    matlab = shutil.which("matlab")
    if matlab is None:
        pytest.skip("MATLAB executable not available")
    eeglab_root = _eeglab_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB source not available for parity reference")

    eeg = _epoched_eeg()
    reject = {
        "rejmanual": np.array([False, True, False, False, False]),
        "rejmanualE": np.array(
            [
                [False, True, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
        "rejthresh": np.array([False, False, True, False, False]),
        "rejthreshE": np.array(
            [
                [False, False, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
        "rejconst": np.zeros(5, dtype=bool),
        "rejconstE": np.zeros((4, 5), dtype=bool),
        "rejjp": np.array([False, False, False, True, False]),
        "rejjpE": np.array(
            [
                [False, False, False, True, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
        "rejkurt": np.zeros(5, dtype=bool),
        "rejkurtE": np.zeros((4, 5), dtype=bool),
        "rejfreq": np.array([True, False, False, False, False]),
        "rejfreqE": np.array(
            [
                [True, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
    }
    eeg["reject"] = reject
    py_out = eeg_rejsuperpose(eeg, 1, 1, 1, 1, 1, 1, 1, 0)

    script = tmp_path / "eeg_rejsuperpose_parity.m"
    output = tmp_path / "out.mat"
    script.write_text(_matlab_rejsuperpose_script(eeglab_root, output), encoding="utf-8")
    result = subprocess.run(
        [matlab, "-batch", f"run('{script.as_posix()}')"], check=False, capture_output=True, text=True
    )
    if result.returncode:
        pytest.fail(result.stdout + result.stderr)
    matlab_out = scipy.io.loadmat(output)

    np.testing.assert_array_equal(
        np.asarray(py_out["reject"]["rejglobal"], dtype=bool), matlab_out["rejglobal"].ravel()
    )
    np.testing.assert_array_equal(np.asarray(py_out["reject"]["rejglobalE"], dtype=bool), matlab_out["rejglobalE"])


def test_pop_rejmenu_can_combine_marks_without_browser():
    eeg = _epoched_eeg()
    eeg["reject"]["rejthresh"] = np.array([False, True, False, False, False])
    eeg["reject"]["rejthreshE"] = np.zeros((4, 5), dtype=bool)

    out, com = pop_rejmenu(eeg, 1, gui=False, return_com=True)

    assert out["reject"]["rejglobal"].tolist() == [False, True, False, False, False]
    assert _console_python_command(com) == "EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 1)"


def test_pop_autorej_preserves_original_epoch_numbers_during_iterative_rejection():
    eeg = _epoched_eeg()

    out, rejected = pop_autorej(eeg, "threshold", 10, "startprob", 20, "maxrej", 40, "nogui", "on")

    assert 2 in rejected
    assert out["trials"] <= eeg["trials"]
    assert rejected == sorted(set(rejected))


def test_channel_and_continuous_rejection_work_on_sample_data_without_ica():
    sample = pop_loadset(SAMPLE_DATASET_PATH)

    _, rejected_channels, measure = pop_rejchan(sample, "measure", "std", "threshold", 1e9, "indexonly", "on")
    _, selected_regions = pop_rejcont(
        sample,
        "elecrange",
        [1],
        "threshold",
        1e9,
        "epochlength",
        0.5,
        "contiguous",
        1,
        "onlyreturnselection",
        "on",
    )

    assert rejected_channels == []
    assert measure.shape == (32,)
    assert selected_regions.shape == (0, 2)
    with pytest.raises(ValueError, match="ICA decomposition is required"):
        pop_eegthresh(sample, 0, [1], -10, 10, 0, 1)


def test_component_selection_and_viewprops_are_replayable_without_scrolling_browser():
    eeg = _epoched_eeg()

    selected, select_com = pop_selectcomps(eeg, [1, 3], reject=[2], plot=False, return_com=True)
    figures, props_com = pop_viewprops(eeg, 0, [1, 2], plot=False, return_com=True)

    assert selected["reject"]["gcompreject"].tolist() == [False, True, False, False]
    assert figures == []
    assert _console_python_command(select_com) == "EEG = pop_selectcomps(EEG, compnum=[1, 3], reject=[2])"
    assert _console_python_command(props_com) == (
        "pop_viewprops(EEG, typecomp=0, chanorcomp=[1, 2], spec_opt=[], erp_opt=[], scroll_event=1, classifier_name='')"
    )


def test_gui_cancel_paths_leave_datasets_unchanged():
    class CancelRenderer:
        def run(self, spec, initial_values=None):
            return None

    eeg = _epoched_eeg()
    out, com = pop_eegthresh(eeg, gui=True, renderer=CancelRenderer(), return_com=True)
    rejchan_out, rejchan_com = pop_rejchan(copy.deepcopy(eeg), gui=True, renderer=CancelRenderer(), return_com=True)

    assert out is eeg
    assert com == ""
    assert rejchan_out["data"].shape == eeg["data"].shape
    assert rejchan_com == ""
    plt.close("all")


def _eeglab_root() -> Path | None:
    candidates = []
    if os.environ.get("EEGPREP_EEGLAB_ROOT"):
        candidates.append(Path(os.environ["EEGPREP_EEGLAB_ROOT"]))
    candidates.append(Path(__file__).resolve().parents[1] / "src" / "eegprep" / "eeglab")
    for candidate in candidates:
        if (candidate / "functions" / "popfunc" / "eeg_rejsuperpose.m").exists():
            return candidate
    return None


def _matlab_rejsuperpose_script(eeglab_root: Path, output: Path) -> str:
    return f"""
addpath(fullfile('{eeglab_root.as_posix()}', 'functions', 'popfunc'));
EEG = struct();
EEG.trials = 5;
EEG.nbchan = 4;
EEG.reject = struct();
EEG.reject.rejmanual = logical([0 1 0 0 0]);
EEG.reject.rejmanualE = logical([0 1 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]);
EEG.reject.rejthresh = logical([0 0 1 0 0]);
EEG.reject.rejthreshE = logical([0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]);
EEG.reject.rejconst = logical([0 0 0 0 0]);
EEG.reject.rejconstE = logical(zeros(4, 5));
EEG.reject.rejjp = logical([0 0 0 1 0]);
EEG.reject.rejjpE = logical([0 0 0 1 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]);
EEG.reject.rejkurt = logical([0 0 0 0 0]);
EEG.reject.rejkurtE = logical(zeros(4, 5));
EEG.reject.rejfreq = logical([1 0 0 0 0]);
EEG.reject.rejfreqE = logical([1 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]);
EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 0);
rejglobal = EEG.reject.rejglobal;
rejglobalE = EEG.reject.rejglobalE;
save('{output.as_posix()}', 'rejglobal', 'rejglobalE');
"""
