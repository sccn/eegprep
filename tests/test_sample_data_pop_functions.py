from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt

from eegprep.functions.adminfunc.eegh import eegh
from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.adminfunc.pop_delset import pop_delset
from eegprep.functions.adminfunc.pop_editoptions import pop_editoptions
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.eeg_runica import eeg_runica
from eegprep.functions.popfunc.pop_adjustevents import pop_adjustevents
from eegprep.functions.popfunc.pop_biosig import pop_biosig
from eegprep.functions.popfunc.pop_chanevent import pop_chanevent
from eegprep.functions.popfunc.pop_chansel import pop_chansel_display_values, pop_chansel_selected_string
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_expevents import pop_expevents
from eegprep.functions.popfunc.pop_expica import pop_expica
from eegprep.functions.popfunc.pop_export import pop_export
from eegprep.functions.popfunc.pop_fileio import pop_fileio
from eegprep.functions.popfunc.pop_importdata import pop_importdata
from eegprep.functions.popfunc.pop_importepoch import pop_importepoch
from eegprep.functions.popfunc.pop_importerplab import pop_importerplab
from eegprep.functions.popfunc.pop_importevent import pop_importevent
from eegprep.functions.popfunc.pop_importpres import pop_importpres
from eegprep.functions.popfunc.pop_interp import pop_interp
from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_newset import pop_newset
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase
from eegprep.functions.popfunc.pop_runica import pop_runica
from eegprep.functions.popfunc.pop_runscript import pop_runscript
from eegprep.functions.popfunc.pop_saveh import pop_saveh
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.popfunc.pop_subcomp import pop_subcomp
from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg
from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy
from eegprep.functions.studyfunc.pop_chanplot import pop_chanplot
from eegprep.functions.studyfunc.pop_precomp import pop_precomp
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.pop_studyerp import pop_studyerp
from eegprep.functions.studyfunc.pop_studywizard import pop_studywizard
from eegprep.plugins.EEG_BIDS.bids_tools import pop_eventinfo, pop_participantinfo, pop_taskinfo, validate_bids
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids
from eegprep.plugins.EEG_BIDS.pop_importbids import pop_importbids
from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel
from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS, pop_icflag
from eegprep.plugins.clean_rawdata.clean_artifacts import clean_artifacts
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr
from eegprep.plugins.clean_rawdata.clean_channels import clean_channels
from eegprep.plugins.clean_rawdata.clean_windows import clean_windows
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata


SAMPLE_SET = Path("sample_data/eeglab_data.set")


@pytest.fixture(scope="module")
def sample_eeg_base():
    return pop_loadset(SAMPLE_SET)


@pytest.fixture
def sample_eeg(sample_eeg_base):
    return copy.deepcopy(sample_eeg_base)


@pytest.fixture(scope="module")
def sample_eeg_with_ica_base(sample_eeg_base):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        eeg, command = pop_runica(copy.deepcopy(sample_eeg_base), maxsteps=1, extended=1, return_com=True)
    assert command == "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 1);"
    return eeg


@pytest.fixture
def sample_eeg_with_ica(sample_eeg_with_ica_base):
    return copy.deepcopy(sample_eeg_with_ica_base)


def test_pop_loadset_loads_eeglab_sample_data_with_core_fields(sample_eeg_base):
    assert sample_eeg_base["data"].shape == (32, 30504)
    assert sample_eeg_base["nbchan"] == 32
    assert sample_eeg_base["pnts"] == 30504
    assert sample_eeg_base["trials"] == 1
    assert sample_eeg_base["srate"] == 128
    assert sample_eeg_base["setname"] == "Continuous EEG Data"
    assert len(sample_eeg_base["event"]) == 154
    assert sample_eeg_base["event"][0]["type"] == "square"
    assert np.issubdtype(np.asarray(sample_eeg_base["icachansind"]).dtype, np.integer)


def test_pop_fileio_loads_sample_set_and_records_replayable_history(sample_eeg_base):
    eeg, command = pop_fileio(SAMPLE_SET, return_com=True)

    assert eeg["data"].shape == sample_eeg_base["data"].shape
    assert eeg["event"][0]["type"] == "square"
    assert eeg["history"] == command
    assert command == "EEG = pop_fileio('sample_data/eeglab_data.set');"


def test_pop_biosig_rejects_sample_set_because_it_is_not_a_biosig_file():
    with pytest.raises(ValueError, match="BIOSIG|EDF|BDF|GDF|Unsupported"):
        pop_biosig(SAMPLE_SET, return_com=True)


def test_pop_select_keeps_named_sample_channels(sample_eeg):
    original_data = sample_eeg["data"].copy()
    selected, command = pop_select(sample_eeg, channel=["FPz", "F3"], return_com=True)

    assert selected["data"].shape == (2, 30504)
    assert selected["nbchan"] == 2
    assert [chan["labels"] for chan in selected["chanlocs"]] == ["FPz", "F3"]
    np.testing.assert_allclose(selected["data"][0], original_data[0])
    np.testing.assert_allclose(selected["data"][1], original_data[2])
    assert command == "EEG = pop_select( EEG, 'channel', {'FPz' 'F3'});"


def test_pop_resample_halves_sample_rate_and_event_latencies(sample_eeg):
    resampled, command = pop_resample(sample_eeg, 64, return_com=True)

    assert resampled["srate"] == 64
    assert resampled["pnts"] == 15252
    assert resampled["data"].shape == (32, 15252)
    assert len(resampled["event"]) == len(sample_eeg["event"])
    assert resampled["event"][0]["latency"] == pytest.approx((sample_eeg["event"][0]["latency"] - 1) * 0.5 + 1)
    assert resampled["icaact"].size == 0
    assert command == "EEG = pop_resample( EEG, 64);"


def test_pop_resample_logs_eeglab_style_progress(sample_eeg, caplog):
    with caplog.at_level(logging.INFO, logger="eegprep.functions.popfunc.pop_resample"):
        pop_resample(sample_eeg, 64, return_com=True)

    messages = [record.getMessage() for record in caplog.records]
    assert any("resampling data 64 Hz" in message for message in messages)
    assert any("resampling event latencies" in message for message in messages)
    assert any("resampling finished" in message for message in messages)


def test_pop_reref_average_references_sample_data_without_nonfinite_values(sample_eeg):
    reref, command = pop_reref(sample_eeg, [], return_com=True)

    assert reref["ref"] == "average"
    assert reref["data"].shape == sample_eeg["data"].shape
    assert np.isfinite(reref["data"]).all()
    np.testing.assert_allclose(reref["data"].mean(axis=0), 0, atol=1e-5)
    assert command == "EEG = pop_reref( EEG, []);"


def test_pop_epoch_extracts_square_locked_sample_epochs(sample_eeg):
    epoched, command = pop_epoch(sample_eeg, ["square"], [-0.1, 0.2], return_com=True)

    assert epoched["trials"] == 80
    assert epoched["pnts"] == 39
    assert epoched["data"].shape == (32, 39, 80)
    assert epoched["xmin"] == pytest.approx(-0.1, abs=1 / sample_eeg["srate"])
    assert epoched["xmax"] == pytest.approx(0.2, abs=1 / sample_eeg["srate"])
    assert len(epoched["epoch"]) == 80
    assert command == "EEG = pop_epoch( EEG, { 'square' }, [-0.1 0.2]);"


def test_pop_adjustevents_shifts_only_requested_sample_event_type(sample_eeg):
    first_square = next(event for event in sample_eeg["event"] if event["type"] == "square")
    first_rt = next(event for event in sample_eeg["event"] if event["type"] == "rt")

    adjusted, command = pop_adjustevents(sample_eeg, addms=10, eventtypes=["square"], force="on", return_com=True)
    adjusted_square = next(event for event in adjusted["event"] if event["type"] == "square")
    adjusted_rt = next(event for event in adjusted["event"] if event["type"] == "rt")

    assert adjusted_square["latency"] == pytest.approx(first_square["latency"] + 1.28)
    assert adjusted_rt["latency"] == pytest.approx(first_rt["latency"])
    assert "'eventtypes', {'square'}" in command


def test_pop_chanevent_adds_edges_from_sample_channel_without_deleting_channel(sample_eeg):
    imported, command = pop_chanevent(
        sample_eeg,
        1,
        "oper",
        "X>0",
        "edge",
        "leading",
        "delchan",
        "off",
        "delevent",
        "off",
        return_com=True,
    )

    assert imported["nbchan"] == sample_eeg["nbchan"]
    assert imported["data"].shape == sample_eeg["data"].shape
    assert len(imported["event"]) > len(sample_eeg["event"])
    assert imported["urevent"]
    assert "pop_chanevent" in command


def test_pop_interp_reconstructs_selected_sample_channel_without_runtime_warnings(sample_eeg):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        interpolated = pop_interp(sample_eeg, [0], "spherical")

    assert interpolated["data"].shape == sample_eeg["data"].shape
    assert np.isfinite(interpolated["data"]).all()
    assert not np.array_equal(interpolated["data"][0], sample_eeg["data"][0])
    np.testing.assert_allclose(interpolated["data"][1:], sample_eeg["data"][1:])


def test_pop_clean_rawdata_all_criteria_off_preserves_sample_data_shape(sample_eeg):
    options = {
        "FlatlineCriterion": "off",
        "Highpass": "off",
        "ChannelCriterion": "off",
        "LineNoiseCriterion": "off",
        "BurstCriterion": "off",
        "WindowCriterion": "off",
    }

    cleaned, command = pop_clean_rawdata(sample_eeg, return_com=True, **options)

    assert cleaned["data"].shape == sample_eeg["data"].shape
    assert cleaned["nbchan"] == sample_eeg["nbchan"]
    assert cleaned["pnts"] == sample_eeg["pnts"]
    assert "'FlatlineCriterion', 'off'" in command


def test_clean_channels_removes_sample_bad_channels_without_warning_noise(sample_eeg, caplog):
    caplog.set_level("WARNING")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cleaned = clean_channels(sample_eeg)

    assert cleaned["data"].shape == (30, 30504)
    assert cleaned["nbchan"] == 30
    assert np.isfinite(cleaned["data"]).all()
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


def test_clean_windows_removes_sample_bad_periods_without_warning_noise(sample_eeg, caplog):
    caplog.set_level("WARNING")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cleaned, sample_mask = clean_windows(sample_eeg)

    assert cleaned["data"].shape == (32, 26242)
    assert cleaned["pnts"] == int(np.count_nonzero(sample_mask))
    assert sample_mask.shape == (30504,)
    assert np.isfinite(cleaned["data"]).all()
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


def test_clean_asr_uses_sample_calibration_windows_without_warning_noise(sample_eeg, caplog):
    caplog.set_level("WARNING")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cleaned = clean_asr(sample_eeg)

    assert cleaned["data"].shape == sample_eeg["data"].shape
    assert cleaned["nbchan"] == sample_eeg["nbchan"]
    assert np.isfinite(cleaned["data"]).all()
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


def test_clean_artifacts_default_runs_on_sample_data_without_warning_noise(sample_eeg, caplog):
    caplog.set_level("WARNING")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cleaned, hp, bur, removed_channels = clean_artifacts(sample_eeg)

    assert cleaned["data"].shape == (30, 30504)
    assert hp["data"].shape[1] == 30504
    assert bur["data"].shape == (30, 30504)
    assert removed_channels.shape == (32,)
    assert int(np.count_nonzero(removed_channels)) == 2
    assert np.isfinite(cleaned["data"]).all()
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


def test_pop_clean_rawdata_gui_defaults_run_on_sample_data_without_warning_noise(sample_eeg, caplog):
    class Renderer:
        def run(self, spec, initial_values=None):
            return {control.tag: control.value for control in spec.controls if control.tag is not None}

    caplog.set_level("WARNING")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cleaned, command = pop_clean_rawdata(sample_eeg, gui=True, renderer=Renderer(), return_com=True)

    assert cleaned["nbchan"] == 30
    assert cleaned["pnts"] < 30504
    assert np.isfinite(cleaned["data"]).all()
    assert "'BurstRejection', 'on'" in command
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


def test_pop_clean_rawdata_riemannian_asr_runs_on_sample_data_without_warning_noise(sample_eeg, caplog):
    caplog.set_level("WARNING")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cleaned, command = pop_clean_rawdata(
            sample_eeg,
            return_com=True,
            Distance="Riemannian",
            BurstCriterion=20,
            BurstRejection="off",
            WindowCriterion="off",
            Highpass="off",
            ChannelCriterion="off",
            LineNoiseCriterion="off",
            FlatlineCriterion="off",
        )

    assert cleaned["data"].shape == sample_eeg["data"].shape
    assert np.isfinite(cleaned["data"]).all()
    assert "'Distance', 'Riemannian'" in command
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


def test_pop_rmbase_zeroes_selected_sample_baseline_channels(sample_eeg):
    baseline = pop_rmbase(sample_eeg, pointrange=range(1, 21), chanlist=[1, 2])

    assert baseline["data"].shape == sample_eeg["data"].shape
    assert np.nanmean(baseline["data"][0, :20]) == pytest.approx(0, abs=1e-5)
    assert np.nanmean(baseline["data"][1, :20]) == pytest.approx(0, abs=1e-5)
    np.testing.assert_allclose(baseline["data"][2], sample_eeg["data"][2])


def test_pop_runica_one_step_returns_finite_sample_decomposition(sample_eeg_with_ica):
    assert sample_eeg_with_ica["icaweights"].shape == (32, 32)
    assert sample_eeg_with_ica["icasphere"].shape == (32, 32)
    assert sample_eeg_with_ica["icawinv"].shape == (32, 32)
    assert sample_eeg_with_ica["icaact"].shape == (32, 30504, 1)
    assert np.isfinite(sample_eeg_with_ica["icaweights"]).all()
    assert np.isfinite(sample_eeg_with_ica["icasphere"]).all()
    assert np.isfinite(sample_eeg_with_ica["icawinv"]).all()
    assert np.isfinite(sample_eeg_with_ica["icaact"]).all()


def test_eeg_runica_one_step_matches_pop_runica_sample_output_contract(sample_eeg):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = eeg_runica(sample_eeg, extended=1, maxsteps=1, verbose=False)

    assert out["icaweights"].shape == (32, 32)
    assert out["icasphere"].shape == (32, 32)
    assert out["icachansind"].tolist() == list(range(32))
    assert np.isfinite(out["icaact"]).all()


def test_pop_subcomp_removes_requested_sample_component(sample_eeg_with_ica):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out, command = pop_subcomp(sample_eeg_with_ica, [1], return_com=True)

    assert out["data"].shape == sample_eeg_with_ica["data"].shape
    assert out["icaweights"].shape == (31, 32)
    assert out["icawinv"].shape == (32, 31)
    assert np.isfinite(out["data"]).all()
    assert command == "EEG = pop_subcomp(EEG, [1], 0);"


def test_pop_icflag_flags_sample_components_with_iclabel_probabilities(sample_eeg_with_ica):
    eeg = copy.deepcopy(sample_eeg_with_ica)
    classes = np.zeros((eeg["icaweights"].shape[0], 7), dtype=float)
    classes[:, 0] = 0.8
    classes[0, 1] = 0.95
    classes[1, 2] = 0.96
    eeg.setdefault("etc", {})["ic_classification"] = {
        "ICLabel": {
            "classifications": classes,
        }
    }

    out, command = pop_icflag(eeg, DEFAULT_ICFLAG_THRESHOLDS, return_com=True)

    np.testing.assert_array_equal(out["reject"]["gcompreject"][:3], [1, 1, 0])
    assert "pop_icflag" in command


def test_pop_expica_exports_sample_ica_weights(sample_eeg_with_ica, tmp_path):
    weights_file = tmp_path / "sample_weights.tsv"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        command = pop_expica(sample_eeg_with_ica, weights_file)

    exported = np.loadtxt(weights_file, delimiter="\t")
    assert exported.shape == (32, 32)
    assert np.isfinite(exported).all()
    assert "pop_expica" in command


def test_pop_export_writes_sample_data_table(tmp_path, sample_eeg):
    output_file = tmp_path / "sample_export.tsv"

    command = pop_export(sample_eeg, output_file, "transpose", "on")

    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("Time\tFPz\tEOG1\tF3")
    assert len(lines) == sample_eeg["pnts"] + 1
    assert "pop_export" in command


def test_pop_export_writes_sample_ica_activity_when_icaact_is_missing(sample_eeg_with_ica, tmp_path):
    output_file = tmp_path / "sample_ica.tsv"
    eeg = copy.deepcopy(sample_eeg_with_ica)
    eeg["icaact"] = np.array([])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        command = pop_export(eeg, output_file, "ica", "on", "transpose", "on")

    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("Time\t")
    assert len(lines) == eeg["pnts"] + 1
    assert "pop_export" in command


def test_pop_expevents_writes_all_sample_events(tmp_path, sample_eeg):
    output_file = tmp_path / "sample_events.tsv"

    command = pop_expevents(sample_eeg, output_file)

    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "latency\tposition\ttype\turevent"
    assert len(lines) == len(sample_eeg["event"]) + 1
    assert "square" in lines[1]
    assert "pop_expevents" in command


def test_pop_importdata_loads_exported_sample_slice(tmp_path, sample_eeg):
    data_file = tmp_path / "sample_slice.tsv"
    np.savetxt(data_file, sample_eeg["data"][:2, :12], delimiter="\t")

    imported, command = pop_importdata("data", data_file, "srate", sample_eeg["srate"], return_com=True)

    assert imported["data"].shape == (2, 12)
    assert imported["srate"] == sample_eeg["srate"]
    np.testing.assert_allclose(imported["data"], sample_eeg["data"][:2, :12])
    assert "pop_importdata" in command


def test_pop_importevent_replaces_sample_events_from_table(tmp_path, sample_eeg):
    events_file = tmp_path / "events.tsv"
    events_file.write_text("type\tlatency\tduration\nnewstim\t10\t0\n", encoding="utf-8")

    imported, command = pop_importevent(sample_eeg, "event", events_file, "timeunit", np.nan, return_com=True)

    assert len(imported["event"]) == 1
    assert imported["event"][0]["type"] == "newstim"
    assert imported["event"][0]["latency"] == 10
    assert len(imported["urevent"]) == 1
    assert "pop_importevent" in command


def test_pop_importerplab_imports_sample_event_table(tmp_path, sample_eeg):
    event_file = tmp_path / "erplab_events.txt"
    event_file.write_text("12 erpstim\n", encoding="utf-8")

    imported, command = pop_importerplab(sample_eeg, event_file, return_com=True)

    assert imported["event"][0]["type"] == "erpstim"
    assert imported["event"][0]["latency"] == 12
    assert "pop_importerplab" in command


def test_pop_importpres_imports_sample_event_table(tmp_path, sample_eeg):
    event_file = tmp_path / "presentation.log"
    event_file.write_text("prestim 14\n", encoding="utf-8")

    imported, command = pop_importpres(sample_eeg, event_file, return_com=True)

    assert imported["event"][0]["type"] == "prestim"
    assert imported["event"][0]["latency"] == 14
    assert "pop_importpres" in command


def test_pop_importepoch_updates_sample_epoch_metadata(tmp_path, sample_eeg):
    epoched, _command = pop_epoch(sample_eeg, ["square"], [-0.1, 0.2], return_com=True)
    epoch_file = tmp_path / "epoch_metadata.tsv"
    rows = ["condition"] + [f"square_{index}" for index in range(1, int(epoched["trials"]) + 1)]
    epoch_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    imported, command = pop_importepoch(epoched, epoch_file, return_com=True)

    assert len(imported["epoch"]) == epoched["trials"]
    assert imported["epoch"][0]["condition"] == "square_1"
    assert imported["event"].size == 0
    assert "pop_importepoch" in command


def test_pop_saveset_roundtrips_sample_dataset(tmp_path, sample_eeg):
    output_file = tmp_path / "roundtrip.set"

    pop_saveset(sample_eeg, str(output_file))
    loaded = pop_loadset(str(output_file))

    assert loaded["data"].shape == sample_eeg["data"].shape
    assert loaded["setname"] == sample_eeg["setname"]
    assert len(loaded["event"]) == len(sample_eeg["event"])
    assert np.issubdtype(np.asarray(loaded["icachansind"]).dtype, np.integer)


def test_pop_study_records_sample_dataset_info(sample_eeg):
    study, alleeg, command = pop_study(None, [sample_eeg], name="Sample study", return_com=True)

    assert study["name"] == "Sample study"
    assert study["datasetinfo"][0]["index"] == 1
    assert study["datasetinfo"][0]["setname"] == sample_eeg["setname"]
    assert alleeg[0]["data"].shape == sample_eeg["data"].shape
    assert command.startswith("STUDY, ALLEEG = pop_study(")


def test_pop_studyerp_marks_sample_study_as_erp(sample_eeg):
    study, alleeg, command = pop_studyerp([sample_eeg], return_com=True)

    assert study["name"] == "Simple ERP STUDY"
    assert study["design"][0]["name"] == "ERP"
    assert alleeg[0]["data"].shape == sample_eeg["data"].shape
    assert command == "STUDY, ALLEEG = pop_studyerp(ALLEEG)"


def test_pop_savestudy_and_pop_loadstudy_roundtrip_sample_study(tmp_path, sample_eeg):
    study, _alleeg = pop_study(None, [sample_eeg], name="Sample study")

    saved, save_command = pop_savestudy(study, sample_eeg, tmp_path / "sample.study", return_com=True)
    loaded, loaded_alleeg, load_command = pop_loadstudy(tmp_path / "sample.study", return_com=True)

    assert saved["filename"] == "sample.study"
    assert loaded["name"] == "Sample study"
    assert loaded["datasetinfo"][0]["setname"] == sample_eeg["setname"]
    assert loaded_alleeg[0]["data"].shape == sample_eeg["data"].shape
    assert "pop_savestudy" in save_command
    assert "pop_loadstudy" in load_command


def test_pop_precomp_and_chanplot_work_on_sample_study(sample_eeg):
    study, alleeg = pop_study(None, [sample_eeg], name="Sample study")

    study, alleeg, precomp_command = pop_precomp(study, alleeg, "channels", spec="on", return_com=True)
    study, plot_command, figure = pop_chanplot(study, alleeg, channels=[1], measure="spec", return_com=True)

    assert study["changrp"][0]["specdata"]
    assert precomp_command.startswith("STUDY, ALLEEG = pop_precomp(")
    assert "measure='spec'" in plot_command
    plt.close(figure)


def test_pop_studywizard_builds_study_from_saved_sample_set(tmp_path, sample_eeg):
    set_file = tmp_path / "sample.set"
    pop_saveset(sample_eeg, str(set_file))

    study, alleeg, command = pop_studywizard([str(set_file)], return_com=True)

    assert study["datasetinfo"][0]["index"] == 1
    assert study["datasetinfo"][0]["setname"] == sample_eeg["setname"]
    assert alleeg[0]["data"].shape == sample_eeg["data"].shape
    assert command.startswith("STUDY, ALLEEG = pop_studywizard(")


def test_pop_saveh_writes_sample_history_commands(tmp_path):
    command = pop_saveh(
        ["EEG = pop_fileio('sample_data/eeglab_data.set');", "EEG = pop_reref( EEG, []);"],
        "sample_hist.m",
        tmp_path,
    )

    lines = (tmp_path / "sample_hist.m").read_text(encoding="utf-8").splitlines()
    assert "EEG = pop_fileio('sample_data/eeglab_data.set');" in lines
    assert "EEG = pop_reref( EEG, []);" in lines
    assert lines.index("EEG = pop_fileio('sample_data/eeglab_data.set');") < lines.index("EEG = pop_reref( EEG, []);")
    assert "pop_saveh" in command


def test_pop_runscript_can_modify_sample_workspace_namespace(sample_eeg, tmp_path):
    script_file = tmp_path / "rename_sample.py"
    namespace = {"EEG": sample_eeg}
    script_file.write_text("EEG['setname'] = 'scripted sample'\n", encoding="utf-8")

    command = pop_runscript(script_file, namespace)

    assert namespace["EEG"]["setname"] == "scripted sample"
    assert "pop_runscript" in command


def test_pop_writeeeg_exports_sample_through_mne_raw(monkeypatch, tmp_path, sample_eeg):
    captured = {}

    def fake_export_raw(path, raw, *, fmt, overwrite):
        captured.update({"path": path, "nchan": raw.info["nchan"], "fmt": fmt, "overwrite": overwrite})

    monkeypatch.setattr("eegprep.functions.popfunc.pop_writeeeg.export_raw", fake_export_raw)

    command = pop_writeeeg(sample_eeg, tmp_path / "sample.edf")

    assert captured == {
        "path": str(tmp_path / "sample.edf"),
        "nchan": 32,
        "fmt": "edf",
        "overwrite": True,
    }
    assert "pop_writeeeg" in command


def test_pop_exportbids_writes_valid_bids_dataset_from_sample(tmp_path, sample_eeg):
    root, command = pop_exportbids(sample_eeg, tmp_path / "bids", return_com=True)
    report = validate_bids(root)

    assert report["errors"] == []
    assert (Path(root) / "dataset_description.json").exists()
    assert list((Path(root) / "sub-01" / "eeg").glob("*_eeg.set"))
    assert "pop_exportbids" in command


def test_pop_importbids_reads_sample_bids_export(tmp_path, sample_eeg):
    root, _command = pop_exportbids(sample_eeg, tmp_path / "bids", return_com=True)

    imported, command = pop_importbids(root, return_com=True)

    assert imported["data"].shape == sample_eeg["data"].shape
    assert imported["event"][0]["type"] == sample_eeg["event"][0]["type"]
    assert "pop_importbids" in command


def test_pop_load_frombids_reads_sample_bids_eeg_file(tmp_path, sample_eeg):
    root, _command = pop_exportbids(sample_eeg, tmp_path / "bids", return_com=True)
    eeg_file = next((Path(root) / "sub-01" / "eeg").glob("*_eeg.set"))

    imported = pop_load_frombids(str(eeg_file), verbose=False)

    assert imported["data"].shape == sample_eeg["data"].shape
    assert imported["event"][0]["type"] == sample_eeg["event"][0]["type"]


def test_pop_iclabel_reports_missing_ica_for_sample_without_decomposition(sample_eeg):
    with pytest.raises(ValueError, match="requires an ICA decomposition"):
        pop_iclabel(sample_eeg, "default")


def test_pop_chansel_formats_sample_channel_display_and_selection(sample_eeg):
    display = pop_chansel_display_values(sample_eeg, withindex="on")
    selection = pop_chansel_selected_string(sample_eeg, ["FPz", "F3"])

    assert display[:3] == ["1  -  FPz", "2  -  EOG1", "3  -  F3"]
    assert selection == "FPz F3"


def test_eeg_store_retrieve_newset_and_delset_use_sample_dataset_indices(sample_eeg):
    alleeg, current, current_set = eeg_store(None, sample_eeg, 0)
    retrieved, alleeg, retrieved_set = eeg_retrieve(alleeg, 1)
    alleeg, current, current_set, newset_command = pop_newset(
        alleeg, retrieved, current_set, "setname", "Stored sample", "overwrite", "on"
    )
    alleeg, del_command = pop_delset(alleeg, 1)

    assert current["data"].shape == sample_eeg["data"].shape
    assert current_set == 1
    assert retrieved["data"].shape == sample_eeg["data"].shape
    assert retrieved_set == 1
    assert newset_command == (
        "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'Stored sample', 'overwrite', 'on');"
    )
    assert alleeg == []
    assert del_command == "ALLEEG = pop_delset( ALLEEG, [1] );"


def test_eeg_emptyset_and_eegh_provide_sample_workflow_defaults():
    history = []
    empty = eeg_emptyset()
    command = eegh("EEG = pop_fileio('sample_data/eeglab_data.set');", history)

    assert empty["data"].size == 0
    assert empty["saved"] == "yes"
    assert command == "EEG = pop_fileio('sample_data/eeglab_data.set');"
    assert history == ["EEG = pop_fileio('sample_data/eeglab_data.set');"]


def test_pop_editoptions_and_bids_metadata_helpers_are_history_recording(sample_eeg):
    options = {"option_allmenus": 0}

    edit_command = pop_editoptions(options, option_allmenus=True)
    eeg, task_command = pop_taskinfo(sample_eeg, TaskName="sample-task")
    eeg, participant_command = pop_participantinfo(eeg, participant_id="sub-01")
    eeg, event_command = pop_eventinfo(eeg, trial_type="square")

    assert options["option_allmenus"] == 1
    assert eeg["etc"]["bids"]["task"]["TaskName"] == "sample-task"
    assert eeg["etc"]["bids"]["participant"]["participant_id"] == "sub-01"
    assert eeg["etc"]["bids"]["event"]["trial_type"] == "square"
    assert edit_command == "LASTCOM = pop_editoptions();"
    assert "pop_taskinfo" in task_command
    assert "pop_participantinfo" in participant_command
    assert "pop_eventinfo" in event_command
