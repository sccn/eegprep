"""Ports of current EEGLAB STUDY plotting-parameter tests."""

from __future__ import annotations

import ast

import eegprep
from eegprep.functions.studyfunc.pop_dipparams import pop_dipparams
from eegprep.functions.studyfunc.pop_erpimparams import pop_erpimparams
from eegprep.functions.studyfunc.pop_erpparams import pop_erpparams
from eegprep.functions.studyfunc.pop_erspparams import pop_erspparams
from eegprep.functions.studyfunc.pop_specparams import pop_specparams
from eegprep.functions.studyfunc.pop_statparams import pop_statparams
from tests.eeglab_tests import eeglab_test


STUDYFUNC_ROOT = "unittesting_studyfunc"


def _reference(wrapper: str, test: str):
    source = f"{STUDYFUNC_ROOT}/{wrapper}/studyfunc_{wrapper}_wrapperTest.m"
    return eeglab_test(source, test)


def _study_with_cached_measures() -> dict:
    fields = {
        "erpdata": [[1.0]],
        "erptimes": [0.0],
        "specdata": [[2.0]],
        "specfreqs": [10.0],
        "erspdata": [[[3.0]]],
        "ersptimes": [0.0],
        "erspfreqs": [10.0],
        "erspbase": [1.0],
        "itcdata": [[[0.5]]],
        "itctimes": [0.0],
        "itcfreqs": [10.0],
        "erpimdata": [[4.0]],
        "erpimtimes": [0.0],
        "erpimtrials": [1],
        "erpimevents": ["stim"],
    }
    return {
        "name": "Generated parameter study",
        "etc": {},
        "cluster": [{"name": "ParentCluster", **fields}],
        "changrp": [{"name": "Cz", **fields}],
    }


@_reference("pop_dipparams", "test_test_pop_dipparams")
def test_pop_dipparams_stores_every_current_eeglab_test_option():
    study = _study_with_cached_measures()
    cases = (
        ("axistight", "on"),
        ("axistight", "off"),
        ("projimg", "on"),
        ("projimg", "off"),
        ("projlines", "on"),
        ("projlines", "off"),
        ("density", "on"),
        ("density", "off"),
        ("centrline", "on"),
        ("centrline", "off"),
    )

    for key, value in cases:
        study = pop_dipparams(study, key, value)
        assert study["etc"]["dipparams"][key] == value

    assert study["etc"]["dipparams"] == {
        "axistight": "off",
        "projimg": "off",
        "projlines": "off",
        "density": "off",
        "centrline": "off",
    }


@_reference("pop_erpimparams", "test_test_pop_erpimparams")
def test_pop_erpimparams_stores_tested_ranges_and_invalidates_erpimage_cache():
    study = _study_with_cached_measures()
    for key, value in (
        ("topotime", 100),
        ("topotime", [50, 100]),
        ("timerange", [0, 200]),
        ("timerange", [-100, 200]),
        ("colorlimits", [0, 200]),
    ):
        study = pop_erpimparams(study, key, value)
        assert study["etc"]["erpimparams"][key] == value

    for collection in (study["cluster"], study["changrp"]):
        assert not {"erpimdata", "erpimtimes", "erpimtrials", "erpimevents"}.intersection(collection[0])


@_reference("pop_erpparams", "test_test_pop_erpparams")
def test_pop_erpparams_stores_every_current_option_and_invalidates_erp_cache():
    study = _study_with_cached_measures()
    cases = (
        ("topotime", 100),
        ("topotime", [100, 200]),
        ("topotime", []),
        ("filter", 20),
        ("filter", []),
        ("timerange", [-100, 200]),
        ("timerange", []),
        ("ylim", [0, 20]),
        ("plotgroups", "together"),
        ("plotgroups", "apart"),
        ("plotconditions", "together"),
        ("plotconditions", "apart"),
        ("averagechan", "on"),
        ("averagechan", "off"),
    )

    for key, value in cases:
        study = pop_erpparams(study, key, value)
        assert study["etc"]["erpparams"][key] == value

    for collection in (study["cluster"], study["changrp"]):
        assert "erpdata" not in collection[0]
        assert "erptimes" not in collection[0]


@_reference("pop_erspparams", "test_test_pop_erspparams")
def test_pop_erspparams_stores_every_current_option_and_invalidates_time_frequency_cache():
    study = _study_with_cached_measures()
    cases = (
        ("timerange", [-100, 400]),
        ("freqrange", [2, 60]),
        ("ersplim", [1, 20]),
        ("itclim", [0, 1]),
        ("itclim", [0, 2]),
        ("topotime", 100),
        ("topotime", [100, 200]),
        ("topofreq", 10),
        ("topofreq", [8, 12]),
        ("subbaseline", "on"),
        ("subbaseline", "off"),
        ("maskdata", "on"),
        ("maskdata", "off"),
    )

    for key, value in cases:
        study = pop_erspparams(study, key, value)
        assert study["etc"]["erspparams"][key] == value

    for collection in (study["cluster"], study["changrp"]):
        assert not {"erspdata", "ersptimes", "erspfreqs", "erspbase", "itcdata", "itctimes", "itcfreqs"}.intersection(
            collection[0]
        )


@_reference("pop_specparams", "test_test_pop_specparams")
def test_pop_specparams_stores_every_current_option_and_invalidates_spectrum_cache():
    study = _study_with_cached_measures()
    cases = (
        ("topofreq", 10),
        ("topofreq", [8, 12]),
        ("freqrange", [2, 60]),
        ("ylim", [0, 20]),
        ("plotgroups", "together"),
        ("plotgroups", "apart"),
        ("plotconditions", "together"),
        ("plotconditions", "apart"),
        ("subtractsubjectmean", "on"),
        ("subtractsubjectmean", "off"),
        ("averagechan", "on"),
        ("averagechan", "off"),
    )

    for key, value in cases:
        study = pop_specparams(study, key, value)
        assert study["etc"]["specparams"][key] == value

    for collection in (study["cluster"], study["changrp"]):
        assert "specdata" not in collection[0]
        assert "specfreqs" not in collection[0]


@_reference("pop_statparams", "test_test_pop_statparams")
def test_pop_statparams_stores_common_eeglab_and_fieldtrip_namespaces():
    study = _study_with_cached_measures()
    cases = (
        ("groupstats", "on", "common", "groupstats"),
        ("groupstats", "off", "common", "groupstats"),
        ("condstats", "on", "common", "condstats"),
        ("condstats", "off", "common", "condstats"),
        ("singletrials", "on", "common", "singletrials"),
        ("singletrials", "off", "common", "singletrials"),
        ("mode", "eeglab", "common", "mode"),
        ("mode", "fieldtrip", "common", "mode"),
        ("method", "param", "eeglab", "method"),
        ("method", "perm", "eeglab", "method"),
        ("method", "bootstrap", "eeglab", "method"),
        ("naccu", 2000, "eeglab", "naccu"),
        ("alpha", 0.5, "eeglab", "alpha"),
        ("mcorrect", "fdr", "eeglab", "mcorrect"),
        ("mcorrect", "holms", "eeglab", "mcorrect"),
        ("mcorrect", "bonferoni", "eeglab", "mcorrect"),
        ("mcorrect", "none", "eeglab", "mcorrect"),
        ("fieldtripmethod", "analytic", "fieldtrip", "method"),
        ("fieldtripmethod", "montecarlo", "fieldtrip", "method"),
        ("fieldtripnaccu", 2000, "fieldtrip", "naccu"),
        ("fieldtripalpha", 0.5, "fieldtrip", "alpha"),
        ("fieldtripmcorrect", "cluster", "fieldtrip", "mcorrect"),
        ("fieldtripmcorrect", "max", "fieldtrip", "mcorrect"),
        ("fieldtripmcorrect", "fdr", "fieldtrip", "mcorrect"),
        ("fieldtripmcorrect", "holms", "fieldtrip", "mcorrect"),
        ("fieldtripmcorrect", "bonferoni", "fieldtrip", "mcorrect"),
        ("fieldtripmcorrect", "none", "fieldtrip", "mcorrect"),
        ("fieldtripclusterparam", [], "fieldtrip", "clusterparam"),
        ("fieldtripchannelneighbor", [], "fieldtrip", "channelneighbor"),
        ("fieldtripchannelneighborparam", [], "fieldtrip", "channelneighborparam"),
    )

    for option, value, namespace, stored_key in cases:
        study = pop_statparams(study, option, value)
        statistics = study["etc"]["statistics"]
        actual = statistics[stored_key] if namespace == "common" else statistics[namespace][stored_key]
        assert actual == value

    study, command = pop_statparams(study, "default", return_com=True)
    assert study["etc"]["statistics"]["fieldtrip"]["alpha"] == 0.5
    assert command == "STUDY = pop_statparams(STUDY)"
    ast.parse(command)


def test_study_parameter_functions_are_public_and_history_replays_empty_values():
    study = _study_with_cached_measures()

    updated, command = eegprep.pop_erpparams(study, timerange=[], return_com=True)
    namespace = {"STUDY": study, "pop_erpparams": pop_erpparams}
    exec(command, namespace)

    assert namespace["STUDY"]["etc"]["erpparams"] == updated["etc"]["erpparams"]
    assert "timerange=[]" in command
