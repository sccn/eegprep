"""Original STUDY parameter workflows and additional Python cache checks."""

from __future__ import annotations


import numpy as np

from eegprep.functions.studyfunc.pop_erpimparams import pop_erpimparams
from eegprep.functions.studyfunc.pop_erpparams import pop_erpparams
from eegprep.functions.studyfunc.pop_erspparams import pop_erspparams
from eegprep.functions.studyfunc.pop_specparams import pop_specparams
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


# Keep each source's option order: later calls operate on the modified STUDY.
_PARAMETER_CASES = {
    "dip": [
        (key, value) for key in ("axistight", "projimg", "projlines", "density", "centrline") for value in ("on", "off")
    ],
    "erpim": [
        ("topotime", 100),
        ("topotime", [50, 100]),
        ("timerange", [0, 200]),
        ("timerange", [-100, 200]),
        ("colorlimits", [0, 200]),
    ],
    "erp": [
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
    ],
    "ersp": [
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
    ],
    "spec": [
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
    ],
    "stat": [
        ("groupstats", "on"),
        ("groupstats", "off"),
        ("condstats", "on"),
        ("condstats", "off"),
        ("singletrials", "on"),
        ("singletrials", "off"),
        ("mode", "eeglab"),
        ("mode", "fieldtrip"),
        ("method", "param"),
        ("method", "perm"),
        ("method", "bootstrap"),
        ("naccu", 2000),
        ("alpha", 0.5),
        ("mcorrect", "fdr"),
        ("mcorrect", "holms"),
        ("mcorrect", "bonferoni"),
        ("mcorrect", "none"),
        ("fieldtripmethod", "analytic"),
        ("fieldtripmethod", "montecarlo"),
        ("fieldtripnaccu", 2000),
        ("fieldtripalpha", 0.5),
        ("fieldtripmcorrect", "cluster"),
        ("fieldtripmcorrect", "max"),
        ("fieldtripmcorrect", "fdr"),
        ("fieldtripmcorrect", "holms"),
        ("fieldtripmcorrect", "bonferoni"),
        ("fieldtripmcorrect", "none"),
        ("fieldtripclusterparam", []),
        ("fieldtripchannelneighbor", []),
        ("fieldtripchannelneighborparam", []),
    ],
}


def _reference_parameter_workflow(kind):
    function = f"pop_{kind}params"

    @_reference(function, f"test_test_{function}")
    def test(eeglab_backend, eeglab_sample_study):
        study, _alleeg = eeglab_sample_study
        failures = []
        for index, (option, value) in enumerate(_PARAMETER_CASES[kind]):
            if not isinstance(value, str):
                value = np.asarray(value, dtype=float)
                value = value.reshape(1, -1) if value.size else np.empty((0, 0))
            study = eeglab_backend(function, study, option, value)
            if kind == "stat":
                params = study["etc"]["statistics"]
                if index < 8:
                    actual = params[option]
                elif index < 17:
                    actual = params["eeglab"][option]
                else:
                    actual = params["fieldtrip"][option[9:]]
            else:
                actual = study["etc"][f"{kind}params"][option]
            # Source isequal compares exact dimensions/values, not tolerances.
            if not np.array_equal(actual, value):
                failures.append(option)
        if kind == "stat":
            eeglab_backend(function, study, "default", nargout=0)
        assert not failures, f"{function} failed original parameter checks: {failures}"

    return test


for _kind in _PARAMETER_CASES:
    globals()[f"test_reference_pop_{_kind}params"] = _reference_parameter_workflow(_kind)
