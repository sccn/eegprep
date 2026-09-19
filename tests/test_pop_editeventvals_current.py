"""Current eeglab_tests coverage for editing individual event values."""

import numpy as np

from eegprep.functions.popfunc.pop_editeventvals import pop_editeventvals
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.eeglab_tests import eeglab_test


@eeglab_test(
    "unittesting_popfunc/pop_editeventvals/popfunc_pop_editeventvals_wrapperTest.m", "test_test_pop_editeventvals"
)
def test_pop_editeventvals_current_suite_changes_multiple_fields_on_one_event():
    eeg = pop_loadset("sample_data/eeglab_data.set")

    output = pop_editeventvals(
        eeg,
        "changefield",
        [1, "latency", 1.1],
        "changefield",
        [1, "position", 2],
    )

    expected_latency = (1.1 - eeg["xmin"]) * eeg["srate"] + 1
    changed = [event for event in output["event"] if np.isclose(event["latency"], expected_latency)]
    assert len(changed) == 1
    assert changed[0]["position"] == 2
    assert output["urevent"][changed[0]["urevent"]]["latency"] == expected_latency
