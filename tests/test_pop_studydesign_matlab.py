"""Compare pop_studydesign results with EEGLAB on the MATLAB batch CI job.

The GitHub Actions MATLAB job serves ``matlab.engine`` through
``tests/matlab_batch_engine.py``. This test skips when
``EEGPREP_SKIP_MATLAB=1`` and otherwise builds the same designs in EEGLAB by
replaying the dialog command history.
"""

from __future__ import annotations

import ast
import os
from typing import Any

import numpy as np
import pytest

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.pop_studydesign import (
    apply_studydesign_add,
    apply_studydesign_delete,
    apply_studydesign_new_variable,
    apply_studydesign_rename,
    apply_studydesign_select,
    pop_studydesign,
    pop_studydesign_dialog_spec,
)
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.studyfunc.std_maketrialinfo import std_maketrialinfo
from tests.fixtures import create_test_eeg

_VARIABLE_FIELDS = ("label", "value", "vartype", "pairing", "level")


class _Renderer:
    def __init__(self, result):
        self.result = result

    def run(self, spec, initial_values=None):
        return self.result


def _dataset(path, subject, condition):
    eeg = create_test_eeg(n_channels=2, n_samples=8, srate=100.0, n_trials=2)
    for trial, event in enumerate(eeg["event"], start=1):
        event["epoch"] = trial
        event["type"] = "stimulus"
        event["rt"] = 300.0 if trial == 1 else 450.0
    eeg.update(
        {
            "setname": f"{subject}-{condition}",
            "subject": subject,
            "condition": condition,
            "session": 1,
            "run": 1,
            "group": "",
        }
    )
    pop_saveset(eeg, str(path))
    return eeg


def _editor_state(study, alleeg):
    return controls_by_tag(pop_studydesign_dialog_spec(study, alleeg))["new_design"].callback.params


def _commit(study, alleeg, state):
    payload = {"designind": state, "subjects": state.get("subject_text") or "", "resave": False}
    return pop_studydesign(study, alleeg, gui=True, renderer=_Renderer(payload), return_com=True)


def _append_variable(state, result):
    updated, warning = apply_studydesign_new_variable(state["variables"], state["varlist"], result)
    assert warning is None
    assert updated is not state["variables"]
    design = state["designs"][int(state["design_index"]) - 1]
    design["variable"] = updated
    state["variables"] = updated
    return updated


def _matlab_literal(value: Any) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)
    if isinstance(value, (list, tuple)):
        return "{" + ",".join(_matlab_literal(item) for item in value) + "}"
    raise AssertionError(f"cannot translate history value {value!r}")


def _matlab_history(command: str) -> list[str]:
    """Translate one EEGPrep history block into EEGLAB statements."""
    statements = []
    for line in command.splitlines():
        if not line.strip():
            continue
        statement = ast.parse(line).body[0]
        if isinstance(statement, ast.Delete):
            index = int(ast.literal_eval(statement.targets[0].slice)) + 1
            # pop_studydesign records this text and evals it with index set.
            statements.append(f"index = {index}; com = 'STUDY.design(index) = [];'; eval(com);")
            continue
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            raise AssertionError(f"unsupported history line: {line}")
        call = statement.value
        name = call.func.id
        if name == "std_maketrialinfo":
            statements.append("[STUDY, alltrialinfo] = std_maketrialinfo(STUDY, ALLEEG);")
            continue
        if name == "std_selectdesign":
            selected = int(ast.literal_eval(call.args[2]))
            statements.append(f"com = 'STUDY = std_selectdesign(STUDY, ALLEEG, {selected});'; eval(com);")
            continue
        if name != "std_makedesign":
            raise AssertionError(f"unsupported history call: {name}")
        design_index = int(ast.literal_eval(call.args[2]))
        pairs = ",".join(
            f"'{keyword.arg}',{_matlab_literal(ast.literal_eval(keyword.value))}" for keyword in call.keywords
        )
        statements.append(f"[STUDY, com] = std_makedesign(STUDY, ALLEEG, {design_index}, {pairs});")
    return statements


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            if value.shape == (1, 1):
                return {name: _plain(value[name][0, 0]) for name in value.dtype.names}
            return [
                _plain({name: value[name][index] for name in value.dtype.names}) for index in np.ndindex(value.shape)
            ]
        if value.size == 0:
            return []
        if value.dtype == object or value.size > 1:
            return [_plain(item) for item in value.ravel(order="F")]
        return _plain(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _as_list(value: Any) -> list[Any]:
    value = _plain(value)
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return [value]


def _struct_rows(value: Any) -> list[dict[str, Any]]:
    """Return one dict per MATLAB struct, including a 1xN struct array."""
    value = _plain(value)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    if len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if all(isinstance(item, dict) for item in value):
        return value
    return []


def _canonical_value(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def _canonical_designs(designs: Any) -> list[dict[str, Any]]:
    output = []
    for design in _struct_rows(designs):
        variables = []
        for variable in _struct_rows(design.get("variable")):
            if not isinstance(variable, dict) or not variable.get("label"):
                continue
            extra = set(variable) - set(_VARIABLE_FIELDS) - {"levelindex"}
            if extra:
                raise AssertionError(f"unexpected STUDY.design variable fields: {sorted(extra)}")
            value = _canonical_value(variable.get("value"))
            if not isinstance(value, list):
                value = [] if value in (None, "") else [value]
            variables.append(
                {
                    "label": str(variable.get("label") or ""),
                    "value": value,
                    "vartype": str(variable.get("vartype") or ""),
                    "pairing": str(variable.get("pairing") or ""),
                    "level": str(variable.get("level") or ""),
                }
            )
        cases = design.get("cases") or {}
        if not isinstance(cases, dict):
            cases = {}
        include = _canonical_value(design.get("include"))
        output.append(
            {
                "name": str(design.get("name") or ""),
                "filepath": str(design.get("filepath") or ""),
                "variable": variables,
                "cases": {
                    "label": str(cases.get("label") or ""),
                    "value": [str(item) for item in _as_list(cases.get("value"))],
                },
                "include": include if isinstance(include, list) else [],
            }
        )
    return output


def _matlab_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


@pytest.mark.skipif(os.environ.get("EEGPREP_SKIP_MATLAB") == "1", reason="MATLAB not available")
def test_pop_studydesign_history_matches_eeglab_designs(tmp_path):
    datasets = []
    for index, (subject, condition) in enumerate((("S01", "standard"), ("S02", "target"), ("S01", "oddball")), start=1):
        datasets.append(_dataset(tmp_path / f"set{index}.set", subject, condition))
    study, alleeg = pop_study(None, datasets, name="Study")
    study, _rows = std_maketrialinfo(study, alleeg)

    state = _editor_state(study, alleeg)
    apply_studydesign_add(state)
    apply_studydesign_rename(state, "Factors")
    condition_index = state["varlist"]["factors"].index("condition")
    condition_values = state["varlist"]["factorvals"][condition_index]
    combined = ["standard", "target"]
    condition_values.append(combined)
    _append_variable(
        state,
        {
            "lbfact0": condition_index + 1,
            "vartype": 1,
            "lbval0": [condition_values.index("oddball") + 1, condition_values.index(combined) + 1],
        },
    )
    rt_index = state["varlist"]["factors"].index("rt")
    _append_variable(state, {"lbfact0": rt_index + 1, "vartype": 2, "lbval0": []})
    apply_studydesign_add(state)
    apply_studydesign_add(state)
    apply_studydesign_rename(state, "Discarded")
    created, alleeg, create_command = _commit(study, alleeg, state)

    state = _editor_state(created, alleeg)
    apply_studydesign_select(state, 4)
    assert apply_studydesign_delete(state) is None
    edited, _alleeg, delete_command = _commit(created, alleeg, state)

    assert "name='Factors'" in create_command
    assert "['standard', 'target']" in create_command
    assert "variable2='rt'" in create_command
    assert "vartype2='continuous'" in create_command
    assert "name='Design 3'" in create_command
    assert "variable1='none'" in create_command
    assert "name='Discarded'" in create_command
    assert "del STUDY['design'][3]" in delete_command
    assert [design["name"] for design in edited["design"]] == ["STUDY.design 1", "Factors", "Design 3"]

    engine = get_eeglab("MAT", auto_file_roundtrip=False)
    loads = []
    for index, (eeg, condition) in enumerate(zip(datasets, ("standard", "target", "oddball")), start=1):
        path = _matlab_quote(str(tmp_path / f"set{index}.set"))
        subject = _matlab_quote(eeg["subject"])
        loads.append(
            "{'index',"
            f"{index},"
            "'load',"
            f"{path},"
            "'subject',"
            f"{subject},"
            "'condition',"
            f"{_matlab_quote(condition)},"
            "'session',1,'run',1}"
        )
    try:
        engine.eval("clear STUDY ALLEEG alltrialinfo com index;", nargout=0)
        engine.eval(
            "[STUDY, ALLEEG] = std_editset([], [], 'name', 'Study', 'updatedat', 'on', 'savedat', 'off', "
            f"'commands', {{{','.join(loads)}}});",
            nargout=0,
        )
        engine.eval("[STUDY, ALLEEG] = std_checkset(STUDY, ALLEEG);", nargout=0)
        engine.eval("[STUDY, alltrialinfo] = std_maketrialinfo(STUDY, ALLEEG);", nargout=0)
        for statement in [*_matlab_history(create_command), *_matlab_history(delete_command)]:
            engine.eval(statement, nargout=0)
        matlab_designs = _canonical_designs(engine.eval("STUDY.design", nargout=1))
        matlab_current = int(engine.eval("STUDY.currentdesign", nargout=1))
    finally:
        engine.eval("clear STUDY ALLEEG alltrialinfo com index;", nargout=0)

    python_designs = _canonical_designs(edited["design"])
    assert matlab_designs == python_designs
    assert matlab_current == int(edited["currentdesign"])
