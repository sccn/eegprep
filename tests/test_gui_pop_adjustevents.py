import unittest

from eegprep.guifunc.qt import QtDialogRenderer
from eegprep.guifunc.spec import controls_by_tag
from eegprep.popfunc.pop_adjustevents import pop_adjustevents_dialog_spec


class PopAdjustEventsGuiSpecTests(unittest.TestCase):
    def test_dialog_spec_matches_eeglab_control_order(self):
        spec = pop_adjustevents_dialog_spec(250.0, ["stim", "resp"])

        self.assertEqual(spec.title, "Adjust event latencies - pop_adjustevents()")
        self.assertEqual(spec.function_name, "pop_adjustevents")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_adjustevents.m")
        self.assertEqual(spec.size, (858, 169))
        self.assertEqual(spec.geometry, ((1, 0.7, 0.5), (1, 0.7, 0.5), (1, 0.7, 0.5), 1))
        self.assertEqual(
            [(control.style, control.string, control.tag) for control in spec.controls],
            [
                ("text", "Event type(s) to adjust (all by default): ", None),
                ("edit", "", "events"),
                ("pushbutton", "...", "events_button"),
                ("text", "Add in milliseconds (can be negative)", None),
                ("edit", "", "edit_time"),
                ("spacer", "", None),
                ("text", "Or add in samples", None),
                ("edit", "", "edit_samples"),
                ("spacer", "", None),
                ("checkbox", "Force adjustment even when boundaries are present", "force"),
            ],
        )

    def test_dialog_callbacks_keep_matlab_metadata(self):
        spec = pop_adjustevents_dialog_spec(250.0, ["stim"])
        controls = controls_by_tag(spec)

        time_callback = controls["edit_time"].callback
        sample_callback = controls["edit_samples"].callback
        event_callback = controls["events_button"].callback

        self.assertEqual(time_callback.name, "sync_time_to_samples")
        self.assertEqual(time_callback.params["srate"], 250.0)
        self.assertIn("edit_samples", time_callback.matlab_callback)
        self.assertEqual(sample_callback.name, "sync_samples_to_time")
        self.assertIn("edit_time", sample_callback.matlab_callback)
        self.assertEqual(event_callback.params["event_types"], ("stim",))
        self.assertIn("pop_chansel", event_callback.matlab_callback)

    def test_numeric_sync_callback_formats_like_matlab_field_updates(self):
        class Field:
            def __init__(self, text):
                self._text = text

            def text(self):
                return self._text

            def setText(self, value):
                self._text = value

        source = Field("20")
        target = Field("")

        QtDialogRenderer._sync_numeric(source, target, 250.0)

        self.assertEqual(target.text(), "5000")


if __name__ == "__main__":
    unittest.main()
