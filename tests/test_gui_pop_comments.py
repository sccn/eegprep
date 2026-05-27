import unittest

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_comments import pop_comments_dialog_spec


class PopCommentsGuiSpecTests(unittest.TestCase):
    def test_dialog_spec_matches_eeglab_comment_editor_shape(self):
        spec = pop_comments_dialog_spec("About this dataset", "old")

        self.assertEqual(spec.title, "Read/Enter text -- pop_comments()")
        self.assertEqual(spec.function_name, "pop_comments")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_comments.m")
        self.assertEqual(spec.size, (1394, 840))
        self.assertFalse(spec.show_help_button)
        self.assertEqual(spec.cancel_label, "CANCEL")
        self.assertEqual(spec.ok_label, "SAVE")
        self.assertEqual(spec.button_size, (150, 45))
        self.assertTrue(spec.cancel_first)
        self.assertEqual(
            [(control.style, control.string, control.tag) for control in spec.controls],
            [
                ("text", "About this dataset", None),
                ("textarea", "", "comments"),
            ],
        )
        self.assertEqual(controls_by_tag(spec)["comments"].value, "old")

    def test_renderer_reads_textarea_plain_text(self):
        class TextArea:
            def __init__(self):
                self._value = None

            def property(self, name):
                return self._value

            def toPlainText(self):
                return "edited\ncomments"

        self.assertEqual(QtDialogRenderer._read_widget(TextArea()), "edited\ncomments")


if __name__ == "__main__":
    unittest.main()
