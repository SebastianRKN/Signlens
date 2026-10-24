from signlens.composer import TextComposer
from signlens.panel import ControlPanel
from test_demo_safety import two_hand_runtime, marked_hand
import numpy as np


def hold(composer, label, start, count=15):
    for i in range(count):
        composer.update(label, start + i * .05)


def test_multiple_letters_spaces_and_period_form_sentence():
    c = TextComposer()
    c.set_active(True)
    for i, letter in enumerate("HOLA"):
        hold(c, letter, i)
    c.space()
    for i, letter in enumerate("AMIGO"):
        hold(c, letter, 4 + i)
    c.period()
    assert c.text == "HOLA AMIGO. "


def test_double_letter_requires_release_and_short_dropout_does_not_repeat():
    c = TextComposer()
    c.set_active(True)
    hold(c, "L", 0)
    c.update(None, .75, present=False)
    hold(c, "L", .8)
    assert c.text == "L"
    for i in range(15):
        c.update(None, 2 + i * .05, present=False)
    hold(c, "L", 3)
    assert c.text == "LL"


def test_uncertainty_with_visible_hand_does_not_rearm_same_letter():
    c = TextComposer()
    c.set_active(True)
    hold(c, "A", 0)
    for i in range(20):
        c.update(None, 1 + i * .05, present=True)
    hold(c, "A", 3)
    assert c.text == "A"


def test_pause_stall_and_fast_changes_do_not_append():
    c = TextComposer()
    hold(c, "A", 0)
    assert c.text == ""
    c.set_active(True)
    c.update("A", 1)
    c.update("A", 10)
    assert c.text == ""
    for i in range(20):
        c.update("A" if i % 2 else "B", 11 + i * .05)
    assert c.text == ""


def test_edits_undo_and_maximum_length():
    c = TextComposer(max_chars=3)
    c.set_active(True)
    for i, letter in enumerate("ABCD"):
        hold(c, letter, i)
    assert c.text == "ABC"
    c.backspace()
    c.undo()
    assert c.text == "ABC"
    c.clear()
    c.undo()
    assert c.text == "ABC"
    c.clear()
    c.space()
    c.period()
    assert c.text == ""


def test_runtime_both_hands_write_automatically(tmp_path):
    app = two_hand_runtime(tmp_path)
    for i in range(25):
        app.process_hands({"Left": marked_hand(1), "Right": marked_hand(2)}, now=i * .05)
    assert app.panel.results["Left"].confirmed == "A"
    assert app.panel.results["Right"].confirmed == "B"
    assert app.composer.text == "AB"
    app.handle("hand")  # The capture selector cannot change subtitle input.
    for i in range(25):
        app.process_hands({"Left": marked_hand(1), "Right": marked_hand(2)}, now=2 + i * .05)
    assert app.composer.text == "AB"
    app.handle("view_data")
    assert not app.composer.active
    app.handle("capture")
    assert not app.composer.active


def test_write_hud_has_no_buttons_and_space_never_starts_capture():
    p = ControlPanel(["A", "B"])
    p.draw(np.zeros((480, 640, 3), dtype=np.uint8))
    actions = {action for _, action in p._buttons}
    assert not actions
    p.on_key(ord(" "))
    assert p.pop_actions() == ["space"]
    p.view = "data"
    p.on_key(ord(" "))
    assert p.pop_actions() == ["capture"]
