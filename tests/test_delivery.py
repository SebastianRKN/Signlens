from signlens.composer import AutomaticTextComposer
from test_demo_safety import two_hand_runtime
import numpy as np
from signlens.config import Config
from signlens.hands import HandResolver
from test_camera_and_tracking import _det


def frames(c, start, count, left=None, right=None):
    for i in range(count):
        c.update_hands({'Left': left, 'Right': right}, start+i*.05,
                       {'Left': left is not None, 'Right': right is not None})


def test_sentence_both_hands_pause_and_edits():
    c = AutomaticTextComposer()
    frames(c, 0, 20, 'H', 'O')
    frames(c, 1, 20, 'L', 'A')
    frames(c, 2, 40)
    frames(c, 4, 20, 'M', 'U')
    frames(c, 5, 20, 'N', 'D')
    frames(c, 6, 20, 'O')
    assert c.text == 'HOLA MUNDO'
    c.backspace()
    assert c.text == 'HOLA MUND'
    c.undo()
    assert c.text == 'HOLA MUNDO'


def test_double_letters_and_no_repetition_from_held_sign():
    c = AutomaticTextComposer()
    frames(c, 0, 40, 'L', 'L')
    assert c.text == 'LL'
    frames(c, 2, 16)
    frames(c, 2.8, 20, 'L')
    assert c.text == 'LLL'


def test_camera_stall_does_not_insert_space():
    c = AutomaticTextComposer()
    frames(c, 0, 20, 'A')
    frames(c, 1, 2)
    frames(c, 10, 2)
    assert c.text == 'A'


def test_gesture_is_opt_in_and_return_from_data_is_automatic(tmp_path):
    app = two_hand_runtime(tmp_path)
    assert not app.gesture_enabled
    app.handle('gesture')
    assert app.gesture_enabled
    app.handle('gesture')
    assert not app.gesture_enabled
    app.handle('view_data')
    assert not app.composer.active
    app.handle('view_write')
    assert app.composer.active


def test_clear_pose_wrist_corrects_previously_swapped_hands():
    r = HandResolver(Config(smoothing=False))
    dets = [_det(.3, .5, 'Right'), _det(.7, .5, 'Left')]
    r.update(dets, None, 1, 0)
    pose = np.zeros((33, 4))
    pose[11] = [.3, .2, 0, 1]
    pose[12] = [.7, .2, 0, 1]
    pose[15] = [.3, .5, 0, 1]
    pose[16] = [.7, .5, 0, 1]
    out = r.update(dets, pose, 1, .1)
    assert out['Left'].pts[0, 0] < out['Right'].pts[0, 0]
