"""Turn a stream of confirmed letters into editable text, without frame-by-frame repeats."""
from __future__ import annotations


class LetterInput:
    """One hand's event latch, independent of the other hand and of text editing."""

    def __init__(self, dwell=.35, release=.6):
        self.dwell, self.release = dwell, release
        self._candidate = None
        self._since = 0.0
        self._last_emitted = None
        self._missing_since = None
        self._last_time = None

    def reset_input(self, rearm=False):
        self._candidate, self._missing_since, self._last_time = None, None, None
        if rearm:
            self._last_emitted = None

    def update(self, label, now, present=True):
        # A stalled camera must not count as holding a letter steadily.
        if self._last_time is not None and (now < self._last_time or now - self._last_time > .4):
            self._candidate = None
            self._missing_since = None
        self._last_time = now
        if not present:
            if self._missing_since is None:
                self._missing_since = now
            if now - self._missing_since >= self.release:
                self._last_emitted = None
            self._candidate = None
            return None
        if self._missing_since is not None and now - self._missing_since >= self.release:
            self._last_emitted = None
        self._missing_since = None
        if not label or not label.isalpha() or len(label) > 3:
            self._candidate = None
            return None
        if label != self._candidate:
            self._candidate, self._since = label, now
        if label == self._last_emitted or now - self._since < self.dwell:
            return None
        self._last_emitted = label
        return label


class TextComposer:
    def __init__(self, dwell=.35, release=.6, max_chars=1000):
        self.text = ""
        self.active = False
        self.max_chars = max_chars
        self._history = []
        self._input = LetterInput(dwell, release)

    def reset_input(self, rearm=False):
        self._input.reset_input(rearm)

    def set_active(self, active):
        self.active = bool(active)
        self.reset_input()

    def update(self, label, now, present=True):
        if self.active:
            event = self._input.update(label, now, present)
            if event and self._replace(self.text + event):
                return event
        return None

    def _replace(self, value):
        if value == self.text or len(value) > self.max_chars:
            return False
        self._history.append(self.text)
        self._history = self._history[-100:]
        self.text = value
        return True

    def space(self):
        if self.text and not self.text.endswith(" "):
            if self._replace(self.text + " "):
                self.reset_input(rearm=True)

    def period(self):
        if self.text.strip() and not self.text.rstrip().endswith("."):
            if self._replace(self.text.rstrip() + ". "):
                self.reset_input(rearm=True)

    def backspace(self):
        self._replace(self.text[:-1])

    def clear(self):
        self._replace("")

    def undo(self):
        if self._history:
            self.text = self._history.pop()


class AutomaticTextComposer(TextComposer):
    """Merge independent left/right events; simultaneous events use anatomical L -> R."""

    def __init__(self, dwell=.35, release=.6, word_pause=1.4, max_chars=1000):
        super().__init__(dwell, release, max_chars)
        self.inputs = {side: LetterInput(dwell, release) for side in ("Left", "Right")}
        self.active = True
        self.word_pause = word_pause
        self._both_missing_since = self._frame_time = None

    def reset_input(self, rearm=False, side=None):
        for key in self.inputs if side is None else (side,):
            self.inputs[key].reset_input(rearm)
        self._both_missing_since = self._frame_time = None

    def update_hands(self, labels, now, present):
        if not self.active:
            return []
        if self._frame_time is not None and (now < self._frame_time or now - self._frame_time > .4):
            self.reset_input()
        self._frame_time = now
        # Unknown signs with visible hands are not a word boundary.
        if any(present.values()):
            self._both_missing_since = None
        elif self._both_missing_since is None:
            self._both_missing_since = now
        elif now - self._both_missing_since >= self.word_pause:
            self.space()
        events = []
        for side, gate in self.inputs.items():
            letter = gate.update(labels.get(side), now, present.get(side, False))
            if letter and self._replace(self.text + letter):
                events.append((side, letter))
        return events
