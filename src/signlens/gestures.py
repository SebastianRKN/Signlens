"""Conservative open-palm lateral push command; separate from letter classification."""
from collections import deque
import numpy as np


class PushToErase:
    def __init__(self):
        self.paths = {s: deque() for s in ('Left', 'Right')}
        self.cooldown = 0.0

    def reset(self, side=None):
        for s in self.paths if side is None else (side,):
            self.paths[s].clear()

    @staticmethod
    def open_palm(hand):
        if hand is None or hand.stale or not np.isfinite(hand.world).all():
            return False
        straight = 0
        for a, b, c in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
            u, v = hand.world[a] - hand.world[b], hand.world[c] - hand.world[b]
            norm = np.linalg.norm(u) * np.linalg.norm(v)
            if norm > 1e-9 and np.dot(u, v) / norm < -.72:
                straight += 1
        return straight >= 3

    def update(self, hands, now, aspect=4/3):
        blocked, event = set(), False
        for side, path in self.paths.items():
            hand = hands.get(side)
            if now < self.cooldown:
                blocked.add(side)
                path.clear()
                continue
            if not self.open_palm(hand):
                path.clear()
                continue
            center = hand.pts[[0, 5, 9, 13, 17], :2].mean(axis=0)
            if not np.isfinite(center).all():
                path.clear()
                continue
            if path and (now <= path[-1][0] or now - path[-1][0] > .25):
                path.clear()
            path.append((now, float(center[0]), float(center[1])))
            while path and now - path[0][0] > 1.1:
                path.popleft()
            if len(path) < 3:
                blocked.add(side)
                continue
            start = path[0]
            duration = now - start[0]
            dx = center[0] - start[1]
            dy = (center[1] - start[2]) / max(aspect, .1)
            # In either mirrored view, move from the starting side across the body.
            direction = 1 if start[1] < .5 else -1
            travel = sum(abs(b[1] - a[1]) for a, b in zip(path, list(path)[1:]))
            moving = abs(dx) > .035
            if duration < .65 or moving:
                blocked.add(side)
            if (.12 <= duration <= 1.1 and direction * dx >= .20
                    and abs(dx) > 1.8 * abs(dy) and abs(dx) / max(travel, .001) >= .8
                    and abs(dx) / duration >= .25):
                event = True
        if event:
            self.cooldown = now + 1.0
            self.reset()
            blocked.update(self.paths)
        return event, blocked
