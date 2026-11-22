"""Vision against the game-logic interface, without depending on the game branch.

FakeGame copies the parts of feature/game-logic that touch vision:
- it emits ("state", ...) before ("buzz_open", seats=...) when buzzing opens,
- main.create_game() routes buzz_open -> vision.open_buzz(seats) and
  state/reset -> vision.close_buzz(),
- vision input comes back through game.buzz(seat), which returns False when rejected,
  and must be called from the game's own loop (here: drain_queue()).
If the real game keeps this interface, these tests say vision fits it.
"""

from queue import Empty, Queue

import numpy as np
import pytest

from vision import ListSource, SeatMap, VisionService
from vision.datatypes import Frame

from .helpers import FakeClock, ScriptedEstimator, make_pose

IMG = np.zeros((600, 800, 3), dtype=np.uint8)
X = {1: 100, 2: 300, 3: 500, 4: 700}


class FakeGame:
    def __init__(self):
        self.vision = None
        self.state = "THINK"
        self.eligible: tuple[int, ...] = ()
        self.paused = False
        self.answering: int | None = None
        self.log: list[str] = []

    # wiring copied from main.create_game(vision=...)
    def emit(self, event, **data):
        if event == "buzz_open":
            self.vision.open_buzz(data["seats"])
        elif event in ("state", "reset"):
            self.vision.close_buzz()

    def open(self, state, seats):
        self.state, self.eligible = state, tuple(seats)
        self.emit("state", state=state)
        self.emit("buzz_open", seats=self.eligible)

    def buzz(self, seat: int) -> bool:
        if self.paused or self.state not in ("BUZZ_OPEN", "STEAL") or seat not in self.eligible:
            self.log.append(f"rejected {seat}")
            return False
        self.answering = seat
        self.log.append(f"accepted {seat}")
        self.state = "JUDGE"
        self.emit("state", state="JUDGE")
        return True


@pytest.fixture
def setup():
    clock = FakeClock()
    buzzes: Queue = Queue()
    vision = VisionService(ListSource([]), ScriptedEstimator([[]]), SeatMap.uniform(),
                           clock=clock, on_buzz=buzzes.put)
    game = FakeGame()
    game.vision = vision

    def run(raised: set[int], seconds: float, dt=1 / 30):
        """Camera frames + the game's main loop draining vision input."""
        for _ in range(int(round(seconds / dt))):
            vision.estimator.script = [[make_pose(X[s], raised="right" if s in raised else None)
                                        for s in (1, 2, 3, 4)]]
            vision.process_frame(Frame(IMG, clock(), 0))
            try:
                result = buzzes.get_nowait()
            except Empty:
                pass
            else:
                if not game.buzz(result.seat):
                    vision.reopen()
            clock.advance(dt)

    return game, vision, run


def test_buzz_reaches_the_game_and_window_closes(setup):
    game, vision, run = setup
    run(set(), 0.2)
    game.open("BUZZ_OPEN", (1, 2, 3, 4))
    assert vision.window_open
    run({3}, 0.5)
    assert game.answering == 3 and game.log == ["accepted 3"]
    assert not vision.window_open  # closed by the game's state event


def test_steal_only_accepts_the_other_team(setup):
    game, vision, run = setup
    run(set(), 0.2)
    game.open("STEAL", (1, 2))  # blue answered wrong; red may steal
    run({3, 4}, 0.5)
    assert game.answering is None
    run({1, 3, 4}, 0.5)
    assert game.answering == 1


def test_rejected_buzz_reopens_window(setup):
    game, vision, run = setup
    run(set(), 0.2)
    game.open("BUZZ_OPEN", (1, 2, 3, 4))
    game.paused = True
    run({2}, 0.5)  # rejected while paused; vision reopens
    assert game.log == ["rejected 2"] and vision.window_open
    game.paused = False
    run({2, 4}, 0.5)  # seat 2's hand never came down: needs a fresh raise, so seat 4 wins
    assert game.answering == 4


def test_no_buzz_after_game_closes_buzzing(setup):
    game, vision, run = setup
    run(set(), 0.2)
    game.open("BUZZ_OPEN", (1, 2, 3, 4))
    game.state = "HINT_THINK"
    game.emit("state", state="HINT_THINK")  # e.g. buzz timer ran out
    run({1}, 0.5)
    assert game.log == [] and not vision.window_open
    assert vision.reopen() is False


def test_next_window_requires_fresh_raise(setup):
    game, vision, run = setup
    run(set(), 0.2)
    game.open("BUZZ_OPEN", (1, 2, 3, 4))
    run({3}, 0.5)
    game.open("STEAL", (3, 4))  # hypothetical: seat 3 keeps the hand up into the next window
    run({3}, 0.5)
    assert game.log == ["accepted 3"]
