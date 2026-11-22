# Wiring vision into the game

For whoever connects Module 3 to the game logic once `feature/game-logic` and
`feature/vision` are both merged into `main`. Expect ~10 lines in two files.

## What already matches

`main.create_game(vision=...)` on `feature/game-logic` calls `vision.open_buzz(seats)` on
`buzz_open` and `vision.close_buzz()` on every `state` / `reset` event. `VisionService` has
both methods, so it can be passed in directly with no adapter.

| Game side | Vision side |
|---|---|
| `vision.open_buzz(seats)` | opens a window for exactly those seats (steals, fairness rule) |
| `vision.close_buzz()` | closes it |
| game owns all timers | vision windows opened this way have no timeout |
| `game.buzz(seat) -> bool` | winners arrive via `on_buzz` (a `BuzzResult`; use `.seat`) |
| buzz rejected (paused, etc.) | call `vision.reopen()` |

This is covered by `tests/vision_tests/test_game_interface.py` and was checked by hand
against the real `GameController` from `feature/game-logic`.

## Step 1: `main.py`, create vision and pass it in

```python
from queue import Queue
from vision import SeatMap, VisionService, WebcamSource, YoloPoseEstimator

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    buzzes = Queue()
    vision = VisionService(
        WebcamSource(0),                      # later: ReachyMiniSource(mini)
        YoloPoseEstimator(),
        SeatMap.load("seats.json"),           # calibrate first: python -m vision.demo, keys c then s
        on_buzz=buzzes.put,                   # called from the vision thread; a Queue is thread-safe
    ).start()
    try:
        OperatorConsole(create_game(vision=vision), vision=vision, buzzes=buzzes).run()
    except KeyboardInterrupt:
        print("\nGame console closed.")
    finally:
        vision.stop()
```

## Step 2: `operator_ui/operator_console.py`, hand buzzes to the game

The game must only be touched from its own loop, so vision results go through the queue
and the loop hands them over, the same way keyboard commands already work.

```python
from queue import Empty, Queue

class OperatorConsole:
    def __init__(self, game, vision=None, buzzes: Queue | None = None) -> None:
        self.game, self.vision, self.buzzes = game, vision, buzzes

    def run(self) -> None:
        ...
        while True:
            self.game.update()
            self.handle_vision_buzz()              # NEW
            try:
                command = commands.get(timeout=0.05)
            ...

    def handle_vision_buzz(self) -> None:          # NEW
        if self.buzzes is None:
            return
        try:
            result = self.buzzes.get_nowait()
        except Empty:
            return
        if not self.game.buzz(result.seat):        # no override: game checks eligibility
            self.vision.reopen()                   # rejected (e.g. paused): keep listening
```

The operator's `buzz SEAT` command keeps working as the manual override.

## Step 3: head position (needs Module 4)

Vision assumes the robot head faces straight ahead whenever a window is open (the camera
is in the head). The game turns the head toward the answering student (`look_at_seat`) but
never turns it back before the next `buzz_open`, for example on a steal.

The fix: on `buzz_open`, Module 4 returns the head to neutral, and vision's window opens only
once the head has arrived. Agree on the exact call with the Module 4 owner.

## Still to agree as a team

- **Seat numbering direction:** which physical side is Seat 1. The game, Module 4 head
  angles and vision's `SeatMap(seat_ids=...)` must match.
- **Settings in `config.py`:** `config.py` says all code reads parameters from it. Candidates
  to move there at integration time: camera source, seat file path, `BuzzConfig.min_hold_s`
  (0.25 s), `BuzzConfig.max_gap_s` (0.15 s), `HandRaiseRule.margin` (0.25), YOLO weights.
- **Hand-raise strictness:** head-scratches can currently count as a raise. See the team
  note on stricter rules before tuning.
