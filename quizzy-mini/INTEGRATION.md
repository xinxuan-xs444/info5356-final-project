# Wiring Module 4 (robot motion & scoreboard) into the game

For whoever merges `feature/robot-motion` once `feature/game-logic` is on `main`. Expect
~10 lines in one file (`main.py`); no operator-console changes -- robot/scoreboard are pure
output adapters, same as `terminal_output()`.

## What already matches

`main.create_game(robot=...)` already calls `robot.look_at_seat(seat)`, `robot.wiggle_antennas()`,
`robot.set_score(team, score)` and `robot.reset()` on the right events. `RobotMotion` implements
all four with that exact signature, so it drops in with no adapter:

| Game event | Robot side |
|---|---|
| `look_at_seat` (seat) | turn the head to that seat |
| `buzz_open` | wiggle both antennas ("go!") |
| `score` (scores dict) | move each team's antenna needle |
| `reset` | zero both needles, head to neutral |

Covered by `tests/robot_tests/test_robot_motion.py`'s `FakeGame`, which copies this dispatch
from the real `create_game()` on `feature/game-logic`.

## Step 1: `main.py` -- construct `RobotMotion` and pass it in, add `look_neutral()` + `on_state()`

This is the one open item from `vision/INTEGRATION.md` ("the robot must return its head to
neutral before vision's buzz window opens -- agree on the exact call with the Module 4
owner"): `robot.look_neutral()`, called on `buzz_open` *before* `vision.open_buzz(...)` /
`robot.wiggle_antennas()`. It also wires the optional `on_state()` hook (idle / celebratory /
teaching-mode animation keyed off every FSM transition) -- safe to skip if that's not wanted
yet, since `on_state()` never changes game state, only animates.

See `robot_integration.patch` in this same directory for the exact diff (git apply
--check already passed against the current `main.py`). Also update the `OperatorConsole(...)`
call if `vision`/`speech` are being wired in the same merge -- this patch only adds `robot=`.

## Step 2 (optional): scoreboard

`scoreboard.py`'s `Scoreboard` is a separate display, not part of the `robot` object. macOS
requires GUI windows to be created and driven from the main thread, so it must be
constructed on main.py's main thread and *pumped* once per loop tick from wherever that
thread already polls -- `OperatorConsole.run()`'s existing loop is the natural place
(next to its `self.game.update()` call):

```python
from scoreboard import Scoreboard

board = Scoreboard()
...
def output(event, **data):
    ...
    if event == "score":
        board.update(data["scores"])
    elif event == "reset":
        board.reset()
```

```python
# in OperatorConsole.run()'s loop, next to self.game.update():
board.pump()
```

Requires a Tkinter-enabled Python (see `requirements-robot.txt`). Check readability on the
actual monitor/tablet with `python -m scoreboard` before a session.

## Step 3: calibrate before game day

`SEAT_HEAD_ANGLES` and `ANTENNA_SCORE_ANGLES` in `config.py` are placeholders. Run:

```
python -m calibration seats
python -m calibration antennas
python -m calibration check     # final walkthrough once both are set
```

against the real robot and paste the printed values into `config.py`.

## Still to agree as a team

- **Seat numbering direction:** which physical side is Seat 1. `SEAT_HEAD_ANGLES` here, the
  game's seat numbers, and vision's `SeatMap(seat_ids=...)` must all agree -- see the same
  item in `vision/INTEGRATION.md`.
- **Sharing one Reachy Mini connection:** `RobotMotion(mini=...)` and vision's
  `ReachyMiniSource(mini=...)` can both take the same live SDK object instead of each opening
  their own -- worth doing once both are on real hardware, to avoid two processes fighting
  over one device.
- **`config.py` consolidation:** candidates already here -- `SEAT_HEAD_ANGLES`,
  `HEAD_NEUTRAL_ANGLE`, `ANTENNA_SCORE_ANGLES`, safety ranges, `IDLE_ANIMATION_INTERVAL`.
