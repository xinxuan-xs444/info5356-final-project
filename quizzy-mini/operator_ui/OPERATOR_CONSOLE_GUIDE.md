# Operator console guide

## Launch the game

From the repository root, run this for a live session. Replace the seat-map path with the calibrated file for your setup.

```powershell
python quizzy-mini/main.py --seats path/to/seats.json
```

To rehearse without a robot, camera, microphone, or scoreboard window, run:

```powershell
python quizzy-mini/main.py --keyboard-only
```

Keep the terminal in focus and press `Enter` after each command. The console prints questions, game states, and scores. You can use the keyboard to buzz or enter an answer if the camera or speech input misses it.

## Set up players

Register four or five players. Use each seat number once, from 1 to 5. Split four players into two teams of two. For five players, put three on either team and two on the other. In a live session, the seat numbers must match the calibrated camera zones and the robot's head positions.

Enter these lines one at a time for a four-player game (team colors and players' names are examples):

```text
mode name
player 1 red Maya
player 2 red Leo
player 3 blue Sam
player 4 blue Riley
start
```

For five players, add `player 5 red Alex` before `start`. Use a five-seat camera calibration for a live game. From the repository root, run:

```powershell
cd quizzy-mini
python -m vision.demo --source reachy --players 5 --seats seats-5.json
```

With all five players visible, press `c` to calibrate and `s` to save. Quit the demo, then run `python main.py --seats seats-5.json` from the same directory. Use `python -m calibration seats` to verify the head angles for your layout, including seat 5.

Use `mode name` to address players by first name or `mode team` to address them by team. Choose the mode before `start`.

## Run a round

1. During `[THINK]`, let players discuss the question. Wait for the buzz prompt.
2. If the camera misses a hand raise, type `buzz <SEAT#>` (using the player's seat) while buzzing is open. The console accepts manual buzzes during the initial window, a steal, or the teaching buzz window. Only the opposing team can buzz on a steal.
3. Wait for the console to call on the player and show `[JUDGE]`. If speech input misses the answer, enter the number they said, such as `answer 12`. Use `correct` or `incorrect` when you need to enter your own judgment.
4. If the first answer is wrong, follow the steal prompt and wait for the other team to buzz. Use `status` to check the eligible seats and time remaining.

## Session controls

| Command | What it does |
| --- | --- |
| `status` | Show the current state, remaining time, eligible seats when available, and scores. |
| `pause` / `resume` | Freeze or resume the game timers. The game rejects buzzes and answers while paused. |
| `skip` | Move to the next question during an active, unpaused game. |
| `reset` | Clear scores and player registrations. Register everyone again before starting a new game. |
| `quit` | Exit the console and shut down the connected modules. |

If the console says a command is unavailable, run `status` and check the current phase. A buzz may have closed, or a spoken answer may already have advanced the game.
